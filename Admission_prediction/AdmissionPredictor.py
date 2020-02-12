# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 00:29:22 2019

@author: harsh
"""

#Problem: preidction whether a student gets admit or not by using various labels
import pandas as pd

#Importing the csv file
df=pd.read_csv("graduate-admissions\\Admission_Predict.csv")

#Inspecting  the labels of the file
df.columns

df.shape

#Descriptive statistics and plotting the  visuslzations to dig deep into the data which helps us to see 
#if we need to apply any transformations or imputation techniques
import seaborn as sns
import matplotlib.pyplot as plt

#plot to show if any of the data is null
def data_clean():
    sns.heatmap(df.isnull())
data_clean()

# Data is clean - No missing data
df.columns
#See whats the mean scores of GRE,TOFEL and avg rating of the college
def desc():
    print(df.describe())
desc()

#Rating vs Count
def Rating_Count():
    plt.title("Count of applicants on verious levels of universities")
    sns.countplot(df['University Rating'])
    plt.show()
Rating_Count()
#Most of the universities are having an average rating of 3


#GRE score VS Chance of being admitted
#Relation between GRE Score and chnace of being admitted
def GRE_Admit():
    plt.title("GRE Score VS Chance Of Admit")
    sns.scatterplot(df['GRE Score'],df['Chance of Admit '])
    plt.show()
GRE_Admit()
#The plot shows that the chance of being admitted increases with increase in the GRE score

def CGPA_LOR():
    fig = sns.lmplot(x="CGPA", y="LOR ", data=df, hue="Research")
    plt.title("CGPA VS LOR")
    plt.show()
CGPA_LOR()


def Research_Count():
    plt.figure(figsize=(10,6))
    sns.countplot(df['Research'])
    plt.grid(alpha=0.5)
    plt.xlabel('Students')
    plt.show()
Research_Count()


#Pairplot tha can help us to identify the relation between one varibale and remiaing varibales
#Pairplot that shows the relatio between the various varibales
def pairplot():
    sns.pairplot(df)
pairplot()


plt.figure(figsize=(20,6))
plt.subplot(1,2,1)
sns.distplot(df['TOEFL Score'])
plt.title('Distributed TOEFL Scores of Applicants')


plt.figure(figsize=(20,6))
sns.distplot(df['CGPA'])
plt.title('CGPA Distribution of Applicants')

plt.figure(figsize=(20,6))
sns.regplot(df['CGPA'], df['Chance of Admit '])
plt.title('CGPA vs Chance of Admit')
plt.show()

df.columns

#Dorpping the unwanted column from the data
df=df.drop(['Serial No.'],axis=1)



#Logistc regression 
#For the purpose of prediction I have assumed that if the chance of admit is 
#greater than mean than he will be getting admitted 
#Else he will not be getting admitted
#the above assumption helps to classify the data into 0 and 1
mean_chance_of_admit=df['Chance of Admit '].mean()
print(mean_chance_of_admit)


def converting_binary(df):
    for i in range(0,len(df['Chance of Admit '])):
        if(df['Chance of Admit '][i]>mean_chance_of_admit):
            df['Chance of Admit '][i]=1
        else:
            df['Chance of Admit '][i]=0
    return df
new_df=converting_binary(df)

new_df.columns

#Preparing the test and train data by using the new_df for the prediction purpose
#predicted varibale is Chance of being admitted and reamining varibales will
#be the features for predicting the admission
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#As the university reating is a catgeroical variable it needs to
#be imputed using One Hot encoding
new_df['University Rating']

df_university_rating=pd.get_dummies(df['University Rating'],drop_first=True)

new_df=pd.concat([new_df,df_university_rating],axis=1)


#Drop the actual categorical colum - university rating
new_df=new_df.drop(['University Rating'],axis=1)
new_df.columns


#Preparing the frature varibales and the target variable
X=new_df.drop(['Chance of Admit '],axis=1)
X.columns

#Predicted variables
y=df['Chance of Admit ']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

new_df.head()


lr=LogisticRegression()
lr.fit(X_train,y_train)


y_pred = lr.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(lr.score(X_test, y_test)))


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

#From the confusion matrix we can say that the model is able to predict 70+59 correct predictions


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


#Testing the algorithm by conducting the PCA on the new dataset

new_df.columns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#Preprocessing helps us to remove the variance among the data 
#{GRE Score is calcuated for a scale of 340 and TOFEL is calucated for 1600
#which causes the data to have high variance}
#I am using standradizing the data such that there is no hight variance between various labels
sc = StandardScaler()  
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test)  

pca=PCA()
X_train = pca.fit_transform(X_train)  
X_test = pca.transform(X_test)  

variance_explained=pca.explained_variance_ratio_


#from the variance_explained we can see that GRE score and TOFEL score and sop are able
#to explain the more than 60% of the variance asscoiated with the data

#Lets see the how the Logistic regression algorithm perform under the various principle componenrs

pca = PCA(n_components=1)  
X_train = pca.fit_transform(X_train)  
X_test = pca.transform(X_test)  

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,y_train)




y_pred = lr.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(lr.score(X_test, y_test)))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

#From the confusion matrix we can say that the model is able to predict 70+59 correct predictions


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


#Increassing the no of components to 2

X=new_df.drop(['Chance of Admit '],axis=1)
X.columns

#Predicted variables
y=df['Chance of Admit ']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)


pca = PCA(n_components=2)  
X_train = pca.fit_transform(X_train)  
X_test = pca.transform(X_test)  

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(lr.score(X_test, y_test)))

#While using the 2 prinicple components the accuracy decreases from 89% to 83%


#Lets test the algorithm with 3 principle components

X=new_df.drop(['Chance of Admit '],axis=1)
X.columns

#Predicted variables
y=df['Chance of Admit ']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)


pca = PCA(n_components=3)  
X_train = pca.fit_transform(X_train)  
X_test = pca.transform(X_test)  

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(lr.score(X_test, y_test)))


#With 3 principle components the accuracy increase to 86%

#Plotting the number of componenst to be explained
import numpy as np
pca = PCA().fit(new_df)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');


#From the graph we can say that 1 lable {gre score} is sufficient to sepearate the data
#Hence GRE score is sufficient to classifiy the admission prediction
#Which is already eveident from the abobe accuracies
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))




from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=110,max_depth=3,random_state=0)
forest.fit(X_train, y_train)
y_predict = forest.predict(X_test)
forest_score = (forest.score(X_test, y_test))*100




from sklearn.ensemble import RandomForestClassifier
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

forest = RandomForestClassifier(n_estimators=110,max_depth=6,random_state=0)
forest.fit(X_train, y_train)
y_predict = forest.predict(X_test)
forest_score = (forest.score(X_test, y_test))*100

from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test, y_predict)
print(conf_mat)
