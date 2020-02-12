# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 16:20:13 2019

@author: harsh
"""

import pandas as pd

df=pd.read_csv("C:\\Users\\harsh\\OneDrive\\Desktop\\TripAdvisor\\Restuarents_Data\\Resturanet_Info.csv")

def Cleaning(df):
    for i in range(0,len(df['Cusines'])):
        try:
            df['Cusines'][i]=df['Cusines'][i].split(',')
        except:
            df['Cusines'][i]=df['Cusines'][i]


    for i in range(0,len(df['Cusines'])):
        try:
            for j in range(0,len(df['Cusines'][i])):
                maps={}
                cusine=df['Cusines'][i][j]
                maps.update({cusine:df['Restuarent_ID'][i]})
                new_df=pd.DataFrame.from_dict(maps,orient='index')
                with open("C:\\Users\\harsh\\OneDrive\\Desktop\\TripAdvisor\\Restuarents_Data\\Cusines.csv",'a') as f:
                    new_df.to_csv(f,sep=",",encoding="utf-8",header=False)
                f.close()   
        except:
            print("NAN")
        return f
cleaned_colums=Cleaning(df)

for i in range(0,len(df['Reviews_Count'])):
    if(df['Reviews_Count'][i]=='1 review'):
        df['Reviews_Count'][i]=df['Reviews_Count'][i].split(' ')[0]