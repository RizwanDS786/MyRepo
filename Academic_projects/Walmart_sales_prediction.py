import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score

estimator = GradientBoostingRegressor(loss="huber")

# Function for the feauture preperation
def featureprep(train, test , dropaxis , splitset):
    # Function for getting labeled dummies
    if splitset == True:
        def xdums(df):
            dums = pd.get_dummies(pd.to_datetime(df["Date"], format="%Y-%m-%d").dt.week)
            dums.columns = map(lambda x: "Week_" + str(x), dums.columns.values)

            return dums
    else:
        def xdums(df):
            dums = pd.get_dummies(df["Store"])
            dums = dums.set_index(df.index)
            dums.columns = map(lambda x: "Store_" + str(x), dums.columns.values)
            out = dums
            dums = pd.get_dummies(df["Dept"])
            dums.columns = map(lambda x: "Dept_" + str(x), dums.columns.values)
            out = out.join(dums)
            dums = pd.get_dummies(df["Type"])
            dums.columns = map(lambda x: "Type_" + str(x), dums.columns.values)
            out = out.join(dums)
            dums = pd.get_dummies(pd.to_datetime(df["Date"], format="%Y-%m-%d").dt.week)
            dums.columns = map(lambda x: "Week_" + str(x), dums.columns.values)
            out = out.join(dums)
            print("values of out {}".format(out.head()))
            return out

    train_x = xdums(train).join(train[["IsHoliday", "Size", "Year", "Day", "Days to Next Christmas"]])
    test_x = xdums(test).join(test[["IsHoliday", "Size", "Year", "Day", "Days to Next Christmas"]])

    # Deal with NAs
    train_x = train_x.dropna(axis=dropaxis)
    test_x = test_x.dropna(axis=dropaxis)
    train_y = train.dropna(axis=dropaxis)["Weekly_Sales"]

    # Remove any train features that aren't in the test features
    for feature in train_x.columns.values:
        if feature not in test_x.columns.values:
            train_x = train_x.drop(feature, axis=1)

    # Remove any test features that aren't in the train features
    for feature in test_x.columns.values:
        if feature not in train_x.columns.values:
            test_x = test_x.drop(feature, axis=1)
    # print("values of train_x : {}" .format(train_x.head()))
    # print("values of test_x : {}" .format(test_x.head()))
    # print("values of train_y : {}" .format(train_y.head()))
    return train_x, train_y, test_x


# Function for returning estimates
def estimates(train, test, splitset):

    # Define estimator

    train_x, train_y, test_x = featureprep(train, test, 1, splitset)

    # Get estimates for columns that have no NAs
    estimator.fit(train_x, train_y)
    out = pd.DataFrame(index=test_x.index)
    out["Weekly_Sales"] = estimator.predict(test_x)
    out["Id"] = out.index

    # Create a dataframe for plotting the training feature regression
    plot = pd.DataFrame(index=train_x.index)
    plot["Weekly_Sales"] = train_y
    plot["Weekly_Predicts"] = estimator.predict(train_x)
    plot["Date"] = plot.index.str.split("_").str[-1]
    plot = plot.groupby("Date")[["Weekly_Sales", "Weekly_Predicts"]].sum()
    print("values : {}" .format(plot))
    return out, plot


def plot_pred(splot):
    print(splot.columns.values)
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Predicted sales", color=color)
    ax1.plot(splot[["Date"]].values, splot.Weekly_Predicts, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel("Original sales", color=color)  # we already handled the x-label with ax1
    ax2.plot(splot[["Date"]].values, splot.Weekly_Sales, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def main():
    # Read in dataframes
    print("Reading and merging the datasets...")
    train = pd.read_csv("E:\\Project\\train1.csv")
    test = pd.read_csv("E:\\Project\\test1.csv")

    # Merge the train and test data sets with stores data set
    stores = pd.read_csv("E:\\Project\\stores.csv")
    train = train.merge(stores, how='left', on='Store')
    test = test.merge(stores, how='left', on='Store')

    train["Id"] = train["Store"].astype(str) + "_" + train["Dept"].astype(str) + "_" + train["Date"].astype(str)
    train = train.set_index("Id")
    test["Id"] = test["Store"].astype(str) + "_" + test["Dept"].astype(str) + "_" + test["Date"].astype(str)
    test = test.set_index("Id")

    # Also make an index by store_dept to split up the dataset
    train["Index"] = train["Store"].astype(str) + "_" + train["Dept"].astype(str)
    test["Index"] = test["Store"].astype(str) + "_" + test["Dept"].astype(str)

    # Add column to fetch year from the date
    train["Year"] = pd.to_datetime(train["Date"], format="%Y-%m-%d").dt.year
    test["Year"] = pd.to_datetime(test["Date"], format="%Y-%m-%d").dt.year

    # Add column to fetch day from date
    train["Day"] = pd.to_datetime(train["Date"], format="%Y-%m-%d").dt.day
    test["Day"] = pd.to_datetime(test["Date"], format="%Y-%m-%d").dt.day

    # Add column for days to next Christmas which calculates the days remaining for next christmas

    train["Days to Next Christmas"] = (pd.to_datetime(train["Year"].astype(str) + "-12-31", format="%Y-%m-%d") -
                                       pd.to_datetime(train["Date"], format="%Y-%m-%d")).dt.days.astype(int)
    test["Days to Next Christmas"] = (pd.to_datetime(test["Year"].astype(str) + "-12-31", format="%Y-%m-%d") -
                                      pd.to_datetime(test["Date"], format="%Y-%m-%d")).dt.days.astype(int)

    # Create store_dept dictionaries for train and test data sets so that we can subset the data.

    print("Splitting the datasets into subsets...")
    traindict = {}
    testdict = {}
    for index in set(test["Index"].tolist()):
        traindict[index] = train[train["Index"] == index]
        testdict[index] = test[test["Index"] == index]

# Run the individual store-departments models
    out = pd.DataFrame()
    plot = pd.DataFrame()
    count = 0
    for key in testdict.keys():
        count += 1
        try:
            ot, pt = estimates(traindict[key], testdict[key], True)
            out = pd.concat([out, ot])
            plot = pd.concat([plot, pt])
        except:
            print("No training data available for {}".format(key))

        # Initial analysis of dataframe

    plt.plot(train.iloc[1:100, 2], train.iloc[1:100, 3])
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.show()

    plt.plot(train.iloc[1:100, 10], train.iloc[1:100, 3])
    plt.xlabel("Days to next christmas")
    plt.ylabel("Sales")
    plt.show()

    plt.plot(train.iloc[1:100, 8], train.iloc[1:100, 3])
    plt.xlabel("year")
    plt.ylabel("Sales")
    plt.show()

    # Run a model of all the data to fill in anything that was NA
    print("Creating giant model to fill in for the missing data")
    sout, splot = estimates(train, test, False)
    sout = sout.join(out, how="left", lsuffix="_Backup")
    # print("values of sout before fillna {}" .format(sout.head()))
    sout["Weekly_Sales"] = sout["Weekly_Sales"].fillna(sout["Weekly_Sales_Backup"])
    # print("values of sout after fillna {}" .format(sout.head()))

    # Format for submission
    sout["Id"] = sout["Id"].fillna(sout["Id_Backup"])
    sout = sout.drop(["Weekly_Sales_Backup", "Id_Backup"], axis=1)
    splot = splot.join(plot, how="left", lsuffix="_Backup")
    splot["Weekly_Sales"] = splot["Weekly_Sales"].fillna(splot["Weekly_Sales_Backup"])
    splot["Weekly_Predicts"] = splot["Weekly_Predicts"].fillna("Weekly_Predicts_Backup")
    splot = splot.drop(["Weekly_Sales_Backup", "Weekly_Predicts_Backup"], axis=1)

    sout.to_csv("E:\\Project\\Sales_predict_data3.csv", index=False)

    # Save the plotting file for plotting later
    sout["Date"] = sout.index.str.split("_").str[-1]
    plot = sout.groupby("Date")[["Weekly_Sales"]].sum()
    plot["Weekly_Predicts"] = plot["Weekly_Sales"]
    plot = plot.drop("Weekly_Sales", axis=1)
    splot = splot.append(plot)
    splot = splot.reset_index().groupby("Date")[["Weekly_Sales", "Weekly_Predicts"]].sum()
    splot.to_csv("E:\\Project\\plotting_data3.csv")
    print("values of splot are : {}" .format(splot.head()))
    splot = pd.read_csv("E:\\Project\\plotting_data3.csv")
    splot['Date'] = pd.to_datetime(splot['Date'])
    plot_pred(splot)


if __name__ == "__main__" :
    main()
