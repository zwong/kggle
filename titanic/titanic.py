import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from shared import getDataSet, loadDataSet, findBlankColumns


#Kaggle Constants
KAGGLE_COMPETITION = "titanic"
PROJECT_NAME = "titanic"

#getDataSet(KAGGLE_COMPETITION, PROJECT_NAME)

train_raw, test_raw = loadDataSet(PROJECT_NAME, indexName="PassengerId")

train_set = train_raw.copy()
train_set_Y = train_raw["Survived"]

#First Check Blank Column
findBlankColumns(train_set)
#Todo: Cabin is 77% blank. Remove?

#Impute values
from sklearn.impute import SimpleImputer

#Age <-- temporarily take median
impute = SimpleImputer(strategy="median")
train_set["Age"] = impute.fit_transform(train_set["Age"].values.reshape(-1,1))

#Cabin <-- remove?
blank = train_set["Cabin"].isna()
cabin = train_set[~blank].copy()

split = cabin["Cabin"].str.extract(pat="([A-Za-z]+)([0-9]*).*")
cabin["CabinClass"] = split[0]
cabin["CabinNumber"] = split[1]

cabin["CabinNumber"].fillna(value=-1, inplace=True)
cabin["CabinNumber"] = cabin["CabinNumber"].astype(int)

#Todo: do cabin numbers have any correlation with surviving?
#--> Seems like higher number survive
g = sns.FacetGrid(data=cabin, row="Survived", col="Pclass")
g.map(plt.hist, "CabinNumber", alpha=0.4)


#Todo: Can we infer embarkation based on PClass, Parch, SibSp, Fare? ==> Does this matter?
#Embarked
impute = SimpleImputer(strategy="most_frequent")
train_set["Embarked"] = impute.fit_transform(train_set["Embarked"].values.reshape(-1,1))

#Todo: Name Salutation
name = train_set["Name"].str.extract(pat="(.+), (.+)\. (.*)")


#Graphs
import seaborn as sns

#which classes belon to which Pclass - this is misleading because 77% don't have cabins
g = sns.FacetGrid(data=train_set.loc[cabin.index], hue="Pclass", sharey=False)
g.map(plt.hist, "CabinClass", alpha=0.5)
g.add_legend()

#Any relationship with cabin number and survival?
g = sns.FacetGrid(data=train_set.loc[cabin.index], row="Pclass", hue="Survived", sharey=False)
g.map(plt.hist, "CabinNumber", alpha=0.5)
g.add_legend()

#which class and how much they paid
g = sns.FacetGrid(data=train_set, hue="Survived", row="Pclass", sharex=False)
g.map(plt.hist, "Fare", alpha=0.5)
g.add_legend()

#Let's see where the different PClass embarked from
g = sns.FacetGrid(data=train_set, row="Pclass", sharey=False)
g.map(plt.hist, "Embarked", alpha=0.5, stacked=True)
g.add_legend()


g = sns.FacetGrid(data=train_set[~blank], hue="Pclass")
g.map(plt.hist, "Embarked", alpha=0.5)
g.add_legend()

g = sns.FacetGrid(hue="Survived", data=train_set)
g.map(plt.hist, "Age", alpha=.5)

g = sns.FacetGrid(hue="Survived", data=train_set, legend_out=True)
g.map(plt.hist, "Sex", alpha=.5)

g = sns.FacetGrid(data=train_set, col="Embarked", row="Sex", hue="Survived")
g.map(plt.hist, "Age", alpha=0.5)

g = sns.FacetGrid(data=train_set, hue="Survived")
g.map(plt.hist, "Pclass", alpha=0.5)

plt.hist(train_set.loc[train_set["Survived"] == 1, "Pclass"], stacked=True, label="Survived", alpha=0.5, color="blue")
plt.hist(train_set.loc[train_set["Survived"] == 0, "Pclass"], stacked=True, label="Not Survived", alpha=0.5, color="green")
plt.legend()

sns.violinplot(x="Parch", y="SibSp", hue="Survived",data=train_set,split=True, scale="count")


plt.hist(train_set.loc[train_set["Survived"] == 1, "SibSp"], stacked=True, label="Survived", alpha=0.5, color="blue")
plt.hist(train_set.loc[train_set["Survived"] == 0, "SibSp"], stacked=True, label="Not Survived", alpha=0.5, color="green")
plt.legend()

plt.hist(train_set.loc[train_set["Survived"] == 1, "Parch"], stacked=True, label="Survived", alpha=0.5, color="blue")
plt.hist(train_set.loc[train_set["Survived"] == 0, "Parch"], stacked=True, label="Not Survived", alpha=0.5, color="green")
plt.legend()

#Todo: Parch can determine adult or not? 20% age is blank

sns.kdeplot(data=x)

# Columns to add
# Age Group
labels = ["{0} - {1}".format(i, i+9) for i in range(0,100,10)]
train_set["AgeGroup"] = pd.cut(train_set["Age"], bins=range(0,110,10), labels=labels)

#Salutations
#Last Names to tie relationshi
train_set["Salutation"] = train_set["Name"].str.extract(pat="^.+,(.+)\.")
train_set["Embarked"].astype("category")

g = sns.FacetGrid(hue="Survived", col="Sex", row="Pclass", data=train_set)
g.map(plt.hist, "Salutation")

train_set.hist(column="Salutation", by=["Sex", "Pclass","Survived"], layout=(2,6), xlabelsize=5, ylabelsize=5, sharey=True)


train_set["Embarked"].fillna(value="Unknown", inplace=True)
train_set["Age"].fillna(value=-1, inplace=True)
train_set.hist(column="Age", by=["Survived", "Embarked"], sharey=True, layout=(3,3))


g = sns.FacetGrid(data=train_set,hue="Survived",col="Embarked")
g.map(plt.hist, "Age", alpha=.8)

plt.show()
#Survived <- 60