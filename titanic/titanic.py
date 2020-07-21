import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

from shared import getDataSet, loadDataSet, findBlankColumns, writeKagglePrediction
#pd.set_option('display.max_columns',10)

#MAC or PC
OS = "MAC"

#Kaggle Constants
KAGGLE_COMPETITION = "titanic"
PROJECT_NAME = "titanic"

#getDataSet(KAGGLE_COMPETITION, PROJECT_NAME)

train_raw, test_raw = loadDataSet(PROJECT_NAME, indexName="PassengerId")

train_set = train_raw.copy()

###########################
# DATA FEATURE TRANSFORM
###########################
#Provide Titanic Train Set and returns new features

def transformDataset(train_set):
    #First Check Blank Column
    findBlankColumns(train_set)


    #Cabin handling
    blank = train_set["Cabin"].isna()
    cabin = train_set[~blank].copy()

    split = cabin["Cabin"].str.extract(pat="([A-Za-z]+)([0-9]*).*")
    cabin["CabinClass"] = split[0]
    cabin["CabinNumber"] = split[1]

    cabin["CabinNumber"].fillna(value=-1, inplace=True)
    cabin.loc[ (cabin["CabinNumber"] == ""), "CabinNumber"] = -1
    cabin["CabinNumber"] = cabin["CabinNumber"].astype(int)

    # Replace Cabin # using Ticket as lookup
    cabtick = train_set[~train_set["Cabin"].isna()][["Cabin","Ticket"]]
    for x in cabtick.index:
        #print("Row:", x)
        y = (train_set["Ticket"].where(train_set["Ticket"] == train_set.loc[x]["Ticket"]))
        y.dropna(inplace=True)
        # print("Rows with same ticket number: " , y.index)
        train_set.loc[y.index, "Cabin"] = train_set.loc[x]["Cabin"]
        # print("Setting index", y.index, "with value ", train_set.loc[x]["Cabin"] )


    #Embarked
    from sklearn.impute import SimpleImputer
    impute = SimpleImputer(strategy="most_frequent")
    train_set["Embarked"] = impute.fit_transform(train_set["Embarked"].values.reshape(-1,1))


    #Name Salutation
    name = train_set["Name"].str.extract(pat="(.+),\s(.+?)\.(.*)")
    train_set["LastName"] = name[0]
    train_set["Salutation"] = name[1]
    train_set["FirstName"] = name[2]

    #Cutdown # of Salutations
    train_set.loc[~train_set["Salutation"].isin(["Mr", "Miss", "Mrs", "Master", "Dr", "Rev"]), "Salutation"] = "Other"

    #Note: Last name doens't help. Could be same last name. More important is the Parch/SibSp numbers or Ticket is same
    #train_set[train_set["FirstName"].str.find(sub="(") > 0][["FirstName","LastName", "Salutation","Parch","SibSp","Age"]]
    train_set[train_set["FirstName"].str.find(sub="(") > 0]


    #For people traveling together (i.e on same ticket), what are survival rates?
    #Note: More likely to NOT survive the greater the number of people traveling together
    #Note: There is an anomaly for large groups that traveled together. A Chinese group.  Is this just an anomally? Assuming it is
    ticketcount = ((train_set["Ticket"].value_counts() > 1) == True)
    ticket = ticketcount[ticketcount==True].index.to_series()
    ticketMultiPerson = train_set[train_set["Ticket"].isin(ticket)]

    if((train_set.columns == "Survived").sum() > 0):
        survive = ticketMultiPerson.pivot_table(index="Ticket", values="Survived", aggfunc=['count', 'sum'])
        survive.columns = survive.columns.get_level_values(0)
        ticketMultiPerson = ticketMultiPerson.merge(right=survive, how="left", on="Ticket")
        ticketMultiPerson.rename(columns={"sum": "FamilySurvived", "count": "FamilySize"}, inplace=True)
        ticketMultiPerson[(ticketMultiPerson["Pclass"] > 0) & (ticketMultiPerson["FamilySize"]>4)][["FirstName","Age","SibSp","Parch","Ticket", "Survived"]].sort_values(by="Ticket")



    #Parch / SibSp Relationship
    #Note:
    #      If you have Parch > 2 , you likely won't survive
    #      If your SibSp > 2, you likely won't survive
    #      Parch <= 3, survived. Could this be because if youre Parch = 3, you're a kid.


    #Age Relationship
    #Todo: Age is missing for 20% of the fields. Based off salutation, identify whether adult or child
    #          Without predicting an age, is there a different way to identify survivability?
    #      Identifier for who is likely a Child? is there a relationship with Parch and SibSp?
    #Note: Assumption - Parch > 2 or SibSp > 1 ==> Likely a Child.

    #If Parch <= 2 or SibSp <= 0 ==> How do we identify whether a child or not? Salutation and Ticket
    #Master <-- for male child
    # train_set[train_set["Salutation"] == "Master"]["Age"].plot.hist()



    #Note:Master used to address male children

    #Logic
    ADULT_AGE = 12

    #Marital Status (Married/Not Married/Unknown)
    train_set["MaritalStatus"] = np.nan
    train_set.loc[train_set["Age"] < ADULT_AGE, "MaritalStatus"] = "Not Married"
    train_set.loc[train_set["Salutation"] == "Mrs", "MaritalStatus"] = "Married"
    train_set.loc[train_set["Salutation"] == "Miss", "MaritalStatus"] = "Not Married"
    train_set.loc[train_set["Salutation"] == "Master", "MaritalStatus"] = "Not Married"
    train_set.loc[(train_set["Sex"] == "male") & (train_set["Age"] > ADULT_AGE) & (train_set["SibSp"] == 1), "MaritalStatus"] = "Married"
    train_set.loc[(train_set["Sex"] == "male") & (train_set["Age"] > ADULT_AGE) & (train_set["SibSp"] == 0), "MaritalStatus"] = "Not Married"
    train_set.loc[(train_set["Sex"] == "female") & (train_set["Age"] > ADULT_AGE) & (train_set["Parch"] > 0), "MaritalStatus"] = "Not Married"
    train_set.loc[(train_set["Parch"] > 2), "MaritalStatus"] = "Not Married"
    train_set.loc[(train_set["SibSp"] > 1), "MaritalStatus"] = "Not Married"
    train_set.loc[(train_set["SibSp"] == 0) & (train_set["Parch"] == 0), "MaritalStatus"] = "Not Married"



    #train_set.loc[(train_set["Age"] > ADULT_AGE) & (train_set[""]), "MaritalStatus"] = "Not Married"


    #Person Type
    train_set["PersonType"] = np.nan
    train_set.loc[ train_set["Salutation"] == "Mrs", "PersonType"] = "Adult"
    train_set.loc[train_set["Salutation"] == "Mrs", "PersonType"] = "Adult"
    train_set.loc[(train_set["Salutation"] == "Master"), "PersonType"] = "Child"
    train_set.loc[(train_set["SibSp"] > 1 ) & (train_set["Parch"] <= 2), "PersonType"] = "Child"
    train_set.loc[(train_set["MaritalStatus"] == "Married"), "PersonType"] = "Adult"

    train_set.loc[(train_set["Age"] < ADULT_AGE), "PersonType"] = "Child"
    train_set.loc[(train_set["Age"] >= ADULT_AGE), "PersonType"] = "Adult"



    #train_set.loc[(train_set["Age"] > ADULT_AGE) & (train_set["Parch"] > 0), "MaritalStatus"] = "Married"

    train_set["GroupSize"] = train_set["Parch"] + train_set["SibSp"]

    train_set.loc[ train_set["GroupSize"] >= 6, "GroupType"] = "LargeGroup"
    train_set.loc[ train_set["GroupSize"] < 6, "GroupType"] = "SmallGroup"
    train_set.loc[ train_set["GroupSize"] == 0, "GroupType"] = "Solo"


    # Columns to add
    # Age Group
    labels = ["{0} - {1}".format(i, i+9) for i in range(0,100,10)]
    train_set["AgeGroup"] = pd.cut(train_set["Age"], bins=range(0,110,10), labels=labels).to_list()
    train_set["AgeGroup"].fillna("Unknown", inplace=True)


    train_set["Embarked"].fillna(value="Unknown", inplace=True)
    train_set["Age"].fillna(value=-1, inplace=True)

    # if (train_set.columns == "Survived").sum() > 0:
    #     train_set.drop(labels=["Survived"], axis=1, inplace=True)




    return train_set

##################
## Data Process
##################
#Receive data set
#OntHotEncodes classifications and imputes missing values
#Returns cleaned dataset

def dataClean(train_set):
    #drop columns that won't be used for training - Name/FirstName/LastName
    train_transform = train_set.copy(deep=True)

    train_str_col = train_transform.columns[(train_transform.dtypes != float) & (train_transform.dtypes != int)]
    train_num_col = train_transform.columns[(train_transform.dtypes == float) | (train_transform.dtypes == int)]
    train_num = pd.DataFrame(dtype=int)
    train_cat = pd.DataFrame()
    train_cat_corr = pd.DataFrame()


    #train categories - go through each column and split OneHotEncode all categories
    for col in list(train_str_col):
        print("Column: ", col)
        train_raw_col = train_transform[col].copy(deep=True)
        #Check for blanks
        imputer = SimpleImputer(strategy="constant", fill_value="Unknown")

        train_raw_col_imputed = imputer.fit_transform(train_raw_col.values.reshape(-1, 1))

        #numerate the categories
        ord_enc = OrdinalEncoder(categories='auto')
        ord_res = ord_enc.fit_transform(train_raw_col_imputed)
        #print("OrdinalEncoder shape: ", ord_res.shape)
        #print("OrdinalEncoder shape: ", ord_enc.categories_)

        #separate category into columns
        onehot_enc = OneHotEncoder(categories='auto')
        onehot_res = onehot_enc.fit_transform(ord_res)
        #print("OneHotEncoder shape: ", onehot_res.shape)

        #prefix column names with original  column
        #print("Categories (", len(ord_enc.categories_[0]), "): " , ord_enc.categories_[0])
        col_names = [col + "-" + str(category) for category in ord_enc.categories_[0]]
        #print("Created column names:", col_names)

        #put results back into dataframe with column names
        onehot_res_df = pd.DataFrame(data=onehot_res.toarray(), columns=col_names)

        #add results into train_cat to save results
        train_cat = pd.concat([train_cat, onehot_res_df], axis=1)

    #Impute the number categories too
    findBlankColumns(train_transform[train_num_col])
    train_num = train_transform[train_num_col].copy(deep=True)
    imputer = SimpleImputer(strategy="constant", fill_value=-1)
    train_num_array = imputer.fit_transform(train_num)
    train_num = pd.DataFrame(data=train_num_array, columns=train_num_col)

    #Combine
    train_processed = train_cat.merge(train_num, how="inner", left_index=True, right_index=True)

    #Feature Scale
    from sklearn.preprocessing import StandardScaler

    scale = StandardScaler()
    train_processed = scale.fit_transform(train_processed)

    return train_processed

def getScore(results):
    scores = results
    print("Scores:" , scores)
    print("Mean:", np.mean(scores))
    print("STD:", np.std(scores))


print("Cleaning train dataset...")
x = transformDataset(train_set)

x.drop(labels=["Name", "Age", "FirstName", "LastName", "Ticket", "Cabin", "SibSp","Parch","GroupSize"], axis=1, inplace=True)
if "Survived" in train_set.columns:
    train_Y = train_set["Survived"]
    train_set.drop("Survived", axis=1, inplace=True)
else:
    print("Survived feature does not exist")


train_processed  = dataClean(x)
print("Cleaning train dataset complete")

#Train Test Split
from sklearn.model_selection import train_test_split

train_split, test_split, trainY_split, testY_split = train_test_split(train_processed, train_Y, random_state=42, test_size=.2)

#Predictor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score



print("Logistic Regression")
logreg = LogisticRegression(solver="lbfgs")
logreg.fit(X=train_split, y=trainY_split)
pred = logreg.predict(test_split)
print("Accuracy: ", accuracy_score(testY_split, pred))
crossval = cross_val_score(logreg, train_processed, train_Y, scoring="accuracy", cv=10)
getScore(crossval)

print("SVM")
svc = SVC(gamma="auto")
svc.fit(X=train_split, y=trainY_split)
pred = svc.predict(train_processed)
print("Accuracy: ", accuracy_score(train_Y, pred))
crossval = cross_val_score(svc, test_split, testY_split, scoring="f1", cv=10)
getScore(crossval)

print("Decision Tree")
dtc = DecisionTreeClassifier()
dtc.fit(X=train_split, y=trainY_split)
pred = dtc.predict(train_processed)
print("Accuracy: ", accuracy_score(train_Y, pred))
crossval = cross_val_score(dtc, test_split, testY_split, scoring="f1", cv=10)
getScore(crossval)


print("Extra Tree Classifier")
etc = ExtraTreeClassifier()
etc.fit(X=train_split, y=trainY_split)
pred = etc.predict(train_processed)
print("Accuracy: ", accuracy_score(train_Y, pred))
crossval = cross_val_score(etc, test_split, testY_split, scoring="f1", cv=10)
getScore(crossval)


print("Grandient Boost")
gbc = GradientBoostingClassifier()
gbc.fit(X=train_split, y=trainY_split)
pred = gbc.predict(train_processed)
print("Accuracy: ", accuracy_score(train_Y, pred))
crossval = cross_val_score(gbc, test_split, testY_split, scoring="f1", cv=10)
getScore(crossval)


#pick strongest model and hyperparameter tune
#logistic Regression provides best F1 score
from sklearn.model_selection import GridSearchCV


logres_param_grid = [{"solver":["liblinear","lbfgs"], "C":[0.001, 0.01, 0.1],"max_iter":[50,100,200,400]}]
logres_grid = GridSearchCV(LogisticRegression(random_state=42),
                           param_grid=logres_param_grid,
                           scoring="f1",
                           verbose=1,
                           n_jobs=-1)
logres_grid.fit(X=train_split, y=trainY_split)

# for f1, params in zip(logres_grid.cv_results_["mean_test_score"], logres_grid.cv_results_["params"]):
#     print(f1, params)
logres_final = logres_grid.best_estimator_
logres_predict = logres_final.predict(test_split)

print("Accuracy:" , accuracy_score(testY_split, logres_predict))
print("F1: ", f1_score(testY_split, logres_predict))
print("Precision:", precision_score(testY_split, logres_predict))
print("Recall:", recall_score(testY_split, logres_predict))


svm_param_grid = [{ "C":[0.5,1,1.5],
                    "kernel":["linear", "poly", "rbf"],
                    "degree":[2.9, 3.1, 3.3],
                    "gamma":["auto", "scale"]
                    }]
svm_grid = GridSearchCV(SVC(random_state=42),
                        param_grid=svm_param_grid,
                        scoring="f1",
                        verbose=1,
                        n_jobs=-1)
svm_grid.fit(X=train_split, y=trainY_split)

# for f1, params in zip(svm_grid.cv_results_["mean_test_score"], svm_grid.cv_results_["params"]):
#     print(f1, params)
svm_final = svm_grid.best_estimator_
svm_predict = svm_final.predict(test_split)

print("Accuracy:" , accuracy_score(testY_split, svm_predict))
print("F1: ", f1_score(testY_split, svm_predict))
print("Precision:", precision_score(testY_split, svm_predict))
print("Recall:", recall_score(testY_split, svm_predict))

#prediction

#transform test set

x = transformDataset(test_raw)
test_processed = dataClean(x)

test_predict = logres_final.predict(test_processed)

predict_df = pd.DataFrame(data={"PassengerId": test_raw.index, "Survived": test_predict})

writeKagglePrediction(PROJECT_NAME, predict_df)




#tickets associaated with cabin


#Graphs
# import seaborn as sns
#
# sns.catplot("Salutation", data=x, kind="count", order=x["Salutation"].value_counts(ascending=False).index)
#
# g = sns.FacetGrid(data=ticketMultiPerson, hue="Survived")
# g.map(plt.hist, "FamilySize", alpha=.2)
# g.add_legend()
#
# sns.catplot(data=ticketMultiPerson, hue="Survived", x="FamilySize", y="FamilySurvived", col="Pclass", alpha=0.5)
# sns.violinplot(x="Parch", y="SibSp", hue="Survived",data=train_set,split=False, scale="count")
#
# g = sns.FacetGrid(data=train_age, col="AgeGroup", col_wrap=5)
# g.map(plt.hist, "SibSp")
#
# g = sns.FacetGrid(data=train_age, col="Parch", row="SibSp")
# g.map(plt.hist, "AgeGroup")
#
# g = sns.FacetGrid(col="AgeGroup", data=train_age, hue="Survived", dropna=True, col_wrap=4)
# g.map(plt.hist, "Pclass", alpha=0.4)
#
# g = sns.catplot(x="Parch", y="SibSp", hue="AgeGroup", data=train_age)
#
# g = sns.FacetGrid(data=train_set, hue="Survived")
# g.map(plt.hist, "GroupSize")
#
# If Parch is 2 or 1, then likely traveling with parents
# sns.swarmplot(x="AgeGroup", y="Parch",hue="Survived", data=train_set[train_set["Parch"] > 2])
#
# train_set.loc[train_set["Age"] == -1, "AgeGroup"] = "Unknown"
#
# g = sns.FacetGrid(hue="Survived", col="Sex", data=train_set)
#  g.map(plt.hist, "Salutation")
#
# train_set.hist(column="Salutation", by=["Sex", "Pclass","Survived"], layout=(2,6), xlabelsize=5, ylabelsize=5, sharey=True)
#
# train_set.hist(column="Age", by=["Survived", "Embarked"], sharey=True, layout=(3,3))
#
# #which classes belon to which Pclass - this is misleading because 77% don't have cabins
# g = sns.FacetGrid(data=train_set.loc[cabin.index], hue="Pclass", sharey=False)
# g.map(plt.hist, "CabinClass", alpha=0.5)
# g.add_legend()
#
# #Any relationship with cabin number and survival?
# g = sns.FacetGrid(data=train_set.loc[cabin.index], row="Pclass", hue="Survived", sharey=False)
# g.map(plt.hist, "CabinNumber", alpha=0.5)
# g.add_legend()
#
# #which class and how much they paid
# g = sns.FacetGrid(data=train_set, hue="Survived", row="Pclass", sharex=False)
# g.map(plt.hist, "Fare", alpha=0.5)
# g.add_legend()
#
# #Let's see where the different PClass embarked from
# g = sns.FacetGrid(data=train_set, row="Pclass", sharey=False)
# g.map(plt.hist, "Embarked", alpha=0.5, stacked=True)
# g.add_legend()
#
# #Note: Seems like those that are staying in a lower number cabin have a higher likelihood to survive
# g = sns.FacetGrid(data=cabin, row="Survived", col="Pclass")
# g.map(plt.hist, "CabinNumber", alpha=0.4)
#
# g = sns.FacetGrid(data=train_set[~blank], hue="Pclass")
# g.map(plt.hist, "Embarked", alpha=0.5)
# g.add_legend()
#
# g = sns.FacetGrid(hue="Survived", data=train_set)
# g.map(plt.hist, "Age", alpha=.5)
#
# g = sns.FacetGrid(hue="Survived", data=train_set, legend_out=True)
# g.map(plt.hist, "Sex", alpha=.5)
#
# g = sns.FacetGrid(data=train_set, col="Embarked", row="Sex", hue="Survived")
# g.map(plt.hist, "Age", alpha=0.5)
#
# g = sns.FacetGrid(data=train_set, hue="Survived")
# g.map(plt.hist, "Pclass", alpha=0.5)
#
# plt.hist(train_set.loc[train_set["Survived"] == 1, "Pclass"], stacked=True, label="Survived", alpha=0.5, color="blue")
# plt.hist(train_set.loc[train_set["Survived"] == 0, "Pclass"], stacked=True, label="Not Survived", alpha=0.5, color="green")
# plt.legend()
#
#
#
# sns.kdeplot(data=x)
#
