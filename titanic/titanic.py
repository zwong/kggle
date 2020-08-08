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
    train_set.loc[~train_set["Salutation"].isin(["Mr", "Miss", "Mrs", "Master", "Rev"]), "Salutation"] = "Other"

    #Last name doens't help. Could be same last name. More important is the Parch/SibSp numbers or Ticket is same
    #train_set[train_set["FirstName"].str.find(sub="(") > 0][["FirstName","LastName", "Salutation","Parch","SibSp","Age"]]
    #train_set[train_set["FirstName"].str.find(sub="(") > 0]


    #For people traveling together (i.e on same ticket), what are survival rates?
    #More likely to NOT survive the greater the number of people traveling together
    #There is an anomaly for large groups that traveled together. A Chinese group.  Is this just an anomally? Assuming it is
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
    train_set["PersonType"] = np.nan
    # train_set["MaritalStatus"] = "Not Married"



    #Age / Salutation
    #SibSp > 1 - must be a traveling with siblings
        #Parch == 0 - could be adult traveling with siblings / could be orphan traveling with siblings
        #Parch == 1 - could be adult traveling with child / could be child traveling with atleast 2 siblings
        #Parch == 2 - could be adult traveling with child / could be child traveling with atleast 2 siblings
    #SibSp == 1 - could traveling with sibling or spouse
        #Parch > 2 - must be adult traveling with kids
        #Parch == 2 - could be traveling with spouse and 2 kids / could be person traveling with sibling and parents
        #Parch == 1 - could be traveling with spouse and 1 kids / could be person traveling with sibling and 1 parent
        #Parch == 0 - could be traveling with spouse only / Could be traveling with sibling
    #SibSp == 0 - traveling with no sibling or spouse
        #Parch == 0 - Solo Traveler / Orphan
        #Parch == 1 - Single parent traveling with child / Single child traveling with single parent
        #Parch == 2 - Single parent traveling with 2 children / Single child traveling with both parents
        #Parch > 2 - Must be Single parent traveling with more than 2 kids

    #
    # #If you are an adult, have SibSp == 1 and Parch > 2, then fits profile of a married  with at least 2 kids
    # train_set.loc[(train_set["Age"] > ADULT_AGE) & (train_set["SibSp"] == 1) &
    #               (train_set["Parch"] > 2), "MaritalStatus"] = "Married"
    #
    # train_set.loc[(train_set["Age"] > ADULT_AGE) & (train_set["SibSp"] == 0) &
    #               (train_set["Parch"] > 2), "MaritalStatus"] = "Married"
    #
    # #If we don't know your age, If you have 1 SibSp and more than 2 Parch, then this fits the profile of a Spouse + >=2 kids
    # train_set.loc[(train_set["SibSp"] == 1) &
    #               (train_set["Parch"] > 2), "MaritalStatus"] = "Married"
    #
    # # Salutation of Mrs == Married
    # train_set.loc[train_set["Salutation"] == "Mrs", "MaritalStatus"] = "Married"
    #
    #
    # #Person Type
    train_set["PersonType"] = "Adult"
    #
    # # train_set.loc[ train_set["Salutation"] == "Mrs", "PersonType"] = "Adult"
    # # train_set.loc[train_set["Salutation"] == "Mrs", "PersonType"] = "Adult"
    #
    # #Master is a salutation for boys
    # train_set.loc[(train_set["Salutation"] == "Master"), "PersonType"] = "Child"
    #
    # #If you aren't married and your SibSp is greater than 1, then you're traveling with siblings. Assume you are a child
    # train_set.loc[(train_set["Age"] < ADULT_AGE) & (train_set["MaritalStatus"] == "Not Married") & (train_set["SibSp"] > 1), "PersonType"] = "Child"
    #
    # # If you aren't married and your are traveling with less than 2 Parent/Children, assuem you're a child traveling with parents
    # train_set.loc[(train_set["Age"] < ADULT_AGE) & (train_set["MaritalStatus"] == "Not Married") & (train_set["Parch"] <= 2), "PersonType"] = "Child"
    #
    # # train_set.loc[(train_set["MaritalStatus"] == "Married"), "PersonType"] = "Adult"
    train_set.loc[(train_set["Age"] < ADULT_AGE), "PersonType"] = "Child"
    # # train_set.loc[(train_set["Age"] >= ADULT_AGE), "PersonType"] = "Adult"
    # # train_set.loc[(train_set["SibSp"] == 0) & (train_set["Parch"] == 0), "PersonType"] = "Adult"
    #
    #


    #train_set.loc[(train_set["Age"] > ADULT_AGE) & (train_set["Parch"] > 0), "MaritalStatus"] = "Married"
    train_set["GroupSize"] = train_set["Parch"] + train_set["SibSp"] + 1

    #Selected 4 instead of 6 or 5 because of the higher correlation with Survivability
    train_set.loc[ train_set["GroupSize"] >= 5, "GroupType"] = "4 or more"
    train_set.loc[ train_set["GroupSize"] < 5, "GroupType"] = "Less than 4"
    train_set.loc[ train_set["GroupSize"] == 1, "GroupType"] = "SingleTraveller"



    #Fare
    #Fare is for the total cost for more than one person. Let's average this out
    train_set["AverageFare"] = train_set["Fare"]/train_set["GroupSize"]

    #Age
    median_age = train_set["Age"].median()
    age_lookup = train_set.groupby(["Parch", "SibSp","Sex","Embarked"])["Age"].median()

    missing_age = train_set[train_set["Age"].isna() == True]
    for idx in missing_age.index:
        record = train_set.loc[idx]
        parch_val = record["Parch"]
        sibsp_val = record["SibSp"]
        sex_val = record["Sex"]
        embarked_val = record["Embarked"]
        #print(idx, ":", parch_val, sibsp_val, sex_val, embarked_val)

        if (parch_val, sibsp_val, sex_val, embarked_val) in age_lookup.index:
            #print("Exists")
            train_set.loc[idx, "Age"] = age_lookup[parch_val, sibsp_val, sex_val, embarked_val] + .1
        else:
            #print("Does not exist")
            train_set.loc[idx, "Age"] = median_age + .1





    # median_childage = train_set[train_set["PersonType"] == "Child"]["Age"].median()
    # median_adultage = train_set[train_set["PersonType"] == "Adult"]["Age"].median()

    # train_set.loc[ (train_set["PersonType"] == "Adult") & (train_set["Age"].isnull()), "Age"] = median_adultage + 0.1
    # train_set.loc[ (train_set["PersonType"] == "Child") & (train_set["Age"].isnull()), "Age"] = median_childage + 0.1

    #median_childage = train_set[train_set["MaritalStatus"] == "Married"]["Age"].median()
    #median_adultage = train_set[train_set["MaritalStatus"] == "Not Married"]["Age"].median()
    # train_set.loc[(train_set["MaritalStatus"] == "Married") & (train_set["Age"].isnull()), "Age"] = median_adultage + 0.1
    # train_set.loc[(train_set["MaritalStatus"] == "Not Married") & (train_set["Age"].isnull()), "Age"] = median_childage + 0.1

    # train_set.loc[(train_set["PersonType"] == "Adult") & (train_set["Age"].isnull()), "Age"] = median_adultage + 0.1
    # train_set.loc[(train_set["PersonType"] == "Not Child") & (train_set["Age"].isnull()), "Age"] = median_childage + 0.1

    # Columns to add
    # Age Group
    labels = ["{0} - {1}".format(i, i+4) for i in range(0,60,5)]
    train_set["AgeGroup"] = pd.cut(train_set["Age"], bins=range(0,65,5), labels=labels).to_list()
    train_set["AgeGroup"].fillna("Over 60", inplace=True)

    train_set["SexAgeGroup"] = train_set["Sex"] +"-" + train_set["AgeGroup"]

    # train_set["Embarked"].fillna(value="Unknown", inplace=True)
    # train_set["Age"].fillna(value=-1, inplace=True)

    #Pclass
    train_set.loc[train_set["Pclass"] == 1, "PClassCat"] = "First"
    train_set.loc[train_set["Pclass"] == 2, "PClassCat"] = "Second"
    train_set.loc[train_set["Pclass"] == 3, "PClassCat"] = "Third"

    #Fare
    labels = ["{0} - {1}".format(i, i+4) for i in range(0, 50, 5)]
    train_set["FareGroup"] = pd.cut(train_set["AverageFare"], bins=range(0,55,5), labels=labels).to_list()
    train_set["FareGroup"].fillna(value="Over $50", inplace=True)

    return train_set

##################
## Data Process
##################
#Receive data set
#OntHotEncodes classifications and imputes missing values
#Returns cleaned dataset

def dataClean(train_set):
    ##Note: Removed Category Columns
    train_set.drop(labels=["Name", "FirstName", "LastName", "Ticket", "Cabin", "SexAgeGroup", "Pclass"], axis=1,
           inplace=True)


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
        print("Categories (", len(ord_enc.categories_[0]), "): " , ord_enc.categories_[0])
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

    train_preprocessed = train_processed.copy()

    #Remove numeric features
    train_processed.drop(labels=["Age", "GroupSize", "SibSp", "Fare","AverageFare","Parch"], axis=1,
           inplace=True)


    print("DataClean: Columns: ", train_set.columns)
    if "Survived" in train_processed.columns:
        print("Survived feature found. Dropping")
        train_processed.drop("Survived", axis=1, inplace=True)
    else:
        print("Survived feature does not exist")

    train_labels = train_processed.columns

    # #Feature Scale
    from sklearn.preprocessing import StandardScaler
    scale = StandardScaler()

    train_processed = pd.DataFrame(columns=train_labels, data = scale.fit_transform(train_processed))

    return  train_processed, train_preprocessed

def getScore(results):
    scores = results
    print("Scores:" , scores)
    print("Mean:", np.mean(scores))
    print("STD:", np.std(scores))


print("Cleaning train dataset...")
trainTransform = transformDataset(train_set)
train_Y = trainTransform["Survived"]

#Todo: Identify most effective identifiers in the dataClean() method

train_processed, train_preprocessed  = dataClean(trainTransform)
print("Cleaning train dataset complete")

#Train Test Split
from sklearn.model_selection import train_test_split

train_split, test_split, trainY_split, testY_split = train_test_split(train_processed, train_Y, random_state=42, test_size=.3)

#Predictor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score



print("Logistic Regression")
logreg = LogisticRegression(solver="lbfgs")
# logreg.fit(X=train_split, y=trainY_split)
# pred = logreg.predict(test_split)
# print("Accuracy: ", accuracy_score(testY_split, pred))
crossval = cross_val_score(logreg, train_processed, train_Y, scoring="f1", cv=10)
getScore(crossval)

print("SVM")
svc = SVC(gamma="auto")
# svc.fit(X=train_split, y=trainY_split)
# pred = svc.predict(train_processed)
# print("Accuracy: ", accuracy_score(train_Y, pred))
crossval = cross_val_score(svc, train_processed, train_Y, scoring="f1", cv=10)
getScore(crossval)

print("Random Forest Tree")
rtc = RandomForestClassifier(n_estimators=20)
# dtc.fit(X=train_split, y=trainY_split)
# pred = dtc.predict(train_processed)
# print("Accuracy: ", accuracy_score(train_Y, pred))
crossval = cross_val_score(rtc, train_processed, train_Y, scoring="f1", cv=10)
getScore(crossval)


print("Decision Tree")
dtc = DecisionTreeClassifier()
# dtc.fit(X=train_split, y=trainY_split)
# pred = dtc.predict(train_processed)
# print("Accuracy: ", accuracy_score(train_Y, pred))
crossval = cross_val_score(dtc, train_processed, train_Y, scoring="f1", cv=10)
getScore(crossval)

print("Extra Tree Classifier")
etc = ExtraTreeClassifier()
# etc.fit(X=train_split, y=trainY_split)
# pred = etc.predict(train_processed)
# print("Accuracy: ", accuracy_score(train_Y, pred))
crossval = cross_val_score(etc, train_processed, train_Y, scoring="f1", cv=10)
getScore(crossval)


print("Grandient Boost")
gbc = GradientBoostingClassifier()
# gbc.fit(X=train_split, y=trainY_split)
# pred = gbc.predict(train_processed)
# print("Accuracy: ", accuracy_score(train_Y, pred))
crossval = cross_val_score(gbc, train_processed, train_Y, scoring="f1", cv=10)
getScore(crossval)


#pick strongest model and hyperparameter tune
#logistic Regression provides best F1 score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

logres_param_grid = [{
                        "solver":["liblinear"],
                        "penalty": ["l1", "l2"],
                        "C":[0.001, 0.01, 0.1, 0.5],
                        "max_iter":[100,200,300]
                    },
                    {
                        "solver": ["lbfgs"],
                        "penalty": ["l2"],
                        "C": [0.001, 0.01, 0.1, 0.5],
                        "max_iter": [100, 200, 400]
                    }]
logres_grid = GridSearchCV(LogisticRegression(random_state=42),
                           param_grid=logres_param_grid,
                           scoring="average_precision",
                           verbose=0,
                           cv=5,
                           return_train_score=True)
logres_grid.fit(X=train_split, y=trainY_split)

# for f1, params in zip(logres_grid.cv_results_["mean_test_score"], logres_grid.cv_results_["params"]):
#     print(f1, params)
logres_final = logres_grid.best_estimator_

#train with full train set

logres_predict = logres_final.predict(test_split)

print("Logistic Regression")
print("====================")
print("Accuracy:" , accuracy_score(testY_split, logres_predict))
print("F1: ", f1_score(testY_split, logres_predict))
print("Precision:", precision_score(testY_split, logres_predict))
print("Recall:", recall_score(testY_split, logres_predict))
print("Confusion Matrix\n", confusion_matrix(testY_split, logres_predict))

svm_param_grid = [{ "C":[0.6, 0.7, 0.8,0.9],
                    "kernel":["linear", "poly", "rbf", "sigmoid"],
                    "degree":[1, 2, 3, 4, 5],
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

print("SVC")
print("====")
print("Accuracy:" , accuracy_score(testY_split, svm_predict))
print("F1: ", f1_score(testY_split, svm_predict))
print("Precision:", precision_score(testY_split, svm_predict))
print("Recall:", recall_score(testY_split, svm_predict))
print("Confusion Matrix\n", confusion_matrix(testY_split, svm_predict))


gb_param_grid = [{ "learning_rate":[0.001,0.01,0.1,1],
                    "max_depth":[1,3,5],
                    "max_features":["auto", "sqrt", "log2"],
                    }]
gb_grid = GridSearchCV(GradientBoostingClassifier(random_state=42),
                        param_grid=gb_param_grid,
                        scoring="f1",
                        verbose=1,
                        n_jobs=-1)
gb_grid.fit(X=train_split, y=trainY_split)

# for f1, params in zip(svm_grid.cv_results_["mean_test_score"], svm_grid.cv_results_["params"]):
#     print(f1, params)
gb_final = gb_grid.best_estimator_
gb_predict = gb_final.predict(test_split)

print("Gradient Boost")
print("==============")
print("Accuracy:" , accuracy_score(testY_split, gb_predict))
print("F1: ", f1_score(testY_split, gb_predict))
print("Precision:", precision_score(testY_split, gb_predict))
print("Recall:", recall_score(testY_split, gb_predict))
print("Confusion Matrix\n", confusion_matrix(testY_split, gb_predict))




#prediction

#transform test set

#Retrain model with the full training data set
logres_final.fit(train_processed, train_Y)

testTransform = transformDataset(test_raw)
test_processed, test_preprocessed = dataClean(testTransform)

test_predict = logres_final.predict(test_processed)
predict_df = pd.DataFrame(data={"PassengerId": test_raw.index, "Survived": test_predict})
writeKagglePrediction(PROJECT_NAME, "logres", predict_df)

# test_predict = svm_final.predict(test_processed)
# predict_df = pd.DataFrame(data={"PassengerId": test_raw.index, "Survived": test_predict})
# writeKagglePrediction(PROJECT_NAME, "svm", predict_df)


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
