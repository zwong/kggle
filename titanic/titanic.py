import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

from shared import getDataSet, loadDataSet, findBlankColumns, writeKagglePrediction
#pd.set_option('display.max_columns',10)


#Kaggle Constants
KAGGLE_COMPETITION = "titanic"
PROJECT_NAME = "titanic"

getDataSet(KAGGLE_COMPETITION, PROJECT_NAME)

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

    cabin["CabinNumber"] = cabin["CabinNumber"].fillna(value=-1)
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


    #Embarked - Commenting out.
    from sklearn.impute import SimpleImputer
    #impute = SimpleImputer(strategy="most_frequent")
    #train_set["Embarked"] = impute.fit_transform(train_set["Embarked"].values.reshape(-1,1))

    #Name Salutation
    name = train_set["Name"].str.extract(pat="(.+),\s(.+?)\.(.*)")
    train_set["LastName"] = name[0]
    train_set["Salutation"] = name[1]
    train_set["FirstName"] = name[2]

    #Mapping Salutations
    train_set.loc[train_set["Salutation"] == "Mme", "Salutation"] = "Mrs"
    train_set.loc[train_set["Salutation"] == "Mlle", "Salutation"] = "Miss"
    train_set.loc[train_set["Salutation"] == "the Countess", "Salutation"] = "Miss"
    train_set.loc[train_set["Salutation"] == "Lady", "Salutation"] = "Miss"
    train_set.loc[train_set["Salutation"] == "Jonkheer", "Salutation"] = "Master"
    train_set.loc[train_set["Salutation"] == "Major", "Salutation"] = "Mr"
    train_set.loc[train_set["Salutation"] == "Col", "Salutation"] = "Mr"
    train_set.loc[train_set["Salutation"] == "Sir", "Salutation"] = "Mr"
    train_set.loc[train_set["Salutation"] == "Capt", "Salutation"] = "Mr"
    train_set.loc[train_set["Salutation"] == "Don", "Salutation"] = "Mr"

    #Cutdown # of Salutations
    train_set.loc[~train_set["Salutation"].isin(["Mr", "Miss", "Mrs"]), "Salutation"] = "Other"

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

    #Person Type
    train_set["PersonType"] = np.nan
    train_set["PersonType"] = "Adult"
    train_set.loc[(train_set["Age"] < ADULT_AGE), "PersonType"] = "Child"
    train_set.loc[(train_set["Salutation"] == "Master") & (train_set["Age"] < ADULT_AGE), "PersonType"] = "Child"


    #Group Size
    train_set["GroupSize"] = train_set["Parch"] + train_set["SibSp"] + 1

    #Selected 4 instead of 6 or 5 because of the higher correlation with Survivability
    train_set.loc[train_set["GroupSize"] == 1, "GroupType"] = "SingleTraveller"
    train_set.loc[(train_set["GroupSize"] <= 4) &
                  (train_set["GroupSize"] > 1), "GroupType"] = "2 - 4"
    train_set.loc[(train_set["GroupSize"] > 4), "GroupType"] = "More than 5"


    #Parch

    train_set.loc[(train_set["Parch"] == 0), "ParchType"] = "NoParch"
    train_set.loc[((train_set["Parch"] == 1) | (train_set["Parch"] == 2)), "ParchType"] = "1-2Parch"
    train_set.loc[(train_set["Parch"] > 2), "ParchType"] = "Over2Parch"

    # train_set["ParchType"] = train_set["Sex"] + train_set["PersonType"] + train_set["ParchType"]

    #SibSp
    train_set.loc[(train_set["SibSp"] == 0), "SibSpType"] = "NoSiblingSpouse"
    train_set.loc[(train_set["SibSp"] == 1), "SibSpType"] = "OneSiblingSpouse"
    train_set.loc[(train_set["SibSp"] > 1) & (train_set["Parch"] <= 4),
             "SibSpType"] = "2-4SiblingSpouse"
    train_set.loc[(train_set["SibSp"] > 4), "SibSpType"] = "Over4SiblingSpouse"

    # train_set["SibSpType"] = train_set["Sex"] + train_set["PersonType"] + train_set["SibSpType"]

    #Fare
    #Fare is for the total cost for more than one person. Let's average this out
    train_set["AverageFare"] = train_set["Fare"]/train_set["GroupSize"]

    #check if there are 0 fare tickets. this shouldn't be the case and we should predict the value based off groupsize/persontype/embarked/class/groupsize
    fare_lookup = train_set.groupby(["PersonType","Sex","Embarked","Pclass"])["AverageFare"].mean()

    defaultfare_lookup = train_set.groupby(["Embarked","Pclass"])["AverageFare"].median()

    # zerofare = train_set[train_set["AverageFare"] == 0]
    #
    # for idx in zerofare.index:
    #     record = train_set.loc[idx]
    #     persontype_val = record["PersonType"]
    #     sex_val = record["Sex"]
    #     embarked_val = record["Embarked"]
    #     pclass_val = record["Pclass"]
    #
    #     if (persontype_val, sex_val, embarked_val, pclass_val) in fare_lookup.index:
    #         #print("Fare Exists")
    #         train_set.loc[idx, "AverageFare"] = fare_lookup[persontype_val, sex_val, embarked_val, pclass_val]
    #     else:
    #         print("Fare does not exist")
    #         #create based
    #         train_set.loc[idx,"AverageFare"] = defaultfare_lookup[embarked_val, pclass_val]
    #


    #Age
    adult_age = train_set[train_set["Age"] >= ADULT_AGE]["Age"].median()
    child_age = train_set[train_set["Age"] < ADULT_AGE]["Age"].median()
    print("Median Adult Age: ", adult_age)
    print("Median Child Age: ", child_age)

    age_lookup = train_set.groupby(["Parch", "SibSp","Sex","Embarked","PersonType"])["Age"].median()

    missing_age = train_set[train_set["Age"].isna() == True]
    for idx in missing_age.index:
        record = train_set.loc[idx]
        parch_val = record["Parch"]
        sibsp_val = record["SibSp"]
        sex_val = record["Sex"]
        embarked_val = record["Embarked"]
        persontype_val = record["PersonType"]


        if (parch_val, sibsp_val, sex_val, embarked_val,persontype_val) in age_lookup.index:
            #print("Exists")
            train_set.loc[idx, "Age"] = age_lookup[parch_val, sibsp_val, sex_val, embarked_val, persontype_val]
        else:
            #print("Does not exist")
            if (persontype_val == "Child"):
                train_set.loc[idx, "Age"] = child_age
                print("[",idx,"] : Child",child_age)
            else:
                train_set.loc[idx, "Age"] = adult_age
                print("[", idx, "] Adult: ", adult_age)

    # Columns to add
    # Age Group
    labels = ["{0} - {1}".format(i, i+4) for i in range(0,60,5)]
    train_set["AgeGroup"] = pd.cut(train_set["Age"], bins=range(0,65,5), labels=labels).to_list()
    train_set["AgeGroup"] = train_set["AgeGroup"].fillna("Over 60")

    train_set["SexAgeGroup"] = train_set["Sex"] + "-" + train_set["AgeGroup"]

    # train_set["Embarked"].fillna(value="Unknown", inplace=True)
    # train_set["Age"].fillna(value=-1, inplace=True)

    #Pclass
    train_set.loc[train_set["Pclass"] == 1, "PclassCat"] = "First"
    train_set.loc[train_set["Pclass"] == 2, "PclassCat"] = "Second"
    train_set.loc[train_set["Pclass"] == 3, "PclassCat"] = "Third"


    # #Fare
    labels = ["{0} - {1}".format(i, i+4) for i in range(0, 30, 5)]
    train_set["FareGroup"] = pd.cut(train_set["AverageFare"], bins=range(0,35,5), labels=labels).to_list()
    train_set["FareGroup"].fillna(value="Over 30", inplace=True)

    #SexPclass
    train_set["SexPclass"] = train_set["Sex"] + train_set["PclassCat"]

    return train_set


##################
## Data Process
##################
#Receive data set
#OntHotEncodes classifications and imputes missing values
#Returns cleaned dataset

def dataClean(train_set):
    ##Note: Removed Category Columns
    train_set.drop(labels=["Name", "FirstName", "LastName", "Ticket", "Cabin", "Pclass", "SexPclass","SexAgeGroup",
                           "FareGroup", "SibSpType","ParchType","Embarked"], axis=1,
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

    # Remove numeric features
    train_processed.drop(labels=["GroupSize", "SibSp", "Parch", "Age", "AverageFare"], axis=1, inplace=True)

    #Feature Scale for numeric categories
    from sklearn.preprocessing import StandardScaler
    scale = StandardScaler()
    train_processed = pd.DataFrame(data = scale.fit_transform(train_processed), columns=train_processed.columns)


    train_preprocessed = train_processed.copy()


    print("DataClean: Columns: ", train_set.columns)
    if "Survived" in train_processed.columns:
        print("Survived feature found. Dropping")
        train_processed.drop("Survived", axis=1, inplace=True)
    else:
        print("Survived feature does not exist")

    train_labels = train_processed.columns

    return  train_processed, train_preprocessed

def getScore(results):
    scores = results
    print("Scores:" , scores)
    print("Mean:", np.mean(scores))
    print("STD:", np.std(scores))


print("Cleaning train dataset...")
trainTransform = transformDataset(train_set)
train_Y = trainTransform["Survived"]

train_processed, train_preprocessed  = dataClean(trainTransform)
print("Cleaning train dataset complete")

#Train Test Split
from sklearn.model_selection import train_test_split

train_split, test_split, trainY_split, testY_split = train_test_split(train_processed, train_Y, random_state=42, test_size=.3)

#Predictor
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score



print("Logistic Regression")
logreg = LogisticRegression(solver='lbfgs')
# logreg.fit(X=train_split, y=trainY_split)
# pred = logreg.predict(test_split)
# print("Accuracy: ", accuracy_score(testY_split, pred))
crossval = cross_val_score(logreg, train_processed, train_Y, scoring="f1", cv=10)
getScore(crossval)

print("Ridge Regression")
ridgereg = RidgeClassifier(alpha=0.5)
# logreg.fit(X=train_split, y=trainY_split)
# pred = logreg.predict(test_split)
# print("Accuracy: ", accuracy_score(testY_split, pred))
crossval = cross_val_score(ridgereg, train_processed, train_Y, scoring="f1", cv=10)
getScore(crossval)

print("SVC")
svc = SVC(gamma="auto")
# svc.fit(X=train_split, y=trainY_split)
# pred = svc.predict(train_processed)
# print("Accuracy: ", accuracy_score(train_Y, pred))
crossval = cross_val_score(svc, train_processed, train_Y, scoring="f1", cv=10)
getScore(crossval)

print("LinearSVC")
svc = LinearSVC(max_iter=10000)
# svc.fit(X=train_split, y=trainY_split)
# pred = svc.predict(train_processed)
# print("Accuracy: ", accuracy_score(train_Y, pred))
crossval = cross_val_score(svc, train_processed, train_Y, scoring="f1", cv=10)
getScore(crossval)


print("Random Forest Tree")
rtc = RandomForestClassifier(n_estimators=500)
# dtc.fit(X=train_split, y=trainY_split)
# pred = dtc.predict(train_processed)
# print("Accuracy: ", accuracy_score(train_Y, pred))
crossval = cross_val_score(rtc, train_processed, train_Y, scoring="f1", cv=10)
getScore(crossval)

# print("BaggingClassifier")
# bc = BaggingClassifier(n_estimators=1000, base_estimator=SVC(gamma="auto"))
# # dtc.fit(X=train_split, y=trainY_split)
# # pred = dtc.predict(train_processed)
# # print("Accuracy: ", accuracy_score(train_Y, pred))
# crossval = cross_val_score(bc, train_processed, train_Y, scoring="f1", cv=10)
# getScore(crossval)


print("Decision Tree")
dtc = DecisionTreeClassifier()
# dtc.fit(X=train_split, y=trainY_split)
# pred = dtc.predict(train_processed)
# print("Accuracy: ", accuracy_score(train_Y, pred))
crossval = cross_val_score(dtc, train_processed, train_Y, scoring="f1", cv=10)
getScore(crossval)

# print("Extra Tree Classifier")
# etc = ExtraTreeClassifier()
# # etc.fit(X=train_split, y=trainY_split)
# # pred = etc.predict(train_processed)
# # print("Accuracy: ", accuracy_score(train_Y, pred))
# crossval = cross_val_score(etc, train_processed, train_Y, scoring="f1", cv=10)
# getScore(crossval)


# print("Grandient Boost")
# gbc = GradientBoostingClassifier()
# # gbc.fit(X=train_split, y=trainY_split)
# # pred = gbc.predict(train_processed)
# # print("Accuracy: ", accuracy_score(train_Y, pred))
# crossval = cross_val_score(gbc, train_processed, train_Y, scoring="f1", cv=10)
# getScore(crossval)


#pick strongest model and hyperparameter tune
#logistic Regression provides best F1 score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

logres_param_grid = [{
                        "solver":["liblinear"],
                        "penalty": ["l1", "l2"],
                        "C":[0.001, 0.01, 0.1, 0.5],
                        "max_iter":[500,1000]
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
                           cv=10,
                           return_train_score=True,
                           n_jobs=-1)
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



# ridgereg_param_grid = [{
#                         "alpha": [0.5,0.7,0.9,0.95,1],
#                         "max_iter": [None, 0.5, 10, 50, 100,150],
#                         "tol": [0.000001, 0.00001,0.0001,0.001,0.1],
#                         "class_weight": [None, "balanced"]
#                     }]
# ridgereg_grid = GridSearchCV(RidgeClassifier(random_state=42),
#                            param_grid=ridgereg_param_grid,
#                            scoring="average_precision",
#                            verbose=1,
#                            cv=10,
#                            return_train_score=True,
#                            n_jobs=-1)
# ridgereg_grid.fit(X=train_split, y=trainY_split)
#
# # for f1, params in zip(logres_grid.cv_results_["mean_test_score"], logres_grid.cv_results_["params"]):
# #     print(f1, params)
# ridgereg_final = ridgereg_grid.best_estimator_
# print(ridgereg_final)
# #train with full train set
#
# ridgereg_predict = ridgereg_final.predict(test_split)
#
# print("Ridge Regression")
# print("====================")
# print("Accuracy:" , accuracy_score(testY_split, ridgereg_predict))
# print("F1: ", f1_score(testY_split, ridgereg_predict))
# print("Precision:", precision_score(testY_split, ridgereg_predict))
# print("Recall:", recall_score(testY_split, ridgereg_predict))
# print("Confusion Matrix\n", confusion_matrix(testY_split, ridgereg_predict))



rfc_param_grid = [{
                        "n_estimators":[100, 200, 300],
                        "max_depth": [None, 8, 9, 10],
                        "min_samples_split": [4,6],
                        "min_samples_leaf": [4,6],
                         "max_features" : ["auto", "sqrt", "log2"]
                    }]
rfc_grid = GridSearchCV(RandomForestClassifier(random_state=42),
                        param_grid=rfc_param_grid,
                        scoring="average_precision",
                        verbose=0,
                        cv=10,
                        return_train_score=True,
                        n_jobs=-1)

rfc_grid.fit(X=train_split, y=trainY_split)

#import pickle
#pickle.dump(rfc_grid,file=open("rfc.grid.obj", "wb"))


for f1, params in zip(logres_grid.cv_results_["mean_test_score"], logres_grid.cv_results_["params"]):
    print(f1, params)
rfc_final = rfc_grid.best_estimator_

#train with full train set

rfc_predict = rfc_final.predict(test_split)

print("Random Forest")
print("====================")
print("Accuracy:" , accuracy_score(testY_split, rfc_predict))
print("F1: ", f1_score(testY_split, rfc_predict))
print("Precision:", precision_score(testY_split, rfc_predict))
print("Recall:", recall_score(testY_split, rfc_predict))
print("Confusion Matrix\n", confusion_matrix(testY_split, rfc_predict))



svm_param_grid = [{ "C":[0.6, 0.8, 1],
                    "kernel":["linear", "sigmoid","poly"],
                    "degree":[2,4],
                    "gamma":["auto"],
                    "coef0": [0.0, 0.4,0.8]
                    }]
svm_grid = GridSearchCV(SVC(random_state=42,max_iter=10),
                        param_grid=svm_param_grid,
                        scoring="average_precision",
                        verbose=1,
                        cv=10,
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



# linsvm_param_grid = [{ "penalty": ["l2"] ,
#                     "loss": ["hinge", "squared_hinge"],
#                     "C": [0.00001, 0.0001 , 0.001, 0.1, 0.2 ],
#                     "max_iter": [1, 10]
#                     }]
# linsvm_grid = GridSearchCV(LinearSVC(random_state=42),
#                         param_grid=linsvm_param_grid,
#                         scoring="average_precision",
#                         verbose=1,
#                         cv=10,
#                         n_jobs=-1)
# linsvm_grid.fit(X=train_split, y=trainY_split)
#
# # for f1, params in zip(svm_grid.cv_results_["mean_test_score"], svm_grid.cv_results_["params"]):
# #     print(f1, params)
# linsvm_final = linsvm_grid.best_estimator_
# linsvm_predict = linsvm_final.predict(test_split)
#
# print("LinearSVC")
# print("====")
# print("Accuracy:" , accuracy_score(testY_split, linsvm_predict))
# print("F1: ", f1_score(testY_split, linsvm_predict))
# print("Precision:", precision_score(testY_split, linsvm_predict))
# print("Recall:", recall_score(testY_split, linsvm_predict))
# print("Confusion Matrix\n", confusion_matrix(testY_split, linsvm_predict))
#


# #Linear SVC with Polynomial Features
# from sklearn.preprocessing import PolynomialFeatures
#
# pf = PolynomialFeatures(degree=3)
# train_pf_split = pf.fit_transform(train_split)
#
#
# linsvm_param_grid = [{ "penalty": ["l2"] ,
#                     "loss": ["hinge", "squared_hinge"],
#                     "C": [0.1, 0.2, 0.3],
#                     "max_iter": [1000,2000]
#                     }]
# linsvm_grid = GridSearchCV(LinearSVC(random_state=42),
#                         param_grid=linsvm_param_grid,
#                         scoring="average_precision",
#                         verbose=1,
#                         cv=10,
#                         n_jobs=-1)
# linsvm_grid.fit(X=train_pf_split, y=trainY_split)
#
# # for f1, params in zip(svm_grid.cv_results_["mean_test_score"], svm_grid.cv_results_["params"]):
# #     print(f1, params)
# linsvm_final = linsvm_grid.best_estimator_
#
# test_pf_split = pf.fit_transform(test_split)
# linsvm_predict = linsvm_final.predict(test_pf_split)
#
# print("LinearSVC + PolynomialFeatures")
# print("====")
# print("Accuracy:" , accuracy_score(testY_split, linsvm_predict))
# print("F1: ", f1_score(testY_split, linsvm_predict))
# print("Precision:", precision_score(testY_split, linsvm_predict))
# print("Recall:", recall_score(testY_split, linsvm_predict))
# print("Confusion Matrix\n", confusion_matrix(testY_split, linsvm_predict))
#
#
# gb_param_grid = [{ "learning_rate":[0.001,0.01,0.1,1],
#                     "max_depth":[1,3,5],
#                     "max_features":["auto", "sqrt", "log2"],
#                     "n_estimators": [100,200,400,800]
#                     }]
# gb_grid = GridSearchCV(GradientBoostingClassifier(random_state=42),
#                         param_grid=gb_param_grid,
#                         scoring="average_precision",
#                         verbose=0,
#                         n_jobs=-1)
# gb_grid.fit(X=train_split, y=trainY_split)
#
# # for f1, params in zip(svm_grid.cv_results_["mean_test_score"], svm_grid.cv_results_["params"]):
# #     print(f1, params)
# gb_final = gb_grid.best_estimator_
# gb_predict = gb_final.predict(test_split)
#
# print("Gradient Boost")
# print("==============")
# print("Accuracy:" , accuracy_score(testY_split, gb_predict))
# print("F1: ", f1_score(testY_split, gb_predict))
# print("Precision:", precision_score(testY_split, gb_predict))
# print("Recall:", recall_score(testY_split, gb_predict))
# print("Confusion Matrix\n", confusion_matrix(testY_split, gb_predict))

#
# bc_param_grid = [{  "n_estimators": [75, 100, 125],
#                     "base_estimator": [linsvm_final, logres_final, ridgereg_final, gb_final, rfc_final, svm_final]
#                     }]
# bc_grid = GridSearchCV(BaggingClassifier(random_state=42),
#                         param_grid=bc_param_grid,
#                         scoring="average_precision",
#                         verbose=2,
#                         n_jobs=-1)
# bc_grid.fit(X=train_split, y=trainY_split)
#
# # for f1, params in zip(svm_grid.cv_results_["mean_test_score"], svm_grid.cv_results_["params"]):
# #     print(f1, params)
# bc_final = gb_grid.best_estimator_
# bc_predict = gb_final.predict(test_split)
#
# print("Bagging Classifier")
# print("==============")
# print("Accuracy:" , accuracy_score(testY_split, bc_predict))
# print("F1: ", f1_score(testY_split, bc_predict))
# print("Precision:", precision_score(testY_split, bc_predict))
# print("Recall:", recall_score(testY_split, bc_predict))
# print("Confusion Matrix\n", confusion_matrix(testY_split, bc_predict))
#
#


#prediction

#transform test set

# logres_final.fit(train_processed, train_Y)
# svm_final.fit(train_processed, train_Y)
# linsvm_final.fit(train_processed, train_Y)
# gb_final.fit(train_processed, train_Y)
# rfc_final.fit(train_processed, train_Y)

print('Logistic Regression Cross validation score: {:.3f}'.format(cross_val_score(logres_final, train_split, trainY_split, cv=20, scoring="f1").mean()))
print('SVM Cross validation score: {:.3f}'.format(cross_val_score(svm_final, train_split, trainY_split, cv=20, scoring="f1").mean()))
# print('LinSVM Cross validation score: {:.3f}'.format(cross_val_score(linsvm_final, train_split, trainY_split, cv=20, scoring="f1").mean()))
# print('Gradient Boost Cross validation score: {:.3f}'.format(cross_val_score(gb_final, train_split, trainY_split, cv=20, scoring="f1").mean()))
print('Random Forest Cross validation score: {:.3f}'.format(cross_val_score(rfc_final, train_split, trainY_split, cv=20, scoring="f1").mean()))
# print('Bagging Cross validation score: {:.3f}'.format(cross_val_score(bc_final, train_split, trainY_split, cv=20, scoring="f1").mean()))
# print('Ridge Regression Cross validation score: {:.3f}'.format(cross_val_score(ridgereg_final, train_split, trainY_split, cv=20, scoring="f1").mean()))



testTransform = transformDataset(test_raw)
test_processed, test_preprocessed = dataClean(testTransform)

#todo: Check if column names are same between train and test
if len(list(test_processed)) == len(list(train_processed)):
    print("Feature number matches, continuing to match features")
else:
    print("Feature number does not match")
for x in list(train_processed):
    if x in list(test_processed):
        print(x, ": Match in train and test")
    else:
        print(x, ": No Match in test (Train only)")

print("Feature check complete.")

#Using just the training set. Did not train against the training-test set
test_predict = logres_final.predict(test_processed)
predict_df = pd.DataFrame(data={"PassengerId": test_raw.index, "Survived": test_predict})
writeKagglePrediction(PROJECT_NAME, "logres", predict_df)

test_predict = rfc_final.predict(test_processed)
predict_df = pd.DataFrame(data={"PassengerId": test_raw.index, "Survived": test_predict})
writeKagglePrediction(PROJECT_NAME, "svc", predict_df)


rfc_final.fit(train_processed, train_Y)

test_predict = rfc_final.predict(test_processed)
predict_df = pd.DataFrame(data={"PassengerId": test_raw.index, "Survived": test_predict})
writeKagglePrediction(PROJECT_NAME, "rfc", predict_df)


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
