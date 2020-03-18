import os
import pandas as pd
import numpy as np

BASE_PATH = os.popen("pwd").read()[:-1]
PROJECT_PATH = os.path.join(BASE_PATH, "housing")
DATASET_PATH = os.path.join(PROJECT_PATH, "dataset")
DOWNLOAD_PATH = os.path.join(DATASET_PATH, "zip")


#Kaggle Constants
KAGGLE_COMPETITION = "house-prices-advanced-regression-techniques"
KAGGLE_COMMAND = "kaggle competitions download -c " + KAGGLE_COMPETITION + " -p " + DOWNLOAD_PATH


#Download and unzip Kaggle dataset
def getDataSet():
    try:
        os.mkdir(DOWNLOAD_PATH)
    except:
        print("Directory already exists")

    print("Download dataset...")
    os.system(KAGGLE_COMMAND)

    print("Unpack dataset...")
    os.system("unzip " + DOWNLOAD_PATH + "/*.zip" + " -d " + DATASET_PATH)

    print("Delete zip file...")
    os.system("rm " + DOWNLOAD_PATH + "/*")


#Load Dataset
#Returns test and train Dataframes
def loadDataSet():
    train = pd.read_csv( os.path.join(DATASET_PATH, "train.csv"))
    test =  pd.read_csv( os.path.join(DATASET_PATH, "test.csv"))

    return train, test

#Write Kaggle Prediction File
#output ID and SalesPrice
def writeKagglePrediction(predict):
    None

def findBlankValueColumns(dataset):
    # Check which columns have blank values
    for col in list(dataset):
        missing = dataset[col].isna().sum()
        if missing > 1:
            print(col, ": ", dataset[col].isna().sum() ," blank values found (", "%.4f pct" % (missing/len(dataset[col])*100),
                  ")")

def displayScores(score):
    print("Scores: ", score)
    print("Mean: ", np.mean(score))
    print("STD: ", np.std(score))

from sklearn.base import BaseEstimator, TransformerMixin
class dataPreprocess(BaseEstimator, TransformerMixin):
    None




'''
Analysis
'''

train_raw, test_raw = loadDataSet()

'''
There are many string objects in this dataset. let's separate strings and run corr on these

These are nummeric features:
Index(['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',
       'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
       'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
       'MiscVal', 'MoSold', 'YrSold', 'SalePrice'],
      dtype='object')
      
These are string category features:
Index(['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
       'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
       'SaleType', 'SaleCondition'],
      dtype='object')
'''
train_set = train_raw.copy()

train_set_Y = train_set["SalePrice"]
train_num_col = train_set.columns[train_set.dtypes != "object"]
train_str_col = train_set.columns[train_set.dtypes == "object"]

'''
There are numeric features that are categories (manually reviewed data spec)
["MSSubClass", "OverallQual", "OverallCond"])

Add numeric category columns into category column remove them from  numeric column
'''
train_numstr_col = pd.Index(["MSSubClass", "OverallQual", "OverallCond"])
train_str_col = train_str_col.append(train_numstr_col)
train_num_col = train_num_col.drop(train_numstr_col)

'''
check which columns are blank

Categories - Replace NA with "NONE"
> 90% Blank: Alley, PoolQC, MiscFeature
> 80% Fence
> 40% FireplaceQu

Number: 
> 10% LotFrontage <-- covered during Num Processing
'''
train_set["Alley"].fillna(value="None", inplace=True)
train_set["PoolQC"].fillna(value="None", inplace=True)
train_set["MiscFeature"].fillna(value="None", inplace=True)
train_set["Fence"].fillna(value="None", inplace=True)
train_set["FireplaceQu"].fillna(value="None", inplace=True)



#ToDo: Look at how many blank values in column. If column is mostly blank, does this imputer make sense?




#Change YearBuilt/YearRemodAdd to a categorized feature (i.e. 50s, 60s, 70s, etc)

train_set["DecadeBuilt"] = (np.floor(train_set["YearBuilt"] / 10) * 10).astype(int)
train_set["DecadeRemodAdd"] = (np.floor(train_set["YearRemodAdd"] / 10) * 10).astype(int)
train_num_col = train_num_col.drop(["YearBuilt", "YearRemodAdd"])
train_num_col = train_num_col.append(pd.Index(["DecadeBuilt", "DecadeRemodAdd"]))

'''
Some of the numeric features are actually categories

Manually reading through the list, these are the ones that identify
MSSubClass: Identifies the type of dwelling involved in the sale.	

        20	1-STORY 1946 & NEWER ALL STYLES
        30	1-STORY 1945 & OLDER
        40	1-STORY W/FINISHED ATTIC ALL AGES
        45	1-1/2 STORY - UNFINISHED ALL AGES
        50	1-1/2 STORY FINISHED ALL AGES
        60	2-STORY 1946 & NEWER
        70	2-STORY 1945 & OLDER
        75	2-1/2 STORY ALL AGES
        80	SPLIT OR MULTI-LEVEL
        85	SPLIT FOYER
        90	DUPLEX - ALL STYLES AND AGES
       120	1-STORY PUD (Planned Unit Development) - 1946 & NEWER
       150	1-1/2 STORY PUD - ALL AGES
       160	2-STORY PUD - 1946 & NEWER
       180	PUD - MULTILEVEL - INCL SPLIT LEV/FOYER
       190	2 FAMILY CONVERSION - ALL STYLES AND AGES

OverallQual: Rates the overall material and finish of the house

       10	Very Excellent
       9	Excellent
       8	Very Good
       7	Good
       6	Above Average
       5	Average
       4	Below Average
       3	Fair
       2	Poor
       1	Very Poor
       
OverallCond: Rates the overall condition of the house

       10	Very Excellent
       9	Excellent
       8	Very Good
       7	Good
       6	Above Average	
       5	Average
       4	Below Average	
       3	Fair
       2	Poor
       1	Very Poor
'''




''' Category Processing'''
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

train_cat = pd.DataFrame(dtype=int)
train_cat_corr = pd.DataFrame()

#go through each column and split OneHotEncode all categories
for col in list(train_str_col):
    #print("Column: ", col)
    train_raw_col = train_set[col].copy()
    #Check for blanks
    imputer = SimpleImputer(strategy="most_frequent")

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



''' Num Process '''
train_num = train_set[train_num_col].copy()
imputer = SimpleImputer(strategy="median", copy=False)
train_num = pd.DataFrame(data=imputer.fit_transform(X=train_num), columns=train_num_col)

'''
Let's find correlations for the columns
'''
#Look at correlation for category
train_cat_corr = pd.concat([train_cat, train_set["SalePrice"]], axis=1).corr()

#Look at correlation for numerical categories
train_num_corr = train_num.corr()



'''
Take columns from both tables and 
'''

THRESHOLD = 0.2

#Let's look at category columns with a > 50% correlation
print(train_cat_corr[np.abs(train_cat_corr["SalePrice"]) > THRESHOLD]["SalePrice"].sort_values(ascending=False))

train_cat_selectcol = train_cat_corr[np.abs(train_cat_corr["SalePrice"]) > THRESHOLD].index
train_cat_selectcol = train_cat_selectcol.drop("SalePrice")
train_cat_select = train_cat[train_cat_selectcol]

#Let's look at numerical columns with a > 50% correlation
print(train_num_corr[np.abs(train_num_corr["SalePrice"]) > THRESHOLD]["SalePrice"].sort_values(ascending=False))

train_num_selectcol = train_num_corr[np.abs(train_num_corr["SalePrice"]) > THRESHOLD].index
train_num_selectcol = train_num_selectcol.drop("SalePrice")
train_num_select = train_num[train_num_selectcol]



#Use feature scaler on the numbers
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(copy=False)
train_num_select_scaled = pd.DataFrame(data=scaler.fit_transform(X=train_num_select), columns=train_num_selectcol)

#Create new dataframe with the selected columns
train_select = train_cat_select.merge(train_num_select_scaled, left_index=True, right_index=True)


#Split train data set to 2 sets - training / testing
from sklearn.model_selection import train_test_split
trainX, testX, trainY, testY = train_test_split(train_select, train_set_Y, test_size=.3, random_state=42)


#Cross Val column with various models - Linear Regression / Ensemble / Forest

from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

print("Linear Regression")
linreg_crossval_score = cross_val_score(estimator=LinearRegression(), X=train_select, y=train_set_Y, cv=10, scoring="neg_mean_squared_error")
displayScores(np.sqrt(-linreg_crossval_score))
print("=====")

print("Kernel Ridge")
kr_crossval_score = cross_val_score(estimator=KernelRidge(), X=train_select, y=train_set_Y, cv=10,  scoring="neg_mean_squared_error")
displayScores(np.sqrt(-kr_crossval_score))
print("=====")

print("SVR")
svr_crossval_score = cross_val_score(estimator=SVR(gamma="scale"), X=train_select, y=train_set_Y, cv=10,  scoring="neg_mean_squared_error")
displayScores(np.sqrt(-svr_crossval_score))
print("=====")

print("Random Forest")
randfor_crossval_score = cross_val_score(estimator=RandomForestRegressor(n_estimators=100), X=train_select, y=train_set_Y, cv=10,  scoring="neg_mean_squared_error")
displayScores(np.sqrt(-randfor_crossval_score))
print("=====")

print("Gradient Boost")
gradboost_crossval_score = cross_val_score(estimator=GradientBoostingRegressor(), X=train_select, y=train_set_Y, cv=10,  scoring="neg_mean_squared_error")
displayScores(np.sqrt(-gradboost_crossval_score))
print("=====")



#Todo: Pipeline

#train_cat_pipeline = Pip
#train_num_pipeline = Pipe



#Imputer
#split cat/num
#decide threshold (Feature selection)
#

#Todo: any new features that can be made based off the existing ones?


#train/test set?
from sklearn.model_selection import cross_val_score


#ToDo: GridSearch to optimize parameters
#Gradient Boost seems to perform best

from sklearn.model_selection import GridSearchCV

param = [ {"learning_rate": [.05,.1,.5],
           "n_estimators": [200,300,400],
           "max_depth": [2,3,4]},
        ]
grid = GridSearchCV(GradientBoostingRegressor(verbose=2, random_state=42), n_jobs=-1, param_grid=param, cv=5)

grid.fit(X=train_select, y=train_set_Y)