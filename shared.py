import os
import zipfile as zip
import pandas as pd


def setConfig(PROJECT_NAME):
    BASE_PATH = os.popen("cd").read()[:-1]
    # For Mac: BASE_PATH = os.popen("pwd").read()[:-1]
    PROJECT_PATH = os.path.join(BASE_PATH, PROJECT_NAME)
    DATASET_PATH = os.path.join(PROJECT_PATH, "dataset")
    DOWNLOAD_PATH = os.path.join(DATASET_PATH, "zip")

    return BASE_PATH, PROJECT_PATH, DATASET_PATH, DOWNLOAD_PATH

#Download and unzip Kaggle dataset
def getDataSet(KAGGLE_COMPETITION, PROJECT_NAME):
    BASE_PATH, PROJECT_PATH, DATASET_PATH, DOWNLOAD_PATH = setConfig(PROJECT_NAME)

    KAGGLE_COMMAND = "kaggle competitions download -c " + KAGGLE_COMPETITION + " -p " + DOWNLOAD_PATH

    try:
        os.mkdir(DOWNLOAD_PATH)
    except OSError:
        print("Directory already exists")

    print("Download dataset...")
    os.system(KAGGLE_COMMAND)

    print("Unpack dataset...")
    #os.system("unzip " + DOWNLOAD_PATH + "/*.zip" + " -d " + DATASET_PATH)
    unzip = zip.ZipFile(DOWNLOAD_PATH + "\\" + KAGGLE_COMPETITION + ".zip")

    unzip.extractall(path=DATASET_PATH)
    unzip.close()
    # print("Delete zip file...")
    # os.system("del /Q " + DOWNLOAD_PATH + "\\*")
    #For Mac os.system("rm " + DOWNLOAD_PATH + "/*")


#Load Dataset
#Returns test and train Dataframes
def loadDataSet(PROJECT_NAME, indexName="Id"):
    BASE_PATH, PROJECT_PATH, DATASET_PATH, DOWNLOAD_PATH = setConfig(PROJECT_NAME)

    if (not os.path.exists(DATASET_PATH)):
        getDataSet()

    train = pd.read_csv( os.path.join(DATASET_PATH, "train.csv"), index_col=indexName)
    test =  pd.read_csv( os.path.join(DATASET_PATH, "test.csv"), index_col=indexName)

    return train, test

#Write Kaggle Prediction File
#output ID and SalesPrice
def writeKagglePrediction(testSet, predict):
    None


def findBlankColumns(dataset):
    # Check which columns have blank values
    assert  isinstance(dataset, pd.DataFrame)

    for col in list(dataset):
        missing = dataset[col].isna().sum()
        if missing > 1:
            print(col, ": ", dataset[col].isna().sum()," blank values found (", "%.4f pct" % (missing/len(dataset[col])*100),
                  ")")
