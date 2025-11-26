import os
import zipfile as zip
import time
import pandas as pd

#LINUX or PC
OS = "LINUX"

def setConfig(PROJECT_NAME):
    #BASE_PATH = os.popen("cd").read()[:-1]
    # For Mac:
    BASE_PATH = os.popen("pwd").read()[:-1]
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
        print("Deleting dataset files..." + DATASET_PATH)
        if (OS == "PC"):
            os.system("del /Q " + DATASET_PATH)
        if (OS == "LINUX"):
            os.system("rm -rf " + DATASET_PATH)
        else:
            print("no OS selected")
            exit(-1)


    print("Download dataset...")
    os.system(KAGGLE_COMMAND)

    print("Unpack dataset...")
    os.system("unzip " + DOWNLOAD_PATH + "/*.zip" + " -d " + DATASET_PATH)
    #unzip = zip.ZipFile(DOWNLOAD_PATH + "\\" + KAGGLE_COMPETITION + ".zip")
    #unzip.extractall(path=DATASET_PATH)
    #unzip.close()
    print("Delete zip file...")
    if (OS == "PC"):
        os.system("del /Q " + PROJECT_PATH + "\\zip")
    if (OS == "LINUX"):
        os.system("rm -rf " + PROJECT_PATH + "/zip")
    else:
        print("no OS selected")
        exit(-1)

#Load Dataset
#Returns test and train Dataframes
def loadDataSet(PROJECT_NAME, indexName="Id"):
    BASE_PATH, PROJECT_PATH, DATASET_PATH, DOWNLOAD_PATH = setConfig(PROJECT_NAME)

   #if (not os.path.exists(DATASET_PATH)):
   #     getDataSet(KAGGLE_COMPETITION,   PROJECT_NAME)

    train = pd.read_csv( os.path.join(DATASET_PATH, "train.csv"), index_col=indexName)
    test =  pd.read_csv( os.path.join(DATASET_PATH, "test.csv"), index_col=indexName)

    return train, test

#Write Kaggle Prediction File
#output ID and SalesPrice
def writeKagglePrediction(PROJECT_NAME, prefix, predict):
    BASE_PATH, PROJECT_PATH, DATASET_PATH, DOWNLOAD_PATH = setConfig(PROJECT_NAME)

    PREDICT_PATH = PROJECT_PATH + "/predict"
    if (not os.path.exists(PREDICT_PATH)):
        os.mkdir(PREDICT_PATH)

    pd.DataFrame(predict).to_csv(PREDICT_PATH+"/"+prefix+"-"+time.strftime("%Y%m%d%H%M%S", time.localtime())+".csv", index=False)



def findBlankColumns(dataset):
    # Check which columns have blank values
    assert  isinstance(dataset, pd.DataFrame)

    for col in list(dataset):
        missing = dataset[col].isna().sum()
        if missing > 1:
            print(col, ": ", dataset[col].isna().sum()," blank values found (", "%.4f pct" % (missing/len(dataset[col])*100),
                  ")")

