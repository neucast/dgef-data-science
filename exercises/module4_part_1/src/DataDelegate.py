import pandas as pd

from FileManager import getInputPath

# Configure.
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.options.display.float_format = "{:,.5f}".format


def getDataFrame(fileName):
    dataFrame = pd.read_csv(getInputPath(fileName), dtype="str", encoding="ISO-8859-1")

    dataFrame[dataFrame.columns] = dataFrame[dataFrame.columns].astype(float)

    print(dataFrame.head())

    return dataFrame
