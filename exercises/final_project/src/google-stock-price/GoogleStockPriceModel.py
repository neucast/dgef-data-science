# Regression model.

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler

from DataPlotter import plotActualVsPredictedData, plotCompareOriginalClosePriceVsPredictedClosePrice
from DataScaler import scaleTrainData, scaleTestData
from FileManager import getInputPath
from LambdaCompute import ridgeLambdaCompute, lassoLambdaCompute
from RegressionModel import regressionModel, predictWithModel, predictNextDays
from SplitTrainAndTestData import get_X_Matrix, get_y_Matrix, get_X_TrainData, get_X_TestData, get_y_TrainData, \
    get_y_TestData, get_X_TestDataWithOutDate, get_X_TrainDataWithOutDate

# Configure.
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

# Data set path.
inputPath = getInputPath("base-final-regresion.csv")

# Reads the CSV data file.
googleStockPriceDataFrame = pd.read_csv(inputPath, delimiter=",", encoding="ISO-8859-1")
googleStockPriceDataFrame.dataframeName = "base-final-regresion.csv"
nRow, nCol = googleStockPriceDataFrame.shape
print(f"There are {nRow} rows and {nCol} columns in the data set.")

# Convert date field from string to Date format.
googleStockPriceDataFrame["date"] = pd.to_datetime(googleStockPriceDataFrame.date, infer_datetime_format=True)

# Sort by date.
orderedDataFrame = googleStockPriceDataFrame.sort_values(by="date")

independentVariables = ["date", "high", "low", "open", "volume"]
independentTrainingVariables = ["high", "low", "open", "volume"]
dependentVariables = ["close"]

X = get_X_Matrix(orderedDataFrame, independentVariables)
y = get_y_Matrix(orderedDataFrame, dependentVariables)

# Split data in trainning and test data sets.
trainPercentage = 0.7  # 70% of data for training purposes, 30% for testing.
XX_train = get_X_TrainData(X, trainPercentage)
XX_test = get_X_TestData(X, trainPercentage)
y_train = get_y_TrainData(X, y, trainPercentage)
y_test = get_y_TestData(X, y, trainPercentage)

print("Test data starting date : {}".format(XX_test["date"].min()))
print("Test data ending   date : {}".format(XX_test["date"].max()))

# Get data without the date.
X_train = get_X_TrainDataWithOutDate(XX_train, independentTrainingVariables)
X_test = get_X_TestDataWithOutDate(XX_test, independentTrainingVariables)

# Normalize data.
# Normalization scales each input variable separately to the range 0-1, which is the range for floating-point
# values where the computing will have the most numeric precision.
scaler = StandardScaler()
X_train = scaleTrainData(X_train, scaler)
X_test = scaleTestData(X_test, scaler)

print("\n")

# ---------------------------------- 1.- Linear Regression ----------------------------------
print("----------------------------- 1.- Linear Regression start -----------------------------")
regressor = LinearRegression(fit_intercept=True, normalize=True)
linearRegressor, score, r2, meanSquaredError, rootMeanSquaredError, meanAbsoluteError = regressionModel(regressor,
                                                                                                        X_train,
                                                                                                        y_train, X_test,
                                                                                                        y_test)
prediction = predictWithModel(linearRegressor, X_test)

plotActualVsPredictedData(prediction, XX_test, y_test, "date", "close",
                          "Linear regression model - Predicted and Actual closing prices of Google", "Year",
                          "Close price (USD)")
print("----------------------------- Linear Regression end -----------------------------")

print("\n")

# ---------------------------------- 2.- Ridge Regression ----------------------------------
print("----------------------------- 2.- Ridge Regression start -----------------------------")
# Compute hyperparameter value.
alphas = 10 ** np.linspace(-1, 1, 1000) * 0.0005
ridgeLambda = ridgeLambdaCompute(alphas, X_train, y_train)

regressor = Ridge(alpha=ridgeLambda, fit_intercept=True, normalize=True)
ridgeRegressor, score, r2, meanSquaredError, rootMeanSquaredError, meanAbsoluteError = regressionModel(regressor,
                                                                                                       X_train, y_train,
                                                                                                       X_test,
                                                                                                       y_test)
prediction = predictWithModel(ridgeRegressor, X_test)

plotActualVsPredictedData(prediction, XX_test, y_test, "date", "close",
                          "Ridge regression model - Predicted and Actual closing prices of Google", "Year",
                          "Close price (USD)")
print("----------------------------- Ridge Regression end -----------------------------")

print("\n")

# ---------------------------------- 3.- Lasso Regression ----------------------------------
print("----------------------------- 3.- Lasso Regression start -----------------------------")
# Compute hyperparameter value.
alphas = 10 ** np.linspace(-1, 1, 1000) * 0.0005
lassoLambda = lassoLambdaCompute(alphas, X_train, y_train)

regressor = Lasso(alpha=lassoLambda, fit_intercept=True, normalize=True)
lassoRegressor, score, r2, meanSquaredError, rootMeanSquaredError, meanAbsoluteError = regressionModel(regressor,
                                                                                                       X_train, y_train,
                                                                                                       X_test,
                                                                                                       y_test)
prediction = predictWithModel(lassoRegressor, X_test)

plotActualVsPredictedData(prediction, XX_test, y_test, "date", "close",
                          "Lasso regression model - Predicted and Actual closing prices of Google", "Year",
                          "Close price (USD)")
print("----------------------------- Lasso Regression end -----------------------------")

print("\n")
