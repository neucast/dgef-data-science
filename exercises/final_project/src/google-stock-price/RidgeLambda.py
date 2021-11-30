import warnings

import numpy as np
import pandas as pd
import sklearn
from numpy import arange
from patsy.highlevel import dmatrices
from sklearn.linear_model import LinearRegression, Lasso, RidgeCV, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from FileManager import getInputPath
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import numpy as np

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
# print("Type of googleStockPriceDataFrame:", type(googleStockPriceDataFrame))

# Exploring the data set.
# print(googleStockPriceDataFrame.head(5))
# Convert date field from string to Date format.
googleStockPriceDataFrame["date"] = pd.to_datetime(googleStockPriceDataFrame.date, infer_datetime_format=True)
# print(googleStockPriceDataFrame.head())

# Sort by date.
googleStockPriceOrderedDataFrame = googleStockPriceDataFrame.sort_values(by="date")

# Truncate not useful columns.
truncatedDataFrame = googleStockPriceDataFrame.drop(columns=["symbol", "date"])
# print(truncatedDataFrame.head(5))
# print("Type of truncatedDataFrame:", type(truncatedDataFrame))
truncatedDataFrameColumns = list(truncatedDataFrame.columns)

scaledtruncatedNPArray = sklearn.preprocessing.scale(truncatedDataFrame, axis=0, with_mean=True,
                                                     with_std=True, copy=True)

# print("Type of scaledtruncatedNPArray:", type(scaledtruncatedNPArray))

scaledtruncatedDataFrame = pd.DataFrame(scaledtruncatedNPArray)
scaledtruncatedDataFrame.columns = truncatedDataFrameColumns
# print("Type of scaledtruncatedDataFrame:", type(scaledtruncatedDataFrame))
# print(scaledtruncatedDataFrame.head(5))

dependentVariables = ["close"]
independentVariables = ["volume", "open", "high", "low", "adjVolume", "adjOpen", "adjClose",
                        "adjHigh", "adjLow", "divCash", "splitFactor"]

X = scaledtruncatedDataFrame[independentVariables]
y = scaledtruncatedDataFrame[dependentVariables]

# print(X.head(5))
# print(y.head(5))

trainPercentage = 0.7  # 70% of data for training purposes, 30% for testing.
X_train = X[:int(X.shape[0] * trainPercentage)]
X_test = X[int(X.shape[0] * trainPercentage):]
y_train = y[:int(X.shape[0] * trainPercentage)]
y_test = y[int(X.shape[0] * trainPercentage):]

# print(X_train.head(5))
# print(X_test.head(5))
# print(y_train.head(5))
# print(y_test.head(5))

# tscv = TimeSeriesSplit(n_splits=5)
# i = 1
# score = []

# Scikit-learn offers a function for time-series validation, TimeSeriesSplit. The function splits training data into
# multiple segments. We use the first segment to train the model with a set of hyper-parameter, to test it with the second.
# Then we train the model with first two chunks and measure it with the third part of the data. In this way we
# do k-1 times of cross-validation. Grid-search of hyper-parameter with TimeSeriesSplit requires some manual coding.

tscv = TimeSeriesSplit(n_splits=5)
model = Ridge(normalize=True)
# cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
grid = dict()
grid["alpha"] = arange(0, 1, 0.1)
search = GridSearchCV(model, grid, scoring='neg_mean_absolute_error', cv=tscv, n_jobs=1)
results = search.fit(X_train, y_train)
print("MAE: %.3f" % results.best_score_)
print("Config: %s" % results.best_params_)

# alphas = 10 ** np.linspace(-1, 1, 1000) * 0.0005
# print(alphas)
# ridgecv = RidgeCV(alphas=alphas, scoring='neg_mean_squared_error', normalize=True)
# ridgecv.fit(X_train, y_train)
# print(ridgecv.alpha_)

# for tr_index, val_index in tscv.split(X_train):
#     # print("tr_index", tr_index)
#     # print("val_index", val_index)
#     X_tr, X_val = X_train[tr_index], X_train[val_index]
#     y_tr, y_val = y_train[tr_index], y_train[val_index]
#     for mf in np.linspace(100, 150, 6):
#         for ne in np.linspace(50, 100, 6):
#             for md in np.linspace(20, 40, 5):
#                 for msl in np.linspace(30, 100, 8):
#                     rfr = RandomForestRegressor(
#                         max_features=int(mf),
#                         n_estimators=int(ne),
#                         max_depth=int(md),
#                         min_samples_leaf=int(msl))
#                     rfr.fit(X_tr, y_tr)
#                     score.append([i,
#                                   mf,
#                                   ne,
#                                   md,
#                                   msl,
#                                   rfr.score(X_val, y_val)])
#     i += 1
#
# print(score)
