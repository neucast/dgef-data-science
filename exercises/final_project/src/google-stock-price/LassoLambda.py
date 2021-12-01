import warnings

import numpy as np
import pandas as pd
import sklearn
from numpy import arange
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

from FileManager import getInputPath

# Configure.
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

# Data set path.
inputPath = getInputPath("base-final-regresion.csv")

# Reads the CSV data file.
googleStockPriceDataFrame = pd.read_csv(inputPath, delimiter=",", encoding="ISO-8859-1")

# Convert date field from string to Date format.
googleStockPriceDataFrame["date"] = pd.to_datetime(googleStockPriceDataFrame.date, infer_datetime_format=True)

# Sort by date.
googleStockPriceOrderedDataFrame = googleStockPriceDataFrame.sort_values(by="date")

# Truncate not useful columns.
truncatedDataFrame = googleStockPriceDataFrame.drop(columns=["symbol", "date"])
truncatedDataFrameColumns = list(truncatedDataFrame.columns)

# Normalize data.
# Normalization scales each input variable separately to the range [0-1], which is the range for floating-point
# values where the computing will have the most numeric precision.
scaledtruncatedNPArray = sklearn.preprocessing.scale(truncatedDataFrame, axis=0, with_mean=True, with_std=True,
                                                     copy=True)
scaledtruncatedDataFrame = pd.DataFrame(scaledtruncatedNPArray)
scaledtruncatedDataFrame.columns = truncatedDataFrameColumns

# Define variables.
dependentVariables = ["close"]
independentVariables = ["volume", "open", "high", "low", "adjVolume", "adjOpen", "adjClose", "adjHigh", "adjLow",
                        "divCash", "splitFactor"]

X = scaledtruncatedDataFrame[independentVariables]
y = scaledtruncatedDataFrame[dependentVariables]

trainPercentage = 0.7  # 70% of data for training purposes, 30% for testing.
X_train = X[:int(X.shape[0] * trainPercentage)]
X_test = X[int(X.shape[0] * trainPercentage):]
y_train = y[:int(X.shape[0] * trainPercentage)]
y_test = y[int(X.shape[0] * trainPercentage):]

# Scikit-learn offers a function for time-series validation, TimeSeriesSplit. The function splits training data into
# multiple segments. We use the first segment to train the model with a set of hyper-parameters, to test it with the second.
# Then we train the model with first two chunks and measure it with the third part of the data. In this way we
# do k-1 times of cross-validation.

print("\nMethod 1: Lasso lambda compute by Time series split with Grid-Search cross validation.")
tscv = TimeSeriesSplit(n_splits=5)
model = Lasso(normalize=True)
# cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
grid = dict()
grid["alpha"] = arange(0, 1, 0.001)
search = GridSearchCV(model, grid, scoring='neg_mean_absolute_error', cv=tscv, n_jobs=1)
results = search.fit(X_train, y_train)
print("Mean absolute error: %.3f" % results.best_score_)
print("Best lambda value: %s" % results.best_params_)

print("\nMethod 2: Lasso lambda compute by sklearn.linear_model Ridge cross validation.")
alphas = 10 ** np.linspace(-1, 1, 1000) * 0.0005
lassocv = LassoCV(alphas=alphas, normalize=True)
lassocv.fit(X_train, y_train)
#print("Mean squared error: %3f" % lassocv.score())
print("Best lambda value: %3f" % lassocv.alpha_)
