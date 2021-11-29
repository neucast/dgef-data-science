# https://towardsdatascience.com/from-linear-regression-to-ridge-regression-the-lasso-and-the-elastic-net-4eaecaf5f7e6

import warnings

import numpy as np
import pandas as pd
import sklearn
from numpy import arange
from sklearn.linear_model import LinearRegression, Lasso, RidgeCV, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.model_selection import train_test_split

from FileManager import getInputPath


def getPrediction(x, w):
    return (np.matmul(x, w))


# The basic loss, sum over the difference of y-ypred.
def loss(y, ypred):
    l = (y - ypred) ** 2
    return (l.sum())


# The Mean Square Error.
def meanSquareError(x, y, w):
    return ((1 / x.shape[0]) * loss(y, getPrediction(x, w)))


# This function computes the weights.
def gradientDescent(x, y, learnRate=0.01, epochs=2000, reg=0):
    global cacheLoss
    cacheLoss = [None] * epochs

    # Random Initialization of weights.
    weights = np.random.rand(x.shape[1])

    weights = np.array(weights)
    weights = weights.reshape(-1, 1)
    m = x.shape[0]

    for i in range(epochs):

        predictions = getPrediction(x, weights)
        cacheLoss[i] = loss(y, predictions)

        weights[0] = weights[0] - (1 / m) * learnRate * (np.matmul(x[:, 0].transpose(), predictions - y))

        for j in range(1, len(weights)):
            weights[j] = weights[j] - (1 / m) * learnRate * (
                    np.matmul(x[:, j].transpose(), predictions - y) + sum(np.dot(weights[j], reg)))
    return (weights)


def getRidgeLambda(x, y):
    bestMeanSquareError = 10e100

    lamList = [l * 0.05 for l in range(0, 300)]

    global ridgeLambda

    for l in lamList:
        wr = gradientDescent(x, y, reg=l)
        if meanSquareError(X_Validate, Y_Validate, wr) < bestMeanSquareError:
            bestMeanSquareError = meanSquareError(X_Validate, Y_Validate, wr)
            ridgeLambda = l

    return (ridgeLambda)


# Cross Validation to get Lasso Paramaters
def getLassoLambda(x, y):
    bestMeanSquareError = 10e100

    alphaList = [l * 0.1 for l in range(1, 200)]

    for a in alphaList:

        lassoModel = Lasso(alpha=a, max_iter=5000, fit_intercept=False)

        lassoModel.fit(x, y)

        getPrediction = lassoModel.predict(X_Validate).reshape(-1, 1)

        meanSquareError = sum((Y_Validate - getPrediction) ** 2)
        if meanSquareError < bestMeanSquareError:
            bestMeanSquareError = meanSquareError
            lassoLambda = a
    return (lassoLambda)


# Search for ideal parameters using cross validation
def getParametersElasticNet(x, y):
    bestMeanSquareError = 10e100

    regList = [l * 0.1 for l in range(1, 500)]
    ratio = [i * 0.1 for i in range(1, 200)]
    global bestAlpha
    global bestRatio
    global bestElasticweights

    for l1 in regList:
        for r in ratio:
            elasticModel = sklearn.linear_model.ElasticNet(
                alpha=l1, l1_ratio=r, fit_intercept=False,
                max_iter=3000, tol=1e-5)

        elasticModel.fit(x, y)
        getPrediction = elasticModel.predict(X_Validate).reshape(-1, 1)

        meanSquareError = sum((Y_Validate - getPrediction) ** 2)
        if meanSquareError < bestMeanSquareError:
            bestMeanSquareError = meanSquareError
            bestAlpha = l1
            bestRatio = r
            bestElasticweights = elasticModel.coef_
    return (bestElasticweights)


# Configure.
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

# Data set path.
inputPath = getInputPath("base-final-regresion.csv")

# Reads the CSV data file.
googleStockPriceDataSet = pd.read_csv(inputPath, delimiter=",", encoding="ISO-8859-1")
googleStockPriceDataSet.dataframeName = "base-final-regresion.csv"
nRow, nCol = googleStockPriceDataSet.shape
print(f"There are {nRow} rows and {nCol} columns in the data set.")

# Exploring the data set.
# print(googleStockPriceDataSet.head(5))

# Truncate not useful columns.
truncatedDataSet = googleStockPriceDataSet.drop(columns=["symbol", "date"])
# print(truncatedDataSet.head(5))

dependentVariables = ["close"]
independentVariables = ["volume", "open", "high", "low", "adjVolume", "adjOpen", "adjClose", "adjHigh",
                        "adjLow", "divCash", "splitFactor"]

test_sample_size = 0.2
x_train, x_test, y_train, y_test = train_test_split(
    truncatedDataSet[independentVariables],
    truncatedDataSet[dependentVariables],
    test_size=test_sample_size,
    random_state=1)

# print(x_train.head(5))
# print(x_test.head(5))

x_train_scaled = sklearn.preprocessing.scale(x_train, axis=0, with_mean=True,
                                             with_std=True, copy=True)

x_test_scaled = sklearn.preprocessing.scale(x_test, axis=0, with_mean=True,
                                            with_std=True, copy=True)
# print(x_train_scaled)
# print(x_test_scaled)

# Turn into numpy objects, y should be a column vector
x_train_scaled = np.array(x_train_scaled)

x_test_scaled = np.array(x_test_scaled)

y_train = np.array(y_train)
y_train = y_train.reshape(-1, 1)

y_test = np.array(y_test)
y_test = y_test.reshape(-1, 1)

alphas = 10 ** np.linspace(10, -2, 100) * 0.5
print(alphas)
ridgecv = RidgeCV(alphas=alphas, scoring='neg_mean_squared_error', normalize=True)
ridgecv.fit(x_train, y_train)
print(ridgecv.alpha_)

ridge = Ridge(alpha=ridgecv.alpha_, normalize=True)
ridge.fit(x_train, y_train)
mean_squared_error(y_test, ridge.predict(x_test))
ridgeCoefs = ridge.coef_
ridgeCoefsDataFrame = pd.DataFrame(ridgeCoefs, columns=independentVariables)
print(ridgeCoefsDataFrame)

model = Ridge(normalize=True)
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
grid = dict()
grid["alpha"] = arange(-1, 1, 0.001)
search = GridSearchCV(model, grid, scoring='neg_mean_squared_error', cv=cv, n_jobs=1)
results = search.fit(x_train, y_train)
print("MAE: %.3f" % results.best_score_)
print("Config: %s" % results.best_params_)

# # we need a column of ones to add as a bias term.
# addBias = np.ones([x_train_scaled.shape[0], 1])
# x_train_scaled = np.append(addBias, x_train_scaled, axis=1)
#
# addBias = np.ones([x_test_scaled.shape[0], 1])
# x_test_scaled = np.append(addBias, x_test_scaled, axis=1)
#
# # Get Linear Estimates
# wlinear = gradientDescent (x_train_scaled, y_train)
# print("wlinear", wlinear)
#
# meanSquaredErrorlinear = meanSquareError(x_test_scaled, y_test, wlinear)
# print("meanSquaredErrorlinear=", meanSquaredErrorlinear)
#
# X_train, X_Validate, Y_train, Y_Validate = sklearn.model_selection.train_test_split(x_train_scaled, y_train,
#                                                                                     test_size=test_sample_size,
#                                                                                     random_state=1)
#
# ridgeLambda = getRidgeLambda(X_train, Y_train)
# print("ridgeLambda=",
#       ridgeLambda)  # ridgeLambda= 1.0 ridgeLambda= 3.25 ridgeLambda= 0.6000000000000001 ridgeLambda= 0.8500000000000001
#
# wridge = gradientDescent (x_train_scaled, y_train, reg=ridgeLambda)
# print("wridge=", wridge)
# RidgemeanSquaredError = meanSquareError(x_test_scaled, y_test, wridge)
# print("RidgemeanSquaredError", RidgemeanSquaredError)
#
# lassoLambda = getLassoLambda(X_train, Y_train)
# print(f"The ideal lambda for Lasso is {lassoLambda}")  # The ideal lambda for Lasso is 0.6000000000000001
#
# fitLasso = sklearn.linear_model.Lasso(alpha=lassoLambda, fit_intercept=False)
#
# fitLasso.fit(x_train_scaled, y_train)
#
# wlasso = fitLasso.coef_
#
# pz = fitLasso.predict(x_test_scaled).reshape(-1, 1)
#
# LassomeanSquaredError = (1 / x_test_scaled.shape[0]) * sum((y_test - pz) ** 2)
#
# print("wlasso", wlasso)
#
# LassomeanSquaredError = (1 / x_test_scaled.shape[0]) * sum((y_test - pz) ** 2)
#
# print("LassomeanSquaredError=", LassomeanSquaredError)
#
# # wlinear [[ 1.21052218e+03]
# #  [-3.17744251e-01]
# #  [ 5.11395812e+01]
# #  [ 5.50858367e+01]
# #  [ 5.45337528e+01]
# #  [-5.50049696e-01]
# #  [ 5.17932190e+01]
# #  [ 5.87049179e+01]
# #  [ 5.46218194e+01]
# #  [ 5.43774887e+01]]
# # meanSquaredErrorlinear= 1117.4663527687642
#
# # ridgeLambda= 0.8500000000000001
# # wridge= [[ 1.21052218e+03]
# #  [-5.46340816e-01]
# #  [ 5.12599125e+01]
# #  [ 5.48750515e+01]
# #  [ 5.47057175e+01]
# #  [-3.17162388e-01]
# #  [ 5.15657071e+01]
# #  [ 5.82561107e+01]
# #  [ 5.48492940e+01]
# #  [ 5.46989526e+01]]
# # RidgemeanSquaredError 1118.8590228352357
#
# # The ideal lambda for Lasso is 0.6000000000000001
# # wlasso [ 1.20992219e+03 -0.00000000e+00  2.96005865e+00  1.56797449e+02
# #   4.67632051e+01 -1.00610876e+00  0.00000000e+00  1.73261292e+02
# #   0.00000000e+00  0.00000000e+00]
# # LassomeanSquaredError= [1097.5292512]
# #
#
# elasticweights = getParametersElasticNet(X_train, Y_train)
# print(f"The ideal alpha for elastic net is {bestAlpha} and the best ratio is {bestRatio}")
# # The ideal alpha for elastic net is 0.1 and the best ratio is 1.5
#
# # plug in ideal parameters
# fitElastic = sklearn.linear_model.ElasticNet(alpha=bestAlpha, l1_ratio=bestRatio, fit_intercept=False)
# fitElastic.fit(x_train_scaled, y_train)
#
# welastic = fitElastic.coef_
# print("welastic", welastic)
#
# pz = fitElastic.predict(x_test_scaled).reshape(-1, 1)
#
# ElasticmeanSquaredError = (1 / x_test_scaled.shape[0]) * sum((y_test - pz) ** 2)
# print("ElasticmeanSquaredError", ElasticmeanSquaredError)
