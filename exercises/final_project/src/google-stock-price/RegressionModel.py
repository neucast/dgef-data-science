import math

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score, \
    mean_gamma_deviance, mean_poisson_deviance


# Compute using the given regression model.
def regressionModel(regressor, XTrain, yTrain, XTest, yTest):
    regressor = regressor
    regressor.fit(XTrain, yTrain)

    prediction = regressor.predict(XTest)

    score = np.round(regressor.score(XTest, yTest), 2) * 100
    print("Score:", score)

    # R2 score for regression R - squared(R2)
    # Is a statistical measure that represents the proportion of the variance for a dependent variable
    # that's explained by an independent variable or variables in a regression model.
    # 1 = Best 0 or < 0 = worse
    r2 = np.round(r2_score(yTest, prediction), 2)
    print("R2 score:", r2)

    # Evaluation metrices RMSE, MSE and MAE Root Mean Square Error(RMSE), Mean Square Error(MSE) and
    # Mean absolute Error(MAE) are a standard way to measure the error of a model in predicting quantitative data.
    mse = np.round(mean_squared_error(yTest, prediction), 2)
    print("Mean squared error:", mse)

    rmse = math.sqrt(mse)
    print("Root mean squared error:", rmse)

    mae = np.round(mean_absolute_error(yTest, prediction), 2)
    print("Mean absolute error:", mae)

    # Explained variance regression score
    # The explained variance score explains the dispersion of errors of a given dataset, and the formula is
    # written as follows: Here Var(y) is the variance of prediction errors and actual values respectively.
    # Scores close to 1.0 are highly desired, indicating better squares of standard deviations of errors.
    train_predict = regressor.predict(XTrain)
    test_predict = prediction
    print("Train data explained variance regression score:", explained_variance_score(yTrain, train_predict))
    print("Test data explained variance regression score:", explained_variance_score(yTest, test_predict))

    print("Train data R2 score:", r2_score(yTrain, train_predict))
    print("Test data R2 score:", r2_score(yTest, test_predict))

    print("Train data Mean Gamma Deviance: ", mean_gamma_deviance(yTrain, train_predict))
    print("Test data Mean Gamma Deviance: ", mean_gamma_deviance(yTest, test_predict))

    print("Train data Mean Poisson Deviance: ", mean_poisson_deviance(yTrain, train_predict))
    print("Test data Poisson Deviance: ", mean_poisson_deviance(yTest, test_predict))

    print("Model coefficients:", regressor.coef_)
    print("Model intercept:", regressor.intercept_)

    return regressor, score, r2, mse, rmse, mae


# Compute a prediction data set using the given regression model.
def predictWithModel(regressor, X):
    prediction = regressor.predict(X)
    return prediction
