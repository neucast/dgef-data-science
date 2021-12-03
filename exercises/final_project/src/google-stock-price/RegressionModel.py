import math

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score, \
    mean_gamma_deviance, mean_poisson_deviance

from sklearn.preprocessing import MinMaxScaler


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


def predictNextDays(df, regressor, time_step, pred_days):
    # closedf = df[["date", "close"]]
    closedf = df[["close"]]
    scaler = MinMaxScaler(feature_range=(0, 1))
    closedf = scaler.fit_transform(np.array(closedf).reshape(-1, 1))
    training_size = int(len(closedf) * 0.60)
    test_size = len(closedf) - training_size
    train_data, test_data = closedf[0:training_size, :], closedf[training_size:len(closedf), :1]

    # time_step = 15
    x_input = test_data[len(test_data) - time_step:].reshape(1, -1)
    print(x_input)
    temp_input = list(x_input)
    print("temp_input 1", temp_input)
    temp_input = temp_input[0].tolist()
    print("temp_input 2", temp_input)

    lst_output = []
    n_steps = time_step
    i = 0
    # pred_days = 30
    while (i < pred_days):

        if (len(temp_input) > time_step):

            print("Yes.")

            x_input = np.array(temp_input[1:])
            print("{} day input {}".format(i, x_input))
            x_input = x_input.reshape(1, -1)
            x_input = x_input.reshape((1, n_steps, 1))

            yhat = regressor.predict(x_input)
            print("{} day output {}".format(i, yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]
            print(temp_input)

            lst_output.extend(yhat.tolist())
            i = i + 1

        else:
            print("No.")
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = regressor.predict(x_input)
            temp_input.extend(yhat[0].tolist())

            lst_output.extend(yhat.tolist())
            i = i + 1

    print("Output of predicted next days: ", len(lst_output))
