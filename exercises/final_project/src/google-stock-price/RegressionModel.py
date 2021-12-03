import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def regressionModel(regressor, XTrain, yTrain, XTest, yTest):
    regressor = regressor
    regressor.fit(XTrain, yTrain)

    prediction = regressor.predict(XTest)

    score = np.round(regressor.score(XTest, yTest), 2) * 100
    print("Score:", score)

    r2 = np.round(r2_score(yTest, prediction), 2)
    print("r2:", r2)

    mse = np.round(mean_squared_error(yTest, prediction), 2)
    print("Mean squared error:", mse)

    mae = np.round(mean_absolute_error(yTest, prediction), 2)
    print("Mean absolute error:", mae)

    print("Model coefficients:", regressor.coef_)
    print("Model intercept:", regressor.intercept_)

    return regressor, score, r2, mse, mae


def predictWithModel(regressor, X):
    prediction = regressor.predict(X)
    return prediction
