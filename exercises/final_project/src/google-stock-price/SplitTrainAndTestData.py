def get_X_Matrix(df, independentVariables):
    X = df[independentVariables]
    return X


def get_y_Matrix(df, dependentVariables):
    y = df[dependentVariables]
    return y


def get_X_TrainData(X, trainPercentage):
    XX_train = X[:int(X.shape[0] * trainPercentage)]
    return XX_train


def get_X_TestData(X, trainPercentage):
    XX_test = X[int(X.shape[0] * trainPercentage):]
    return XX_test


def get_y_TrainData(X, y, trainPercentage):
    y_train = y[:int(X.shape[0] * trainPercentage)]
    return y_train


def get_y_TestData(X, y, trainPercentage):
    y_test = y[int(X.shape[0] * trainPercentage):]
    return y_test


def get_X_TrainDataWithOutDate(X, independentVariables):
    X_train = X[independentVariables]
    return X_train


def get_X_TestDataWithOutDate(X, independentVariables):
    X_test = X[independentVariables]
    return X_test
