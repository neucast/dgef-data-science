# Get independent variables "X" matrix.
def get_X_Matrix(df, independentVariables):
    X = df[independentVariables]
    return X


# Get dependent variables "y" matrix.
def get_y_Matrix(df, dependentVariables):
    y = df[dependentVariables]
    return y


# Get independent variables training data.
def get_X_TrainData(X, trainPercentage):
    XX_train = X[:int(X.shape[0] * trainPercentage)]
    return XX_train


# Get independent variables testing data.
def get_X_TestData(X, trainPercentage):
    XX_test = X[int(X.shape[0] * trainPercentage):]
    return XX_test


# Get dependent variables training data.
def get_y_TrainData(X, y, trainPercentage):
    y_train = y[:int(X.shape[0] * trainPercentage)]
    return y_train


# Get dependent variables testing data.
def get_y_TestData(X, y, trainPercentage):
    y_test = y[int(X.shape[0] * trainPercentage):]
    return y_test


# Get independent variables training data without date data.
def get_X_TrainDataWithOutDate(X, independentVariables):
    X_train = X[independentVariables]
    return X_train


# Get independent variables testing data without date data.
def get_X_TestDataWithOutDate(X, independentVariables):
    X_test = X[independentVariables]
    return X_test
