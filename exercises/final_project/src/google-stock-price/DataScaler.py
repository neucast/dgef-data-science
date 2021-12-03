# Training data scaler.
def scaleTrainData(X, scaler):
    X_train_scaled = scaler.fit_transform(X)
    return X_train_scaled


# Testing data scaler.
def scaleTestData(X, scaler):
    X_test_scaled = scaler.transform(X)
    return X_test_scaled
