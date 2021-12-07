# Training data scaler.
def scaleTrainData(X, scaler):
    X_train_scaled = scaler.fit_transform(X)
    return X_train_scaled


# Testing data scaler.
def scaleTestData(X, scaler):
    X_test_scaled = scaler.transform(X)
    return X_test_scaled


# Transform back to original form training data.
def inverseScaleTrainData(X, scaler):
    X_original = scaler.inverse_transform(X)
    return X_original


# Transform back to original form test data.
def inverseScaleTestData(X, scaler):
    X_original = scaler.inverse_transform(X)
    return X_original


# GUI data scaler.
def scaleGUIData(X, scaler):
    X_GUI_scaled = scaler.transform(X)
    print("Scaled values:", X_GUI_scaled)
    return X_GUI_scaled
