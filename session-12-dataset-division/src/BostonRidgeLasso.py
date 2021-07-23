import numpy as np  # linear algebra
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold

boston_x, boston_y = datasets.load_boston(return_x_y=True)

print(boston_y)
print(boston_x)

kf = KFold(n_splits=6, shuffle=True)

alpha = 1.0
scores = []

for train_index, test_index in kf.split(boston_x):
    boston_x_train, boston_x_test = boston_x(train_index), boston_x(test_index)
    boston_y_train, boston_y_test = boston_y(train_index), boston_y(test_index)
    model = Ridge(alpha)
    print('alpha: ', alpha)
    alpha = alpha + 1

model.fit(boston_x_train, boston_y_train)
boston_true_y = model.predict(boston_x_test)
accuracy = mean_squared_error(boston_true_y, boston_y_test)
scores = np.append(scores, accuracy)

print(scores)



rkf = RepeatedKFold(n_splits=10, n_repeats=5)
model = svm.SVC()
scores = []
clean_dataset.describe()

for train_index, test_index in rkf.split(clean_dataset):
    x_train, x_test = x_data(train_index), x_data(test_index)
    y_train, y_test = y_data(train_index), y_data(test_index)

    model.fit(x_train, y_train)
    true_y = model.predict(x_test)
    accuracy = accuracy_score(true_y, y_test)
    scores = np.append(scores, accuracy)

print(scores)
