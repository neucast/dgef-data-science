# Binary trees.
import os
import numpy as np
import pandas as pd
from sklearn.tree import \
    DecisionTreeRegressor  # Source: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
from sklearn.tree import \
    DecisionTreeClassifier  # Source: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# inputPath to the CSV file.
inputPath = os.path.join(os.path.expanduser("~"), "development", "dgef-data-science",
                         "exercises", "module3_part_1",
                         "data", "2D-binary-tree-example.csv")

# Prints the absolute inputPath to the CSV file.
print("The input CSV file is: ", inputPath)

# Reads the CSV data file.
sourceMatrix = pd.read_csv(inputPath, dtype='str', encoding="ISO-8859-1")
sourceMatrix[["Y", "X1", "X2"]] = sourceMatrix[["Y", "X1", "X2"]].astype(float)
print(sourceMatrix.head())


# Mean square error function.
def meanSquareErrorFunction(v1, v2):
    accumulator, columns = 0, len(v1)

    for i in range(columns):
        accumulator = accumulator + (v1[1] - v2[1]) ** 2
    return (accumulator ** 0.5) / columns


# Binary tree construction.
# Transforming data as a Numpy array.
X_training_set = np.asarray(sourceMatrix[["X1", "X2"]].copy(deep=True).reset_index(drop=True))
X_testing_set = np.asarray(sourceMatrix[["X1", "X2"]].copy(deep=True).reset_index(drop=True))
Y_training_set = np.asarray(sourceMatrix[["Y"]].copy(deep=True).reset_index(drop=True))
Y_testing_set = np.asarray(sourceMatrix[["Y"]].copy(deep=True).reset_index(drop=True))

# Calculation using python scikit-learn defined functions.
# scikit-learn method comparison.

# Decision tree regressor method.
decisionTreeRegressor = DecisionTreeRegressor(max_depth=2)
decisionTreeRegressor.fit(X_training_set, Y_training_set)
Y_predicted_set1 = decisionTreeRegressor.predict(X_testing_set)
meanSquareError1 = meanSquareErrorFunction(Y_predicted_set1, Y_testing_set)
print("Decision Tree Regressor (DTR) mean square error = ", meanSquareError1)

# Decision tree classification procedure using the entropy function.
decisionTreeClassifier = DecisionTreeClassifier(criterion='entropy')
decisionTreeClassifier.fit(X_training_set, Y_training_set)
Y_predicted_set2 = decisionTreeClassifier.predict(X_testing_set)
meanSquareError2 = meanSquareErrorFunction(Y_predicted_set2, Y_testing_set)
print("Decision Tree Classifier (DTC) with entropy criterion mean square error = ", meanSquareError2)

# Data frame definition to compare the results.
compareMatrix = pd.DataFrame(index=range(len(Y_predicted_set2)),
                             columns=["Real data", "DTR Approximate data", "DTC Approximate data"])
compareMatrix["Real data"] = Y_testing_set
compareMatrix["DTR Approximate data"] = Y_predicted_set1
compareMatrix["DTC Approximate data"] = Y_predicted_set2

print("Prediction compare matrix:")
print(compareMatrix)
