# Binary trees.

import os
import numpy as np
import pandas as pd

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
    print("accumulator", accumulator)
    print("columns", columns)
    for i in range(columns):
        accumulator = accumulator + (v1[1] - v2[1]) ** 2
    return (accumulator ** 0.5) / columns


# Binary tree construction.

# Transforming data as a Numpy array.
X_training_set = np.asarray(sourceMatrix[["X1", "X2"]].copy(deep=True).reset_index(drop=True))
print("X_training_set")
print(X_training_set)
X_testing_set = np.asarray(sourceMatrix[["X1", "X2"]].copy(deep=True).reset_index(drop=True))
print("X_testing_set")
print(X_testing_set)
Y_training_set = np.asarray(sourceMatrix[["Y"]].copy(deep=True).reset_index(drop=True))
print("Y_training_set")
print(Y_training_set)
Y_testing_set = np.asarray(sourceMatrix[["Y"]].copy(deep=True).reset_index(drop=True))
print("Y_testing_set")
print(Y_testing_set)


# Creates class node.
class TNode:
    def __init__(self, depth, X, Y):
        self.depth = depth
        self.X = X  # Explanatory vars matrix.
        self.Y = Y  # Answer vars matrix.
        # "Split" params initialize.
        self.j = None  # Coordinate to make partition.
        self.xi = None  # Partition value in the coordinate.
        # Empty child initialize.
        self.left = None  # Subsequently at "constructSubtree" function, a tree is defined.
        self.right = None  # Subsequently at "constructSubtree" function, a tree is defined.
        # Predictor node initialize.
        self.g = None

    def CalculateLoss(self):
        if (len(self.Y) == 0):
            return 0
        else:
            return np.sum(np.power(self.Y - self.Y.mean(), 2))


treeRoot = TNode(0, X_training_set, Y_training_set)
print("Tree depth: ", treeRoot.depth)
print("Tree explanatory vars: ", treeRoot.X)
print("Tree result vars:", treeRoot.Y)

# Data frame values locale observation.
print("X_training_set[:, 0]")
print(X_training_set[:, 0])  # Returns matrix first column.
print("X_training_set[:, 1]")
print(X_training_set[:, 1])  # Returns matrix second column.
ids_bis = X_training_set[:, 0] <= 6
print("id_bis: ", ids_bis)


# Data split function.
def dataSplitFunction(X, Y, j, xi):
    ids = X[:, j] <= xi  # X[:,j] is an array formed by the "j" entries of each vector from the original array.
    Xtrue = X[ids == True, :]  # Original array elements that meet "idf".
    Xfalse = X[ids == False, :]
    Ytrue = Y[ids == True]
    Yfalse = Y[ids == False]
    return Xtrue, Ytrue, Xfalse, Yfalse


# Example.
Xtrue, Ytrue, Xfalse, Yfalse = dataSplitFunction(X_training_set, Y_training_set, 1, 15)
print("Xtrue", Xtrue)
print("Xfalse", Xfalse)
print("Ytrue", Ytrue)
print("Yfalse", Yfalse)

# Shape function example.
rows, columns = X_training_set.shape
print("Number of rows: ", rows)
print("Number of columns: ", columns)


# Optimun split function, just in the initial case.
def computeOptimalSplit(node):
    X = node.X
    Y = node.Y
    best_var = 0  # Dimension where de partition will be made.
    best_xi = X[0, best_var]  # Value in each coordinate to make the feasible region division.
    best_split_val = node.CalculateLoss()
    rows, columns = X.shape
    for j in range(0, columns):
        for i in range(0, rows):
            xi = X[i, j]
            Xtrue, Ytrue, Xfalse, Yfalse = dataSplitFunction(X, Y, j, xi)
            tmpt = TNode(0, Xtrue, Ytrue)
            tmpf = TNode(0, Xfalse, Yfalse)
            loss_t = tmpt.CalculateLoss()
            loss_f = tmpf.CalculateLoss()
            curr_val = loss_t + loss_f
            if (curr_val < best_split_val):
                best_split_val = curr_val
                best_var = j
                best_xi = xi
    return best_var, best_xi  # Partition coordinate value (note that is a value from the sample).


# Example.
best_var, best_xi = computeOptimalSplit(treeRoot)
print("Optimal coordinate to make the first partition of the feasible region: ", best_var)
print("Coordinate optimal value to make the partition: ", best_xi)


# Recursive function example.
def factorialFunction(columns):
    if columns == 0 or columns == 1:
        Y = 1
    else:
        Y = columns * factorialFunction(columns - 1)
    return Y


# Example.
columns = 5
print("The computed factorial for number ", columns, " is: ", factorialFunction(columns))


# Subtree constuction.
def constructSubtree(node, max_depth):
    if (node.depth == max_depth or len(
            node.Y) == 1):  # The value "1" is arbitrary to stop the algorithm execution and show the minimal value of numbers to average.
        node.g = node.Y.mean()  # Here goes the function that returns the binary tree value.
    else:
        j, xi = computeOptimalSplit(node)
        node.j = j  # Coordinate to make the partition.
        node.xi = xi  # Coordinate value to make the partition.
        Xtrue, Ytrue, Xfalse, Yfalse = dataSplitFunction(node.X, node.Y, j, xi)

        if (len(Ytrue) > 0):
            node.left = TNode(node.depth + 1, Xtrue, Ytrue)  # Adds a right node to the exterior node.
            constructSubtree(node.left, max_depth)  # Recursive function.
        if (len(Yfalse) > 0):
            node.right = TNode(node.depth + 1, Xfalse, Yfalse)  # Adds a right node to the left node.
            constructSubtree(node.right, max_depth)  # Recursive function.
    return node


# Example.
maxDepth = 2
subTree = constructSubtree(treeRoot, maxDepth)

# Constructed binary tree display example.
print("************************Level 0************************")
print("Current level", subTree.depth)
print("Classifying coordinate", subTree.j)
print("Classifying coordinate value", subTree.xi)
print("************************Level 1 left************************")
node1_left = subTree.left
print("Current level", node1_left.depth)
print("Classifying coordinate", node1_left.j)
print("Classifying coordinate value", node1_left.xi)
print("************************Level 1 right************************")
node1_right = subTree.right
print("Current level", node1_right.depth)
print("Classifying coordinate", node1_right.j)
print("Classifying coordinate value", node1_right.xi)
print("************************Level 2 left - left************************")
node2_left_left = node1_left.left
print("Current level", node2_left_left.depth)
print("Classifying coordinate", node2_left_left.j)
print("Classifying coordinate value", node2_left_left.xi)
print("************************Level 2 left - right************************")
node2_left_right = node1_left.right
print("Current level", node2_left_right.depth)
print("Classifying coordinate", node2_left_right.j)
print("Classifying coordinate value", node2_left_right.xi)
print("************************Level 2 right - left************************")
node2_right_left = node1_right.left
print("Current level", node2_right_left.depth)
print("Classifying coordinate", node2_right_left.j)
print("Classifying coordinate value", node2_right_left.xi)
print("************************Level 2 right - right************************")
node2_right_right = node1_right.right
print("Current level", node2_right_right.depth)
print("Classifying coordinate", node2_right_right.j)
print("Classifying coordinate value", node2_right_right.xi)


def predictFunction(X, node):
    if (node.right == None and node.left != None):
        return predictFunction(X, node.left)
    if (node.right != None and node.left == None):
        return predictFunction(X, node.right)
    if (node.right == None and node.left == None):
        return node.g
    else:
        # Note that is given a feature row as parameter, not the full matrix.
        if (X[node.j] <= node.xi):
            return predictFunction(X, node.left)
        else:
            return predictFunction(X, node.right)


# Example.
y_hat = np.zeros(len(X_testing_set))
print("y_hat", y_hat)

for i in range(len(X_testing_set)):
    y_hat[i] = predictFunction(X_testing_set[i], treeRoot)

print("predicted y_hat", y_hat)

meanSquareError1 = meanSquareErrorFunction(y_hat, Y_testing_set)

print("Mean square error = ", meanSquareError1)

# Data frame definition to compare the results.
compareMatrix1 = pd.DataFrame(index=range(len(y_hat)), columns=["Real data", "Approximate data"])
compareMatrix1["Real data"] = Y_testing_set
compareMatrix1["Approximate data"] = y_hat

print("compareMatrix1:")
print(compareMatrix1)

# Particular prediction example.
forecastQuadrant1 = predictFunction([1, 3], treeRoot)
forecastQuadrant2 = predictFunction([7, 4], treeRoot)
forecastQuadrant3 = predictFunction([2, 8], treeRoot)
forecastQuadrant4 = predictFunction([9, 10], treeRoot)
print("First quadrant forecast ", forecastQuadrant1)
print("Second quadrant forecast ", forecastQuadrant2)
print("Third quadrant forecast ", forecastQuadrant3)
print("Fourth quadrant forecast ", forecastQuadrant4)

# Calculation using python defined functions.
# SkLearn method comparison.
# Source: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
from sklearn.tree import DecisionTreeRegressor

regTree = DecisionTreeRegressor(max_depth=2)
regTree.fit(X_training_set, Y_training_set)
y_hat2 = regTree.predict(X_testing_set)
meanSquareError2 = meanSquareErrorFunction(y_hat2, Y_testing_set)
print("Mean square error = ", meanSquareError2)

# Data frame definition to compare the results.
compareMatrix2 = pd.DataFrame(index=range(len(y_hat2)), columns=["Real data", "Approximate data"])
compareMatrix2["Real data"] = Y_testing_set
compareMatrix2["Approximate data"] = y_hat2

print("compareMatrix2:")
print(compareMatrix2)
