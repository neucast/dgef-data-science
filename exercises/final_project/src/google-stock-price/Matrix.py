import numpy as np


# Compute prediction.
def predict(arrayA, arrayB, intercept):
    return matrixMultiplication(arrayA, arrayB) + np.asarray(intercept)


# Multiply two arrays.
def matrixMultiplication(arrayA, arrayB):
    return np.matmul(np.asarray(arrayA), np.asarray(arrayB).transpose())
