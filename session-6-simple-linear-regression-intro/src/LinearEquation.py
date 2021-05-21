# Exercise. Calculate the equation of a line that passes through two points and graph it.
# Step 1. Save the coordinates of a point in a vector or one-dimensional array.
# Step 2. Define a function that receives as input two vectors that represent the coordinates of two points, and as a result it provides the slope of the line that passes through those two points slope = (y2-y1) / (x2- x1).
# Step 3. Define a function that receives as input parameters two vectors y: i) using the previous function calculate the slope between the points, ii) from the expression y = slope (x - x1) + y1 allows to evaluate , given the "x" coordinate, the value of the "y" coordinate on the line.
# Step 4. Graph the line if x belongs to the interval [-5,5].

import numpy as np
import matplotlib.pyplot as plt

a, b = [1, 2], [2, 6]


# Funtion to compute the slope.
def computeSlope(a, b):
    numerator = b[1] - a[1]
    denominator = b[0] - a[0]
    slope = numerator / denominator
    return slope


# Test 1. Compute slope.
slope = computeSlope(a, b)
print("The computed slope value is: ", slope)


def linearEquationFunction(a, b, x):
    slope = computeSlope(a, b)
    y = slope * (x - a[0]) + a[1]
    return y


# Test 2. Linear equiation.
x = 1
y = linearEquationFunction(a, b, x)
print("If x = ", x, ", then y = ", y)

# Plot the line.
# Reference matplotlib (https://matplotlib.org/stable/tutorials/introductory/pyplot.html)
fig = plt.figure(figsize=(10, 5))

vx = np.linspace(-5, 5, 100).tolist()
vy = []

for x in vx:
    y = linearEquationFunction(a, b, x)
    vy.extend([y])

ax = fig.add_subplot(1, 1, 1)

plt.plot(vx, vy, color="blue", markersize=1.0, label="Linear equation")
plt.legend(loc="best")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.show()
