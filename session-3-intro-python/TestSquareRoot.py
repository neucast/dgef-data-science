from SquareRoot import compute_square_root_by_newton_raphson_numeric_method, \
    compute_square_root_by_bisection_numeric_method

inputFloatValue = 9
lowerLimit = 0.5
upperLimit = 7.2
epsilon = 1e-10
maxIterations = 100

print("Let's compute the square root value of: ", inputFloatValue)

# Newton-Raphson numeric method.
approxSquareRootFloatValue = compute_square_root_by_newton_raphson_numeric_method(inputFloatValue)
print('Computed square root value using Newton-Raphson numeric method: ', approxSquareRootFloatValue)

# Bisection numeric method.
[approxSquareRootFloatValue, maxIterations] = compute_square_root_by_bisection_numeric_method(inputFloatValue,
                                                                                              lowerLimit,
                                                                                              upperLimit, epsilon,
                                                                                              maxIterations)
print('Computed square root value using Bisection numeric method: ', approxSquareRootFloatValue)
# vector = computeSquareRootByBisectionNumericMethod(inputFloatValue, lowerLimit, upperLimit, epsilon, maxIterations)
# print(vector)
# print(vector[0] ** 2)
