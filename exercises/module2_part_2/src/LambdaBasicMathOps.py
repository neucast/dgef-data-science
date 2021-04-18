from math import nan
import numpy as np

"""
Lambda function that can perform the addition, subtraction, division and multiplication 
functions indicated as parameters between two given numbers.
"""


def execute_all_basic_math_operations(value1, value2):
    sum = lambda x, y: x + y
    subtraction = lambda x, y: x - y
    multiplication = lambda x, y: x * y
    division = lambda x, y: x / y if y != 0 else nan
    execute_operation = lambda f, value1, value2: f(value1, value2)

    sumResult = execute_operation(sum, value1, value2)
    subtractionResult = execute_operation(subtraction, value1, value2)
    multiplicationResult = execute_operation(multiplication, value1, value2)
    divisionResult = execute_operation(division, value1, value2)
    return np.array([sumResult, subtractionResult, multiplicationResult, divisionResult])


def get_result(operation, resultSet):
    result = lambda n, resultSet: resultSet[n]

    if operation == "+":
        return result(0, resultSet)
    elif operation == "-":
        return result(1, resultSet)
    elif operation == "*":
        return result(2, resultSet)
    else:
        return result(3, resultSet)


def show_result(value1, value2, operation, resultSet):
    print(value1, operation, value2, "=", get_result(operation, resultSet))


def show_sum_result(value1, value2, resultSet):
    show_result(value1, value2, "+", resultSet)


def show_subtraction_result(value1, value2, resultSet):
    show_result(value1, value2, "-", resultSet)


def show_multiplication_result(value1, value2, resultSet):
    show_result(value1, value2, "*", resultSet)


def show_division_result(value1, value2, resultSet):
    show_result(value1, value2, "/", resultSet)


def show_all_basic_math_operations_results(value1, value2):
    # Compute results.
    resultSet = execute_all_basic_math_operations(value1, value2)

    # Show results.
    show_sum_result(value1, value2, resultSet)  # sum result.
    show_subtraction_result(value1, value2, resultSet)  # subtraction result.
    show_multiplication_result(value1, value2, resultSet)  # multiplication result.
    show_division_result(value1, value2, resultSet)  # division result.
