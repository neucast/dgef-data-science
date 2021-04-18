from math import pi
from math import e

from exercises.module2_part_2.src.LambdaBasicMathOps import show_all_basic_math_operations_results

"""
Create a lambda function that can perform the addition, subtraction, division and multiplication 
functions indicated as parameters between two given numbers.
"""

print("Test #1:")
show_all_basic_math_operations_results(0, 0)
print("\n")

print("Test #2:")
show_all_basic_math_operations_results(0, 4)
print("\n")

print("Test #3:")
show_all_basic_math_operations_results(6, 0)
print("\n")

print("Test #4:")
show_all_basic_math_operations_results(6, 4)
print("\n")

print("Test #5:")
show_all_basic_math_operations_results(pi, e)
print("\n")
