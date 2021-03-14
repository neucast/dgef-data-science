from NewtonRaphson import newtonRaphsonNumericMethod


def computeSquareRootByNewtonRaphsonNumericMethod(inputFloatValue) -> object:
    f = lambda x: x ** 2 - inputFloatValue
    Df = lambda x: 2 * x
    approxSquareRootFloatValue = newtonRaphsonNumericMethod(f, Df, 1, 1e-10, 10)
    return approxSquareRootFloatValue


"""
Function that approximates the square root of a number ("inputValue").

Step 1. Propose a close number "lowerLimit" (lower value close to
the square root)
and another close number "upperLimit" (upper value close to the square root).

Step 2. Define the "proposed value" as the average of these two numbers.

Step 3. If the square of the "proposed value" is close (according to "epsilon") to
"wanted value" then, return this value.

Step 4. If the "proposed value" squared is to the right of the
"searched value", then the algorithm is repeated with the value of "lowerLimit"
and "upperLimit" takes the "proposed value". On the contrary, if the "value
proposed "squared is to the left of the" sought value ", then
the variable "lowerLimit" takes the "proposed value" and the algorithm is repeated, respecting
the value of "upperLimit".

Step 5. It is recommended to define a variable that allows only a maximum number
of iterations "maxIterations".
"""


def computeSquareRootByBisectionNumericMethod(inputFloatValue, lowerLimitFloatValue, upperLimitFloatValue,
                                              epsilonFloatValue,
                                              maxIterationsIntValue) -> object:
    for iteration in range(maxIterationsIntValue):
        average = computeAverage(lowerLimitFloatValue, upperLimitFloatValue)
        if abs(average ** 2 - inputFloatValue) <= epsilonFloatValue:
            # print('Computed square root value is: ', average)
            break
        elif average ** 2 > inputFloatValue:
            upperLimitFloatValue = average
        else:
            lowerLimitFloatValue = average
    print('Found solution after', iteration, 'iterations.')
    return [average, iteration]


def computeAverage(firstFloatValue, secondFloatValue):
    average = (firstFloatValue + secondFloatValue) / 2
    return average
