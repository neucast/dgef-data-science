"""Number base conversion."""

"""Converts an integer decimal number to its equivalent binary number.

Parameters:
----------
integerNumberValue : Integer number value
    Is the integer value to convert to binary.

Returns:
-------
binaryNumberStringValue : String binary "number" value.
    Is the computed binary string value.

Example:
--------
>>> integerNumberValue = 9
>>> integer_to_binary(integerNumberValue)
Integer base ten number: 9 is: 1001 on the base two.
"""


def integer_to_binary(integerNumberValue):
    if integerNumberValue <= 0:
        return "0"

    # Binary number string value accumulator.
    binaryNumberStringValue = ""

    # Division by two loop.
    while integerNumberValue > 0:
        # Division by two residue, must be 0 or 1.
        residue = int(integerNumberValue % 2)

        # Division by two.
        integerNumberValue = int(integerNumberValue / 2)

        # Accumulate the string value of the residue at the left side.
        binaryNumberStringValue = str(residue) + binaryNumberStringValue
    return binaryNumberStringValue


"""Converts binary string number to its equivalent integer decimal number.

Parameters:
----------
binaryNumberStringValue : String binary "number" value.
    Is the binary string value to convert to its decimal number equivalent.

Returns:
-------
integerNumberValue : Integer number value
    Is the computed integer value.


Example:
--------
>>> binaryNumberStringValue = 1001
>>> binary_to_integer(binaryNumberStringValue)
Base two number: 1001 is: 9 integer number on the base ten.
"""


def binary_to_integer(binaryNumberStringValue):
    # Startup the digit position counter.
    digitPositionIntegerValue = 0

    # Startup the computed integer value.
    integerNumberValue = 0

    # Invert the "binary" string because we must travel it from right to left.
    binaryNumberStringValue = binaryNumberStringValue[::-1]

    for digitCharValue in binaryNumberStringValue:
        # Raise base two to the power given by position
        multiplier = 2 ** digitPositionIntegerValue

        # Multiply by the selected digit.
        integerNumberValue += int(digitCharValue) * multiplier

        # Go to the next digit.
        digitPositionIntegerValue += 1
    return integerNumberValue
