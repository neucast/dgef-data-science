from exercises.module2_part_1.src.NumberBaseConversion import integer_to_binary

inputBooleanFlag = False
integerNumberValue = 0

while inputBooleanFlag == False:
    try:
        integerNumberValue = int(input("Please input a positive integer decimal number: "))
        if isinstance(integerNumberValue, int):
            if integerNumberValue >= 0:
                inputBooleanFlag = True
            else:
                inputBooleanFlag = False
                integerNumberValue = 0
        else:
            inputBooleanFlag = False
            integerNumberValue = 0
    except ValueError:
        inputBooleanFlag = False
        integerNumberValue = 0

binaryNumberStringValue = integer_to_binary(integerNumberValue)
print(f"Integer base ten number: {integerNumberValue} is: {binaryNumberStringValue} on the base two.")
