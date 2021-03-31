from exercises.module2_part_1.src.NumberBaseConversion import integer_to_binary, binary_to_integer

inputBooleanFlag = False
binaryNumberStringValue = ""

while inputBooleanFlag == False:
    try:
        binaryNumberStringValue = input("Please input a positive non floating point binary number string: ")

        if isinstance(binaryNumberStringValue, str):
            for binaryDigitCharValue in binaryNumberStringValue:
                if binaryDigitCharValue == "0":
                    inputBooleanFlag = True
                elif binaryDigitCharValue == "1":
                    inputBooleanFlag = True
                else:
                    inputBooleanFlag = False
                    binaryNumberStringValue = ""
                    break
        else:
            inputBooleanFlag = False
            binaryNumberStringValue = ""
    except ValueError:
        inputBooleanFlag = False
        binaryNumberStringValue = ""

integerNumberValue = binary_to_integer(binaryNumberStringValue)
print(f"Base two number: {binaryNumberStringValue} is: {integerNumberValue} integer number on the base ten.")
