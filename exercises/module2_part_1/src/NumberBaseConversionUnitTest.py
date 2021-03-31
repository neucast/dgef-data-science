from exercises.module2_part_1.src.NumberBaseConversion import integer_to_binary, binary_to_integer

integerNumberValue = 25
binaryNumberStringValue = integer_to_binary(integerNumberValue)
print(f"Integer base ten number: {integerNumberValue} is: {binaryNumberStringValue} on the base two.")
if bin(integerNumberValue) == "0b" + binaryNumberStringValue:
    print("Test passed.")
else:
    print("Test failed.")

integerNumberValue = binary_to_integer(binaryNumberStringValue)
print(f"Base two number: {binaryNumberStringValue} is: {integerNumberValue} integer number on the base ten.")
if int(str(binaryNumberStringValue), 2) == integerNumberValue:
    print("Test passed.")
else:
    print("Test failed.")