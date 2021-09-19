import decimal
import struct
from decimal import Decimal

import numpy as np


def createBinaryRepresentation(number):
    binaryRepresentation = struct.pack(">d", number)
    binaryRepresentation = "".join(f"{b:08b}" for b in binaryRepresentation)
    binaryRepresentation = binaryRepresentation[0] + " " + binaryRepresentation[1:12] + " " + binaryRepresentation[12:]
    print(f"{binaryRepresentation}", "is the binary representation of number:", number)


a = 1
b = 2
c = a + b
print("Test 1: ", c)

alpha = 0.08
beta = 0.04
gamma = alpha + beta
print("Test 2:", gamma)

x = 0.1
y = 0.2
z = x + y
correctResult = 0.3
difference = 0
print("x =", x)
print("y =", y)
print("Correct result =", correctResult)
print("Test 3: z = x + y = ", z)
print("Test 4 ¿Is z = correctResult?:", z == correctResult)

if z > correctResult:
    difference = z - correctResult
    print("z is greater than correctResult by:", difference, "units.")
else:
    difference = correctResult - z
    print("z is less than correctResult by:", difference, "units.")

# IEEE-754.
print("float x=", f"{x:.60f}")
print("float y=", f"{y:.60f}")
print("float z=", f"{z:.60f}")
print("float correctResult=", f"{correctResult:.60f}")
createBinaryRepresentation(x)
createBinaryRepresentation(y)
createBinaryRepresentation(z)
createBinaryRepresentation(correctResult)

# Accurate representation
xDecimal = Decimal('0.1')
yDecimal = Decimal('0.2')
zDecimal = xDecimal + yDecimal
correctResultDecimal = Decimal('0.3')
print("xDecimal = ", xDecimal)
print("yDecimal = ", yDecimal)
print("zDecimal = ", zDecimal)
print("correctResultDecimal = ", correctResultDecimal)
print("Test 5 ¿Is zDecimal = correctResultDecimal?:", zDecimal == correctResultDecimal)

omegaRounded = round(2.675, 2)
omegaDecimalRounded = round(Decimal('2.675'), 2)
print('omegaRounded =', omegaRounded)
print('omegaDecimalRounded =', omegaDecimalRounded)

aNonDecimal = 0.1
bNonDecimal = 0.2
print('aNonDecimal =', aNonDecimal)
print(type(aNonDecimal))
print('bNonDecimal =', bNonDecimal)
print(type(bNonDecimal))
cNonDecimal = aNonDecimal + bNonDecimal
print('cNonDecimal =', cNonDecimal)
print(type(cNonDecimal))
nonDecimalArray = np.array([aNonDecimal, bNonDecimal, aNonDecimal + bNonDecimal])
print('nonDecimalArray =', nonDecimalArray)
print('nonDecimalArray[2] = ', nonDecimalArray[2])
print(type(nonDecimalArray[2]))

aDecimal = decimal.Decimal('0.1')
bDecimal = decimal.Decimal('0.2')
print('aDecimal =', aDecimal)
print(type(aDecimal))
print('bDecimal =', bDecimal)
print(type(bDecimal))
decimalArray = np.array([aDecimal, bDecimal, aDecimal + bDecimal], dtype=np.dtype(decimal.Decimal))
print('decimalArray =', decimalArray)
print('decimalArray[2] = ', decimalArray[2])
print(type(decimalArray[2]))
