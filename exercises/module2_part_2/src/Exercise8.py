from exercises.module2_part_2.src.Gradient import compute_P1

"""
Step 1. Define a Python function that corresponds to the analytical expression f (x, y) = x ^ 2 - y ^ 2 + 1.
Step 2. Define a function in Python that corresponds to the gradient of the function f (x, y). 
Where the gradient is defined as Gradient (f) (x, y) = (fx (x, y), fy (x, y)), with fx the partial derivative 
of the function f with respect to "x" and fy the derivative partial of the function f with respect to "y".
Step 3. Define a third function in Python that, given a point (x0, y0) in R ^ 2, evaluates the gradient 
defined in Step 2, and returns the value of the function f (defined in Step 1) evaluated at the point 
(x1, y1) = (x0, y0) - 0.5Gradient (f) (x, y).
Step 4. If in Step 3, (x0, y0) = (1,1), which order relation is correct, 
f (x1, y1)> f (x0, y0) of (x1, y1) <f (x0, y0)? 
"""

# Define the functions.
f = lambda x, y: x ** 2 - y ** 2 + 1
Dfx = lambda x: 2 * x
Dfy = lambda y: -2 * y

# P0 = (x0, y0).
x0 = 1
y0 = 1
print("x0 = ", x0)
print("y0 = ", y0)

# Compute the value of P1 = (x1, y1).
P1 = compute_P1(Dfx, Dfy, x0, y0)
x1 = P1[0]
y1 = P1[1]
print("x1 = ", x1)
print("y1 = ", y1)

# Evaluate f(x0,y0).
fP0 = f(x0, y0)
print("f( x0 , y0 ) = f(", x0, ",", y0, ") =", fP0)

# Evaluate f(x1,y1).
fP1 = f(x1, y1)
print("f( x1 , y1 ) = f(", x1, ",", y1, ") =", fP1)

# Compare f(x1,y1) with f(x0,y0).
print("The answer is:")
if fP1 > fP0:
    print(fP1, ">", fP0)
    print("f(x1,y1) > f(x0,y0)")
elif fP1 == fP0:
    print(fP1, "=", fP0)
    print("f(x1,y1) = f(x0,y0)")
else:
    print(fP1, "<", fP0)
    print("f(x1,y1) < f(x0,y0)")
