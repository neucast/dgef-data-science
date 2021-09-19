# Program to multiply two matrices using nested loops

# 3x3 matrix
X = [[1,1],
    [-2,4]]

# 1x3 matrix
Y = [[1],[1]]

# result is 1x3
result = [[0],[0]]

# iterate through rows of X
for i in range(len(X)):
   # iterate through columns of Y
   for j in range(len(Y[0])):
       # iterate through rows of Y
       for k in range(len(Y)):
           result[i][j] += X[i][k] * Y[k][j]

for r in result:
   print(r)

