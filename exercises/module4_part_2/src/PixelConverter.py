import numpy as np


# A es una matriz (en formato array) en donde cada entrada es un vector de dimensión 3 (pixel)
def pixelToMatrix(A):
    M = []
    n = len(A)
    m = len(A[0])
    for i in range(n):
        vaux1 = []
        vaux2 = []
        vaux3 = []
        for j in range(m):
            vaux1.extend([A[i][j][0]])
            vaux2.extend([A[i][j][1]])
            vaux3.extend([A[i][j][2]])
        M.extend([vaux1])
        M.extend([vaux2])
        M.extend([vaux3])
    return M


# M es una matriz (en formato lectura csv) cada grupo de 3 renglones determina las entradas de un pixel
def matrixToPixel(M):
    MP = []
    n = len(M)
    m = len(M[0])
    i = 0
    while i <= n - 3:
        vr = []
        for j in range(m):
            vr.extend([[M[i][j], M[i + 1][j], M[i + 2][j]]])
        MP.extend([vr])
        i = i + 3
    MP = np.asarray(MP)
    return (MP)
