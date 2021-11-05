import numpy as np


def getEigenValuesVector(Aaux, p):  # Aaux is a covariance variance matrix.
    A = np.asarray(Aaux)
    val_prop, vec_prop = np.linalg.eig(A)
    P = np.matrix.transpose(vec_prop)  # In the matrix P the eigenvectors are found in the rows.
    d = {}  # dictionary whose key is the eigen value and the eigenvector as its object.
    den = 0,
    for i in range(len(val_prop)):
        d[val_prop[i]] = P[i]
        den = den + val_prop[i]
    vkaux = list(d.keys())
    vkaux.sort()
    vk = []
    for i in range(len(vkaux)):
        vk.extend([vkaux[len(vkaux) - 1 - i]])
    vval = []
    vvec = []
    acum = vk[0] / den
    i = 0
    while (acum <= float(p)) and (i < len(vk)):
        vval.extend([vk[i]])
        vvec.extend([d[vk[i]]])
        acum = acum + vk[min([i + 1, len(vk) - 1])] / den
        i = i + 1
    vvec = np.asarray(vvec)
    vvec = np.matrix.transpose(vvec)  # vvec has the eigenvectors as columns.
    return [val_prop, vec_prop, vval, vvec]
