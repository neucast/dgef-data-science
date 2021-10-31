import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage

from DataDelegate import getDataFrame

# Constant.
METRIC = "euclidean"
METHOD = "average"


def plotGap(dataSet):
    # Calculate distances between points or groups of points
    Z = linkage(dataSet, metric=METRIC, method=METHOD)

    # Obtain the last 10 distances between points
    last = Z[-10:, 2]
    num_clustres = np.arange(1, len(last) + 1)

    # Calculate Gap
    gap = np.diff(last, n=2)  # second derivative
    plt.plot(num_clustres[:-2] + 1, gap[::-1], "ro-", markersize=8, lw=2)
    plt.show()


if __name__ == "__main__":
    plotGap(getDataFrame("base-ejercicio-tarea-clusters.csv"))
