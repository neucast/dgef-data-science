import warnings

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

from DataDelegate import getDataFrame

# Configure.
warnings.filterwarnings("ignore")

# Constant.
METRIC = "euclidean"
METHOD = "average"


# Dendrogram.
def plotDendrogram(dataSet):
    # Calculate distances between points or groups of points.
    Z = linkage(dataSet, metric=METRIC, method=METHOD)

    plt.title("Dendrogram")
    plt.xlabel("Points")
    plt.ylabel("Euclidean Distance")

    # Generate Dendrogram
    dendrogram(
        Z,
        truncate_mode="lastp",
        p=12,
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True
    )

    plt.show()


if __name__ == "__main__":
    plotDendrogram(getDataFrame("base-ejercicio-tarea-clusters.csv"))
