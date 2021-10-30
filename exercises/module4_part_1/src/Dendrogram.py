import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage

# Configure.
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.options.display.float_format = "{:,.5f}".format

# Constant.
METRIC = "euclidean"
METHOD = "average"

# inputPath to the CSV file.
inputPath = os.path.join(os.path.expanduser("~"), "development", "dgef-data-science",
                         "exercises", "module4_part_1",
                         "data", "base-ejercicio-tarea-clusters.csv")

# Prints the absolute inputPath to the CSV file.
print("The input CSV file is: ", inputPath)

# Reads the CSV data file.
sourceMatrix = pd.read_csv(inputPath, dtype="str", encoding="ISO-8859-1")
sourceMatrix[sourceMatrix.columns] = sourceMatrix[sourceMatrix.columns].astype(float)
print(sourceMatrix.head())


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
    plotDendrogram(sourceMatrix)
