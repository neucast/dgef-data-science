# Clusters.
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch  # https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html
from sklearn.cluster import AgglomerativeClustering

# Configure.
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.options.display.float_format = "{:,.5f}".format

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


# 2D data plot function.
def plot2D(X, Y, colors, colorMap, xLabel, yLabel, plotTitle, pngFileName):
    plotFigureXvsY, ax = plt.subplots()
    ax.set(xlabel=xLabel, ylabel=yLabel, title=plotTitle)
    ax.grid(True)
    ax.scatter(X, Y, c=colors, cmap=colorMap)
    plotFigureXvsY.tight_layout()

    outputPath = os.path.join(os.path.expanduser("~"), "development", "dgef-data-science",
                              "exercises", "module4_part_1",
                              "output", pngFileName)
    plotFigureXvsY.savefig(outputPath)

    plt.show()


# 3D data plot function.
def plot3D(xMin, xMax, yMin, yMax, zMin, zMax, sampleNumber, xData, yData, zData, xLabel, yLabel, zLabel, colors,
           colorMap):
    ax3D = plt.axes(projection="3d")

    # Data for a three-dimensional line.
    zLine = np.linspace(zMin, zMax, num=sampleNumber)
    xLine = np.linspace(xMin, xMax, num=sampleNumber)
    yLine = np.linspace(yMin, yMax, num=sampleNumber)
    ax3D.plot3D(xLine, yLine, zLine, "gray")

    # Orthogonal.
    ax3D.view_init(30, 60)

    ax3D.set_xlabel(xLabel)
    ax3D.set_ylabel(yLabel)
    ax3D.set_zlabel(zLabel)
    ax3D.scatter3D(xData, yData, zData, c=colors, cmap=colorMap);
    plt.show()


# Data plot.
# X1 vs X2.
plot2D(sourceMatrix["X1"], sourceMatrix["X2"], sourceMatrix["X3"], "hsv", "X1", "X2", "X1 vs X2",
       "X1vsX2.png")

# X1 vs X3.
plot2D(sourceMatrix["X1"], sourceMatrix["X3"], sourceMatrix["X2"], "hsv", "X1", "X3", "X1 vs X3", "X1vsX3.png")

# X2 vs X3.
plot2D(sourceMatrix["X2"], sourceMatrix["X3"], sourceMatrix["X1"], "hsv", "X2", "X3", "X2 vs X3", "X2vsX3.png")

# 3D plot.
plot3D(-7., 5., -6., 13., -12., 13., 1000, sourceMatrix["X1"], sourceMatrix["X2"], sourceMatrix["X3"], "X1", "X2", "X3",
       sourceMatrix["X3"], "hsv")

# Dendogram.
dendrogram = sch.dendrogram(sch.linkage(sourceMatrix, method="single", metric="euclidean"))
plt.title("Dendogram")
plt.xlabel("Dot")
plt.ylabel("Euclidean distances")
plt.show()

dataCluster = AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage="single")
computedCluster = dataCluster.fit_predict(sourceMatrix)
sourceMatrix["Tag"] = computedCluster
print(sourceMatrix.head())

# Computed cluster analysis 2D plot.
# X1 vs X2.
plot2D(sourceMatrix["X1"], sourceMatrix["X2"], computedCluster, "rainbow", "X1", "X2", "X1 vs X2",
       "X1vsX2ComputedCluster.png")

# X1 vs X3.
plot2D(sourceMatrix["X1"], sourceMatrix["X3"], computedCluster, "rainbow", "X1", "X3", "X1 vs X3",
       "X1vsX3ComputedCluster.png")

# X2 vs X3.
plot2D(sourceMatrix["X2"], sourceMatrix["X3"], computedCluster, "rainbow", "X2", "X3", "X2 vs X3",
       "X2vsX3ComputedCluster.png")

# Computed cluster analysis 3D plot.
plot3D(-7., 5., -6., 13., -12., 13., 1000, sourceMatrix["X1"], sourceMatrix["X2"], sourceMatrix["X3"], "X1", "X2", "X3",
       computedCluster, "rainbow")
