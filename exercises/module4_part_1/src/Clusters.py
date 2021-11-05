# Clusters.
import warnings

import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch  # https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html
from sklearn.cluster import AgglomerativeClustering

from ClusterPlotter import plot2D, plot3D, plot3DPlotty
from DataDelegate import getDataFrame
from FileManager import getOutputPath

# Constant.
NUM_CLUSTERS = 4
AFFINITY = "euclidean"
LINKAGE = "average"

# Note:
# "NUM_CLUSTERS" (the number of clusters) was determined and corroborated using four different procedures:
# graphic inspection, dendrogram, elbow method, and gap method. See: 2D and 3D plots, Dendrogram.py, ElbowMethod.py
# and Gap.py.

# Configure.
warnings.filterwarnings("ignore")

# Reads the CSV data file.
sourceMatrix = getDataFrame("base-ejercicio-tarea-clusters.csv")

# Data plot.
# X1 vs X2.
plot2D(sourceMatrix["X1"], sourceMatrix["X2"], sourceMatrix["X3"], "hsv", "X1", "X2", "X1 vs X2",
       getOutputPath("cluster-method-X1vsX2.png"))

# X1 vs X3.
plot2D(sourceMatrix["X1"], sourceMatrix["X3"], sourceMatrix["X2"], "hsv", "X1", "X3", "X1 vs X3",
       getOutputPath("cluster-method-X1vsX3.png"))

# X2 vs X3.
plot2D(sourceMatrix["X2"], sourceMatrix["X3"], sourceMatrix["X1"], "hsv", "X2", "X3", "X2 vs X3",
       getOutputPath("cluster-method-X2vsX3.png"))

# 3D plot.
plot3D(-7., 5., -6., 13., -12., 13., 1000, sourceMatrix["X1"], sourceMatrix["X2"], sourceMatrix["X3"], "X1", "X2", "X3",
       sourceMatrix["X3"], "hsv")

plot3DPlotty(sourceMatrix, sourceMatrix["X3"], abs(sourceMatrix["X3"]))

# Dendrogram.
dendrogram = sch.dendrogram(sch.linkage(sourceMatrix, method=LINKAGE, metric=AFFINITY))
plt.title("Dendrogram")
plt.xlabel("Dot")
plt.ylabel("Euclidean distances")
plt.show()

# Cluster method.
clusterModel = AgglomerativeClustering(n_clusters=NUM_CLUSTERS, affinity=AFFINITY, linkage=LINKAGE)
tags = clusterModel.fit_predict(sourceMatrix)
sourceMatrix["Tag"] = tags
print(sourceMatrix.head())

# Computed cluster analysis 2D plot.
# X1 vs X2.
plot2D(sourceMatrix["X1"], sourceMatrix["X2"], tags, "rainbow", "X1", "X2", "X1 vs X2",
       getOutputPath("cluster-method-X1vsX2-computed-cluster.png"))

# X1 vs X3.
plot2D(sourceMatrix["X1"], sourceMatrix["X3"], tags, "rainbow", "X1", "X3", "X1 vs X3",
       getOutputPath("cluster-method-X1vsX3-computed-cluster.png"))

# X2 vs X3.
plot2D(sourceMatrix["X2"], sourceMatrix["X3"], tags, "rainbow", "X2", "X3", "X2 vs X3",
       getOutputPath("cluster-method-X2vsX3-computed-cluster.png"))

# Computed cluster analysis 3D plot.
plot3D(-7., 5., -6., 13., -12., 13., 1000, sourceMatrix["X1"], sourceMatrix["X2"], sourceMatrix["X3"], "X1", "X2", "X3",
       tags, "rainbow")

plot3DPlotty(sourceMatrix, sourceMatrix["X3"], abs(sourceMatrix["X3"]))

plot3DPlotty(sourceMatrix, sourceMatrix["X3"], tags + 1)
