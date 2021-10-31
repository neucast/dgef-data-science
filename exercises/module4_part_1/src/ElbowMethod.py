import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from DataDelegate import getDataFrame

# Constant.
LOOPS = 20
MAX_ITERATIONS = 10
INITIALIZE_CLUSTERS = "k-means++"
CONVERGENCE_TOLERANCE = 0.001


def plotElbow(inertials):
    x, y = zip(*[inertia for inertia in inertials])
    plt.plot(x, y, "ro-", markersize=8, lw=2)
    plt.grid(True)
    plt.xlabel("Num. Clusters")
    plt.ylabel("Inertia")
    plt.show()


def selectClusters(dataSet, loops, max_iterations, init_cluster, tolerance):
    inertia_clusters = list()

    for i in range(1, loops + 1, 1):
        # Object KMeans
        kMeansEngine = KMeans(n_clusters=i, max_iter=max_iterations, init=init_cluster, tol=tolerance)

        # Calculate Kmeans
        kMeansEngine.fit(dataSet)

        # Obtain inertia
        inertia_clusters.append([i, kMeansEngine.inertia_])

    plotElbow(inertia_clusters)


if __name__ == "__main__":
    selectClusters(getDataFrame("base-ejercicio-tarea-clusters.csv"), LOOPS, MAX_ITERATIONS, INITIALIZE_CLUSTERS,
                   CONVERGENCE_TOLERANCE)
