import matplotlib.pyplot as plt
import numpy as np


# 2D data plot function.
def plot2D(X, Y, colors, colorMap, xLabel, yLabel, plotTitle, outputPath):
    plotFigureXvsY, ax = plt.subplots()
    ax.set(xlabel=xLabel, ylabel=yLabel, title=plotTitle)
    ax.grid(True)
    ax.scatter(X, Y, c=colors, cmap=colorMap)
    plotFigureXvsY.tight_layout()

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
