import matplotlib.pyplot as plt
import skimage
from skimage import io
from skimage.metrics import structural_similarity as ssim

from FileManager import getOutputPath, getInputPath
from MeanSquaredError import mse


# To install opencv package with conda run one of the following:
# conda install -c conda-forge opencv
# conda install -c conda-forge/label/gcc7 opencv
# conda install -c conda-forge/label/broken opencv
# conda install -c conda-forge/label/cf201901 opencv
# conda install -c conda-forge/label/cf202003 opencv

def compareImages(imageA, imageB, title):
    # Compute the mean squared error and structural similarity
    # index for the images.
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB, multichannel=True)
    # Setup the figure.
    fig = plt.figure(title)
    plt.suptitle(title + " - " + "MSE: %.2f, SSIM: %.2f" % (m, s))
    # Show first image.
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap=plt.cm.gray)
    plt.axis("off")
    # Show the second image.
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap=plt.cm.gray)
    plt.axis("off")

    # Show the images.
    plt.show()


originalImage = io.imread(getInputPath("lena-color.jpg"))
originalImage = skimage.img_as_float(originalImage)

pcaCompressMethod1Image = io.imread(getOutputPath("method-1-pca-compressed-lena.jpg"))
pcaCompressMethod1Image = skimage.img_as_float(pcaCompressMethod1Image)

pcaCompressMethod2Image = io.imread(getOutputPath("method-2-pca-compressed-lena.jpg"))
pcaCompressMethod2Image = skimage.img_as_float(pcaCompressMethod2Image)

# Initialize the figure.
fig = plt.figure("Images")
images = ("Original", originalImage), ("PCA method 1", pcaCompressMethod1Image), (
    "PCA method 2", pcaCompressMethod2Image)

# Loop over the images.
for (i, (name, image)) in enumerate(images):
    # Show the image.
    ax = fig.add_subplot(1, 3, i + 1)
    ax.set_title(name)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.axis("off")

# Show the figure.
plt.show()

# Compare the images.
compareImages(originalImage, originalImage, "Original vs. Original")
compareImages(originalImage, pcaCompressMethod1Image, "Original vs. PCA method 1")
compareImages(originalImage, pcaCompressMethod2Image, "Original vs. PCA method 2")
compareImages(pcaCompressMethod1Image, pcaCompressMethod2Image, "PCA method 1 vs. PCA method 2")
