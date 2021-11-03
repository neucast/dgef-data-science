import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA

from EigenValues import getEigenValuesVector
from FileManager import getInputPath, getOutputPath
from PixelConverter import pixelToMatrix, matrixToPixel

# Constant.
NUM_COMPONENTS = 72

# Read original image.
originalImage = Image.open(getInputPath("lena-color.jpg"))

# Show image.
plt.imshow(originalImage)
plt.title("Original")
plt.show()

# Get original image array.
originalImageArray = np.asarray(originalImage)

# Original image shape.
print("Original image shape:", originalImageArray.shape)

# Method 1.
# Splitting color image into its three channels.
redChannelImage, greenChannelImage, blueChannelImage = originalImage.split()

# Red channel image shape.
redChannelImageArray = np.asarray(redChannelImage)
print("Red image shape:", redChannelImageArray.shape)
redChannelImageTmpDF = pd.DataFrame(data=redChannelImageArray)
# print(redChannelImageTmpDF)
# Divide all the data at the channel by 255 so that the data is scaled between 0 and 1.
redChannelImage255 = redChannelImageArray / 255
# print(redChannelImage255)

# Green channel image shape.
greenChannelImageArray = np.asarray(greenChannelImage)
print("Green image shape:", greenChannelImageArray.shape)
greenChannelImageTmpDF = pd.DataFrame(data=greenChannelImageArray)
# print(greenChannelImageTmpDF)
# Divide all the data at the channel by 255 so that the data is scaled between 0 and 1.
greenChannelImage255 = greenChannelImageArray / 255
# print(greenChannelImage255)

# Blue channel image shape.
blueChannelImageArray = np.asarray(blueChannelImage)
print("Blue image shape:", blueChannelImageArray.shape)
blueChannelImageTmpDF = pd.DataFrame(data=blueChannelImageArray)
# print(blueChannelImageTmpDF)
# Divide all the data at the channel by 255 so that the data is scaled between 0 and 1.
blueChannelImage255 = blueChannelImageArray / 255
# print(blueChannelImage255)

# Plotting each channel image.
fig = plt.figure(figsize=(15, 7.2))
fig.add_subplot(131)
plt.title("Red Channel")
plt.imshow(redChannelImageArray)
fig.add_subplot(132)
plt.title("Green Channel")
plt.imshow(greenChannelImageArray)
fig.add_subplot(133)
plt.title("Blue Channel")
plt.imshow(blueChannelImageArray)

plt.show()

# Each channel has 500 dimensions, and we will now consider only 100 dimensions for PCA
# and fit and transform the data and check how much variance is explained after
# reducing data to 50 dimensions.
# PCA red channel.
redChannelPCAEngine = PCA(n_components=NUM_COMPONENTS)
redChannelPCAModel = redChannelPCAEngine.fit(redChannelImageArray)
transformedPCARedChannel = redChannelPCAEngine.transform(redChannelImageArray)
print("Transformed PCA red channel shape:", transformedPCARedChannel.shape)

# PCA green channel.
greenChannelPCAEngine = PCA(n_components=NUM_COMPONENTS)
greenChannelPCAModel = greenChannelPCAEngine.fit(greenChannelImageArray)
transformedPCAGreenChannel = greenChannelPCAEngine.transform(greenChannelImageArray)
print("Transformed PCA green channel shape:", transformedPCAGreenChannel.shape)

# PCA blue channel.
blueChannelPCAEngine = PCA(n_components=NUM_COMPONENTS)
blueChannelPCAModel = blueChannelPCAEngine.fit(blueChannelImageArray)
transformedPCABlueChannel = blueChannelPCAEngine.transform(blueChannelImageArray)
print("Transformed PCA blue channel shape:", transformedPCABlueChannel.shape)

# Only using 100 components we can keep around 99% of the variance in the data.
print(f"Red channel explained variance ratio sum: {sum(redChannelPCAModel.explained_variance_ratio_)}")
print(f"Green channel explained variance ratio sum: {sum(greenChannelPCAModel.explained_variance_ratio_)}")
print(f"Blue channel  explained variance ratio sum: {sum(blueChannelPCAModel.explained_variance_ratio_)}")

# Plot bar charts to check the explained variance ratio by each Eigenvalues separately
# for each of the 3 channels.
fig = plt.figure(figsize=(15, 7.2))
fig.add_subplot(131)
plt.title("Red channel")
plt.ylabel('Variation explained')
plt.xlabel('Eigen Value')
plt.bar(list(range(1, NUM_COMPONENTS + 1)), redChannelPCAModel.explained_variance_ratio_)

fig.add_subplot(132)
plt.title("Green channel")
plt.ylabel('Variation explained')
plt.xlabel('Eigen Value')
plt.bar(list(range(1, NUM_COMPONENTS + 1)), greenChannelPCAModel.explained_variance_ratio_)

fig.add_subplot(133)
plt.title("Blue channel")
plt.ylabel('Variation explained')
plt.xlabel('Eigen Value')
plt.bar(list(range(1, NUM_COMPONENTS + 1)), blueChannelPCAModel.explained_variance_ratio_)

plt.show()

# Reconstruct the reduced image and visualize it.
# Reversing transform the data.
reconstructedRedChannelArray = np.asarray(redChannelPCAEngine.inverse_transform(transformedPCARedChannel))
print("Reconstructed red channel array shape: ", reconstructedRedChannelArray.shape)
# print(reconstructedRedChannelArray)

reconstructedGreenChannelArray = np.asarray(greenChannelPCAEngine.inverse_transform(transformedPCAGreenChannel))
print("Reconstructed green channel array shape: ", reconstructedGreenChannelArray.shape)
# print(reconstructedGreenChannelArray)

reconstructedBlueChannelArray = np.asarray(blueChannelPCAEngine.inverse_transform(transformedPCABlueChannel))
print("Reconstructed blue channel array shape: ", reconstructedBlueChannelArray.shape)
# print(reconstructedBlueChannelArray)

# Merging the data of all the 3 channels into one.
reducedImageArray = np.array(originalImage)
reducedImageArray[:, :, 0] = reconstructedRedChannelArray
reducedImageArray[:, :, 1] = reconstructedGreenChannelArray
reducedImageArray[:, :, 2] = reconstructedBlueChannelArray

reducedImage = Image.fromarray(reducedImageArray)
print("Method 1 - compressed image shape: ", reducedImageArray.shape)
reducedImage.save(getOutputPath("method-1-pca-compressed-lena.jpg"))
# reducedImage.show()

# Show reduced image.
plt.imshow(reducedImage)
plt.title("Method 1 - compressed image")
plt.show()

# Display both images in order to compare them.
fig = plt.figure(figsize=(10, 7.2))
fig.add_subplot(121)
plt.title("Original Image")
plt.imshow(originalImage)
fig.add_subplot(122)
plt.title("Method 1 - compressed Image")
plt.imshow(reducedImage)
plt.show()

# Method 2.
originalImageMatrix = np.asarray(pixelToMatrix(originalImageArray))
print("Original image matrix shape:", originalImageMatrix.shape)
# Variance - covariance matrix compute.
originalImageTransposeMatrix = np.matrix.transpose(originalImageMatrix)
print("Original image transpose matrix shape:", originalImageTransposeMatrix.shape)
varianceCovarianceMatrix = np.cov(originalImageTransposeMatrix)
print("Variance - covariance matrix shape:", varianceCovarianceMatrix.shape)
# eigenValuesVector = getEigenValuesVector(varianceCovarianceMatrix, 1.00)
eigenValuesVector = getEigenValuesVector(varianceCovarianceMatrix, 0.99)
eigenValuesMatrix = eigenValuesVector[3]
print("Eigenvalues matrix shape: ", eigenValuesMatrix.shape)
# print("Eigenvalues matrix: ", eigenValuesMatrix)
eigenValuesTransposeMatrix = np.matrix.transpose(eigenValuesMatrix)
compressedImageMatrix = np.dot(eigenValuesTransposeMatrix, originalImageTransposeMatrix)
compressedImageTransposeMatrix = np.matrix.transpose(compressedImageMatrix)
eigenValuesVector = np.dot(compressedImageTransposeMatrix, eigenValuesTransposeMatrix)
compressedImage = matrixToPixel(eigenValuesVector)
compressedImage = Image.fromarray(compressedImage.astype('uint8'))
compressedImage.save(getOutputPath("method-2-pca-compressed-lena.jpg"))

# Show compressed image.
plt.imshow(compressedImage)
plt.title("Method 2 - compressed image")
plt.show()

# Display both images in order to compare them.
fig = plt.figure(figsize=(10, 7.2))
fig.add_subplot(121)
plt.title("Original Image")
plt.imshow(originalImage)
fig.add_subplot(122)
plt.title("Method 2 - compressed Image")
plt.imshow(compressedImage)
plt.show()
