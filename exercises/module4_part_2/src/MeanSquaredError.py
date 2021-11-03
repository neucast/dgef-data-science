import numpy as np


def mse(imageA, imageB):
    # The 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: The two images must have the same dimension.
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # Return the MSE, the lower the error, the more "similar"
    # the two images are.
    return err
