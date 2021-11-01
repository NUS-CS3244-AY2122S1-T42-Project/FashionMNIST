# Import the standard tools for pythonic data analysis
import numpy as np
from scipy import ndimage

def sobel(X_train, X_test):

    def apply_sobel_filters(img):
        # takes 1d img array, converts to 2d, applies sobel filters, convert to 1d and returns
        img = np.array(np.split(img, 28)) # convert to 2d
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
        
        Ix = ndimage.filters.convolve(img, Kx)
        Iy = ndimage.filters.convolve(img, Ky)
        
        G = np.hypot(Ix, Iy)
        G = G / G.max() * 255
        theta = np.arctan2(Iy, Ix)
        G = G.flatten()
        return G

    # apply sobel filters to all train and test
    # note: for each image, we append the raw image data to the edge detection data
    X_train_sobeled = list(map(lambda img: img + apply_sobel_filters(img), X_train))
    X_test_sobeled = list(map(lambda img: img + apply_sobel_filters(img), X_test))

    return (X_train_sobeled, X_test_sobeled)

