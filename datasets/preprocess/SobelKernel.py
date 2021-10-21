# Import the standard tools for pythonic data analysis
import csv
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage

# Read the CSV file
df = pd.read_csv('fashion-mnist_train.csv')
df2 = pd.read_csv('fashion-mnist_test.csv')

# Get the feature matrix
X_train = df[df.columns[df.columns != 'label']].copy()
X_test = df2[df2.columns[df.columns != 'label']].copy()
# Get the label
y_train = df['label'].copy()
y_test = df2['label'].copy()


# Transform the data into numpy array
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
# Transform the labels into lists
y_train = y_train.to_list()
y_test = y_test.to_list()

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
X_train_sobeled = list(map(apply_sobel_filters, X_train))
X_test_sobeled = list(map(apply_sobel_filters, X_test))