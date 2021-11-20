
# import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import csv
import math
import random
from scipy import ndimage
from sklearn.manifold import TSNE


# Read the CSV file
df = pd.read_csv("/Users/yangyue/Desktop/CS3244/Fashion-MNIST/data/fashion-mnist_train.csv")
df2 = pd.read_csv("/Users/yangyue/Desktop/CS3244/Fashion-MNIST/data/fashion-mnist_test.csv")

# Get the feature matrix
# TODO: raw data
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

# train_df = pd.read_csv("/Users/yangyue/Desktop/CS3244/Fashion-MNIST/data/fashion-mnist_train.csv")
# features_df = train_df.iloc[:,1:].values
# labels_df = train_df.iloc[:,0].values
#
# X_train,X_test,y_train,y_test = train_test_split(features_df,labels_df,test_size=0.2)

# kernel
def kernel(x_train):
    def apply_sobel_filters(img):
        # takes 1d img array, converts to 2d, applies sobel filters, convert to 1d and returns
        img = np.array(np.split(img, 28))  # convert to 2d
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

        Ix = ndimage.filters.convolve(img, Kx)
        Iy = ndimage.filters.convolve(img, Ky)

        G = np.hypot(Ix, Iy)
        G = G / G.max() * 255
        theta = np.arctan2(Iy, Ix)
        G = G.flatten()

        return G

    x_train_sobeled = list(map(apply_sobel_filters, x_train))
    return x_train_sobeled

X_train_sobeled = kernel(X_train)
X_test_sobeled = kernel(X_test)

#t-SNE
def t_sne(x_train):
    X_embedded = TSNE(n_components=40, verbose=1, perplexity=40, n_iter=300).fit_transform(x_train)
    return X_embedded

X_train_embedded = t_sne(X_train)
X_test_embedded = t_sne(X_test)


training_x = [(X_train, X_test), (X_train_sobeled, X_test_sobeled), (X_train_embedded, X_test_embedded)]
naming = ["raw data", "kernel", "t-SNE"]
for n in range(3):
    accuracy = []
    for i in range(1, 12, 2):
        clf = KNeighborsClassifier(n_neighbors=i)
        clf.fit(training_x[n][0], y_train)
        y_pred = clf.predict(training_x[n][1])
        acc = accuracy_score(y_test, y_pred)
        accuracy.append((i, acc))
    print(naming[n] + ": ")
    for item in accuracy:
        print("k = " + str(item[0]) + ", accuracy: " + str(item[1]))
