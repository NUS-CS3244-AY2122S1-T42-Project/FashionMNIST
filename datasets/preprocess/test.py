import numpy as np
import pandas as pd
from SobelKernel import sobel

# Read the CSV file
df = pd.read_csv('../raw/fashion-mnist_train.csv')
df2 = pd.read_csv('../raw/fashion-mnist_test.csv')

# Get the feature matrix
X_train = df[df.columns[df.columns != 'label']].copy()
X_test = df2[df2.columns[df.columns != 'label']].copy()

# Transform the data into numpy array
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

trained, tested = sobel(X_train, X_test)

print(len(trained[0]))
print(len(tested[0]))
print(len(X_train[0]))
print(X_train[0])
print(len(trained[0][:784]))
print(np.array_equal(X_train[0], trained[0][:784]))