
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
#
# # Read the CSV file
# def load_dataset():
#     df = pd.read_csv("/Users/yangyue/Desktop/CS3244/Fashion-MNIST/data/fashion-mnist_train.csv")
#     df2 = pd.read_csv("/Users/yangyue/Desktop/CS3244/Fashion-MNIST/data/fashion-mnist_test.csv")
#
#     # Get the feature matrix
#     # TODO: raw data
#     X_train = df[df.columns[df.columns != 'label']].copy()
#     X_test = df2[df2.columns[df.columns != 'label']].copy()
#     # Get the label
#     y_train = df['label'].copy()
#     y_test = df2['label'].copy()
#
#     X_train = X_train.to_numpy()
#     X_test = X_test.to_numpy()
#     # Transform the data into numpy array
#     X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
#     X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))
#     # Transform the labels into lists
#     y_train = to_categorical(y_train)
#     y_test = to_categorical(y_test)
#     return X_train, y_train, X_test, y_test
#
#
# def prep_pixels(train, test):
#     # convert from integers to floats
#     train_norm = train.astype('float32')
#     test_norm = test.astype('float32')
#     # normalize to range 0-1
#     train_norm = train_norm / 255.0
#     test_norm = test_norm / 255.0
#     # return normalized images
#     return train_norm, test_norm
#
#
# def define_model():
#     model = Sequential()
#     model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
#     model.add(MaxPooling2D((2, 2)))
#     model.add(Flatten())
#     model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
#     model.add(Dense(10, activation='softmax'))
#     # compile model
#     opt = SGD(lr=0.01, momentum=0.9)
#     model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
#     return model
#
#
# def evaluate_model(X_data, Y_data, n_folds=5):
#     scores, histories = list(), list()
#     # prepare cross validation
#     kfold = KFold(n_folds, shuffle=True, random_state=1)
#     # enumerate splits
#     for train_ix, test_ix in kfold.split(X_data):
#         # define model
#         model = define_model()
#         # select rows for train and test
#         trainX, trainY, testX, testY = X_data[train_ix], Y_data[train_ix], X_data[test_ix], Y_data[test_ix]
#         # fit model
#         history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
#         # evaluate model
#         _, acc = model.evaluate(testX, testY, verbose=0)
#         print('> %.3f' % (acc * 100.0))
#         # append scores
#         scores.append(acc)
#         histories.append(history)
#     return scores, histories
#
#
# # plot diagnostic learning curves
# def summarize_diagnostics(histories):
#     for i in range(len(histories)):
#         # plot loss
#         pyplot.subplot(211)
#         pyplot.title('Cross Entropy Loss')
#         pyplot.plot(histories[i].history['loss'], color='blue', label='train')
#         pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')
#         # plot accuracy
#         pyplot.subplot(212)
#         pyplot.title('Classification Accuracy')
#         pyplot.plot(histories[i].history['accuracy'], color='blue', label='train')
#         pyplot.plot(histories[i].history['val_accuracy'], color='orange', label='test')
#     pyplot.show()
#
#
# # summarize model performance
# def summarize_performance(scores):
#     # print summary
#     print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores) * 100, std(scores) * 100, len(scores)))
#     # box and whisker plots of results
#     pyplot.boxplot(scores)
#     pyplot.show()
#
#
# def run_test():
#     x_train, y_train, x_test, y_test = load_dataset()
#     X_train, X_test = prep_pixels(x_train, x_test)
#     scores, histories = evaluate_model(X_train, y_train)
#     summarize_diagnostics(histories)
#     summarize_performance(scores)
#
# run_test()
#

