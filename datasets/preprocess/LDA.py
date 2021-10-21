from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

"""
Applies LDA to training dataset then transforms both training and test datasets
Note: normalize data before using

Arguments:
  - X_train: training dataset
  - y_train: labels for training dataset
  - X_test: test dataset
  - n_components: number of dimensions to reduce to
"""
def lda(X_train, y_train, X_test, n_components):
  lda = LinearDiscriminantAnalysis(n_components=n_components)
  X_train_lda = lda.fit_transform(X_train, y_train)
  X_test_lda = lda.transform(X_test)
  return (X_train_lda, X_test_lda)
