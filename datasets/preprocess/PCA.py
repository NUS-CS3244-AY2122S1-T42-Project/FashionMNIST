from sklearn.decomposition import PCA

"""
Applies PCA to training dataset then transforms both training and test datasets
Note: normalize data before using

Arguments:
  - X_train: training dataset
  - X_test: test dataset
  - n_components: number of dimensions to reduce to
"""
def pca(X_train, X_test, n_components):
  # Set number of components to use
  pca = PCA(n_components=n_components)
  # Fit the model with training data and then transform both the training and test data
  X_train_pca = pca.fit_transform(X_train)
  X_test_pca = pca.transform(X_test)

  return (X_train_pca, X_test_pca)
