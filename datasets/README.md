Data sets are too large to be uploaded, please download them from [Fashion MNIST | Kaggle](https://www.kaggle.com/zalando-research/fashionmnist) instead

# Preprocess

Python snippets to preprocess the data

## Canny

Script that applies edge detection to the original training and test dataset images using the skimage Canny filter. Generates the processed training and test dataset images as `X_train_cannied` and `X_test_cannied` respectively.

Contains the function `canny(X_train, X_test)` that applies the Canny edge detection kernel.

Arguments:

- `X_train` - Training dataset
- `X_test` - Test dataset

Returns `(X_train_cannied, X_test_cannied)` where:

- `X_train_cannied` is the transformed X_train
- `X_test_cannied` is the transformed X_test

Each transformed image consists of the edge detected image appended to the original image.

Example usage:

```
from datasets.preprocess.Canny import canny

X_train_cannied, X_test_cannied = canny(X_train, X_test)
```

## SobelKernel

Script that applies edge detection to the original training and test dataset images using the Sobel operator. Generates the processed training and test dataset images as `X_train_sobeled` and `X_test_sobeled` respectively.

Contains the function `sobel(X_train, X_test)` that converts the image into a scipy image and applies the Sobel edge detection kernel.

Arguments:

- `X_train` - Training dataset
- `X_test` - Test dataset

Returns `(X_train_sobeled, X_test_sobeled)` where:

- `X_train_sobeled` is the transformed X_train
- `X_test_sobeled` is the transformed X_test

Each transformed image consists of the edge detected image appended to the original image.

Example usage:

```
from datasets.preprocess.SobelKernel import sobel

X_train_sobeled, X_test_sobeled = sobel(X_train, X_test)
```

## LDA

Contains the function `lda(X_train, y_train, X_test, n_components)` that applies sklearn's LDA to `X_train` with respect to `y_train` then transforms both `X_train` and `X_test`.

Arguments:

- `X_train` - Training dataset
- `y_train` - Labels for training dataset
- `X_test` - Test dataset
- `n_components` - Number of linear discriminants to use i.e. number of dimensions of result

Returns `(X_train_lda, X_test_lda)` where:

- `X_train_lda` is the transformed X_train
- `X_test_lda` is the transformed X_test

Example usage:

```
from datasets.preprocess.LDA import lda

X_train_lda, X_test_lda = lda(X_train, y_train, X_test, 9)
```

_Note: normalize data before using_

## PCA

Contains the function `pca(X_train, X_test, n_components)` that applies sklearn's PCA to `X_train` then transforms both `X_train` and `X_test`.

Arguments:

- `X_train` - Training dataset
- `X_test` - Test dataset
- `n_components` - Number of principal components to use i.e. number of dimensions of result

Returns `(X_train_pca, X_test_pca)` where:

- `X_train_pca` is the transformed X_train
- `X_test_pca` is the transformed X_test

Example usage:

```
from datasets.preprocess.PCA import pca

X_train_pca, X_test_pca = pca(X_train, X_test, 80)
```

_Note: normalize data before using_

# Visualise

Python snippets to visualise the data

## tSNE

Contains the function `tsne(dataset, labels)` that applies sklearn's t-SNE using 2 dimensions to the given `dataset` and `labels` then plots the results.

Arguments:

- `dataset` - Input dataset
- `labels` - Labels for input dataset

Returns void - plots a 2D t-SNE graph

Example usage:

```
from datasets.visualise.tSNE import tsne

tsne(X_train, y_train)
```
