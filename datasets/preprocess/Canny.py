from skimage import filters, feature
import numpy as np
def canny(X_train, X_test):
  # apply canny edge detection to all train and test
  # for each image, we append the edge detected image to the original image
  def apply_canny_filter(img): # img is a 1d array
    # takes 1d img array, applies filter and returns 1d array
    original_img = img.reshape(28,28)
    cannied_img = feature.canny(original_img)
    combined_img = np.concatenate((original_img, cannied_img))
    # plt.imshow(combined_img, cmap = 'Greys') # show concatenated image
    return combined_img.flatten()
  X_train_cannied = list(map(apply_canny_filter, X_train))
  X_test_cannied = list(map(apply_canny_filter, X_test))
  return (X_train_cannied, X_test_cannied)