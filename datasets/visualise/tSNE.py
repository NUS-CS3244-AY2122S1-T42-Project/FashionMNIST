import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE

"""
Applies t-SNE using 2 dimensions to the given dataset then plots the results

dataset: the dataset to visualise via t-SNE
labels: the corresponding labels of the dataset
"""
def tsne(dataset, labels):
  tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
  tsne_result = tsne.fit_transform(dataset)

  tsne_df = pd.DataFrame()
  tsne_df['t-SNE First Dimension'] = tsne_result[:,0]
  tsne_df['t-SNE Second Dimension'] = tsne_result[:,1]

  label_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
  tsne_df['label'] = list(map(lambda label_num: label_names[label_num], labels))

  plt.figure(figsize=(16,10))
  sns.scatterplot(
      x="t-SNE First Dimension", y="t-SNE Second Dimension",
      hue="label",
      hue_order = label_names,
      palette=sns.color_palette("hls", 10),
      data=tsne_df,
      legend="full",
      alpha=0.3
  )
  