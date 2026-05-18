import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import math
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import datetime

from utils import (
    plot_centroids, plot_centroids_customers,
    plot_clusters, getImage,
    plot_customers_hierarchical,
    plot_hierarchical_cluster,
    pairwise_euclidean_distances,
    plot_dendrogram
)

#cluster done with hierarchical clustering

costumer = pd.read_csv('costumer_preprocessed.csv')

#Training the model
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
clustering = model.fit(costumer)

#creating the figure
fig = plt.figure(figsize=(10, 7))
plt.title("Hierarchical Clustering Dendrogram")

#calling the dendrogram plot function
plot_dendrogram(clustering, truncate_mode="level", p=3)

plt.show()

#based on what we saw in the dendrogram plot we choose the number of clusters (4)
final_model = AgglomerativeClustering(n_clusters=4)
clusters = final_model.fit_predict(costumer)

#adding to the original DataFrame too see each group profile
costumer['cluster_id'] = clusters
print(costumer.groupby('cluster_id').mean())

