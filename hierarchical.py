import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram

from customer_utils import load_data, plot_cluster_sizes, plot_cluster_profile


def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    
    for i in range(len(model.children_)):
        left, right = model.children_[i]
        counts[i] = sum(
            1 if child < n_samples else counts[child - n_samples]
            for child in (left, right)
        )

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    dendrogram(linkage_matrix, **kwargs)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xticks([])
    plt.xlabel("Customers")
    plt.show()

def fit_hierarchical(data):

    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    model.fit(data)
    
    return model


def run_hierarchical(data, n_clusters=4):

    model = AgglomerativeClustering( n_clusters=n_clusters)
    labels = model.fit_predict(data)

    return model, labels



if __name__ == "__main__":
    # Load data
    costumer_preprocessed, costumer_featured = load_data()

    # Dendrogram
    exploratory = fit_hierarchical(costumer_preprocessed)
    plot_dendrogram(exploratory, truncate_mode="level", p=4)

    # Clustering
    model, labels = run_hierarchical(costumer_preprocessed, n_clusters=5)

    # Analyze clusters
    plot_cluster_profile(costumer_preprocessed, costumer_featured, labels)

    # Cluster sizes
    plot_cluster_sizes(labels, title=f"Cluster sizes — Hierarchical")