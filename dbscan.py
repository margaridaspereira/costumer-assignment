from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

from customer_utils import load_data, plot_cluster_sizes, plot_cluster_profile, fit_umap, plot_umap_clusters


def eps_exploration(data, eps_values=[0.3, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5], min_samples=5):
    for eps in eps_values:
        model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = model.fit_predict(data)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise = (labels == -1).sum()
        print(f"eps={eps} : clusters={n_clusters}  noise_points={noise}")


def assign_noise(embedding, labels):
    noise_mask = labels == -1
    if noise_mask.sum() == 0:
        return labels
    nbrs = NearestNeighbors(n_neighbors=1).fit(embedding[~noise_mask])
    _, indices = nbrs.kneighbors(embedding[noise_mask])
    labels[noise_mask] = labels[~noise_mask][indices.flatten()]
    return labels

def run_dbscan(embedding, eps=0.8, min_samples=5):

    model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)  # n_jobs=-1 uses all available CPU cores
    labels = model.fit_predict(embedding)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise = (labels == -1).sum()
    print(f"Clusters found  : {n_clusters}")
    print(f"Noise points    : {noise} ({noise / len(labels) * 100:.1f}%)")

    return model, labels


if __name__ == "__main__":
    CHOSEN_EPS         = 0.3
    CHOSEN_MIN_SAMPLES = 5
    
    costumer_preprocessed, costumer_featured = load_data()

    _, embedding = fit_umap(costumer_preprocessed)

    eps_exploration(embedding)

    model, labels = run_dbscan(embedding, eps=CHOSEN_EPS, min_samples=CHOSEN_MIN_SAMPLES)

    labels = assign_noise(embedding, labels)
    plot_umap_clusters(embedding, labels)

    plot_cluster_profile(costumer_preprocessed, costumer_featured, labels)
    plot_cluster_sizes(labels, title="Cluster Sizes — DBSCAN")