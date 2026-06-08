from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="umap")

from customer_utils import load_data, plot_cluster_sizes, plot_cluster_profile


def fit_umap(data, n_components=2, random_state=16):
    """Reduce dimensionality with UMAP before clustering."""
    reducer = UMAP(n_components=n_components, random_state=random_state)
    embedding = reducer.fit_transform(data)
    return reducer, embedding


def eps_exploration(data, eps_values=[0.3, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5], min_samples=10):
    for eps in eps_values:
        model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = model.fit_predict(data)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise = (labels == -1).sum()
        print(f"eps={eps} : clusters={n_clusters}  noise_points={noise}")


def run_dbscan(embedding, eps=0.8, min_samples=10):

    model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)  # n_jobs=-1 uses all available CPU cores
    labels = model.fit_predict(embedding)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise = (labels == -1).sum()
    print(f"Clusters found  : {n_clusters}")
    print(f"Noise points    : {noise} ({noise / len(labels) * 100:.1f}%)")

    return model, labels

def plot_umap_clusters(embedding, labels, title="DBSCAN clusters — UMAP projection"):
    """Scatter plot of the 2-D UMAP embedding coloured by cluster."""
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        embedding[:, 0], embedding[:, 1],
        c=labels, cmap="tab10", s=5, alpha=0.6
    )
    plt.colorbar(scatter, label="Cluster")
    plt.title(title)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    CHOSEN_EPS         = 0.3
    CHOSEN_MIN_SAMPLES = 5
    
    costumer_preprocessed, costumer_featured = load_data()

    data_for_umap = costumer_preprocessed.drop(columns='has_loyalty_card')  # DBSCAN doesn't handle binary vars well, so we drop it for the UMAP embedding used by DBSCAN

    _, embedding = fit_umap(data_for_umap)

    eps_exploration(embedding)
    # CHOSEN_EPS = 1.0
    # CHOSEN_MIN_SAMPLES = 5

    model, labels = run_dbscan(embedding, eps=CHOSEN_EPS, min_samples=CHOSEN_MIN_SAMPLES)

    plot_umap_clusters(embedding, labels)

    plot_cluster_profile(costumer_preprocessed, costumer_featured, labels)
    plot_cluster_sizes(labels, title="Cluster Sizes — DBSCAN")