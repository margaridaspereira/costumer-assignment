import pandas as pd
import matplotlib.pyplot as plt
from umap import UMAP
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="umap")


def load_data():
    costumer_preprocessed = pd.read_csv("costumer_preprocessed.csv")
    costumer_featured = pd.read_csv("costumer_featured.csv")

    return costumer_preprocessed, costumer_featured


def plot_cluster_sizes(labels, title="Cluster Sizes", color = "steelblue", ax=None):
    sizes = pd.Series(labels).value_counts().sort_index()
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(sizes.index.astype(str), sizes.values, edgecolor="black", color=color)
    ax.set_title(title)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Customers")
    if standalone:
        plt.tight_layout()
        plt.show()


def plot_cluster_profile(costumer_preprocessed, costumer_featured, labels):
    costumer_featured['cluster_id'] = labels
    cols = list(costumer_preprocessed.columns) + ['cluster_id']
    profile = costumer_featured[cols].groupby('cluster_id').mean().round(3)
    print(profile)
    profile.to_csv("cluster_profile.csv")
    print("Saved : cluster_profile.csv")


def fit_umap(data, n_components=2, random_state=16):
    """Reduce dimensionality with UMAP before clustering."""
    reducer = UMAP(n_components=n_components, random_state=random_state)
    embedding = reducer.fit_transform(data)
    return reducer, embedding


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
