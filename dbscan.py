import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

from customer_utils import load_data, plot_cluster_sizes, plot_cluster_profile


def eps_exploration(data, eps_values=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0], min_samples=5):
    for eps in eps_values:
        model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = model.fit_predict(data)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise = (labels == -1).sum()
        print(f"eps={eps} : clusters={n_clusters}  noise_points={noise}")


def run_dbscan(data, eps=1.0, min_samples=5):

    model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)  # n_jobs=-1 uses all available CPU cores
    labels = model.fit_predict(data)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise = (labels == -1).sum()
    print(f"Clusters found  : {n_clusters}")
    print(f"Noise points    : {noise} ({noise / len(labels) * 100:.1f}%)")

    return model, labels


if __name__ == "__main__":
    costumer_preprocessed, costumer_featured = load_data()

    eps_exploration(costumer_preprocessed)
    CHOSEN_EPS = 1.0
    CHOSEN_MIN_SAMPLES = 5

    model, labels = run_dbscan(costumer_preprocessed, eps=CHOSEN_EPS, min_samples=CHOSEN_MIN_SAMPLES)

    plot_cluster_profile(costumer_preprocessed, costumer_featured, labels)
    plot_cluster_sizes(labels, title="Cluster Sizes — DBSCAN")