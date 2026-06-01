import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

from customer_utils import load_data, plot_cluster_sizes, plot_cluster_profile



def elbow_curve(data, k_range=range(1, 20)):
    inertia = [
        KMeans(n_clusters=k, random_state=16, n_init="auto").fit(data).inertia_
        for k in k_range
    ]

    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), inertia, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.xticks(list(k_range))
    plt.tight_layout()
    plt.show()
    return inertia


def silhouette_scores(data, k_range=range(2, 12)):
    scores = {
        k: silhouette_score(data, KMeans(n_clusters=k, random_state=16, n_init="auto").fit_predict(data))
        for k in k_range
    }

    best_k = max(scores, key=lambda k: scores[k])
    print(f"Best k by silhouette: {best_k}  (score={scores[best_k]:.4f})")

    plt.figure(figsize=(8, 5))
    plt.bar(scores.keys(), scores.values(), color="steelblue", edgecolor="black")
    plt.xlabel("k")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette scores — K-Means")
    plt.tight_layout()
    plt.show()
    return scores, best_k


def silhouette_plot(data, n_clusters):
    labels = KMeans(n_clusters=n_clusters, random_state=16, n_init="auto").fit_predict(data)
    sample_scores = silhouette_samples(data, labels)

    plt.figure(figsize=(8, 5))
    y_lower = 0
    yticks_positions = []
    for i in range(n_clusters):
        cluster_scores = sorted(sample_scores[labels == i])
        y_upper = y_lower + len(cluster_scores)
        plt.barh(range(y_lower, y_upper), cluster_scores)
        yticks_positions.append((y_lower + y_upper) / 2)
        y_lower = y_upper

    plt.yticks(yticks_positions, range(n_clusters))
    plt.axvline(x=0, color="black")
    plt.axvline(x=np.mean(sample_scores), color="red", linestyle="--", label=f"Average silhouette = {np.mean(sample_scores):.2f}")
    plt.xlabel("Silhouette score")
    plt.ylabel("Cluster")
    plt.title(f"Silhouette plot — K-Means")
    plt.legend()
    plt.tight_layout()
    plt.show()

def run_kmeans(data, n_clusters=6):
    
    model = KMeans(n_clusters=n_clusters, random_state=16, n_init="auto")
    labels = model.fit_predict(data)

    return model, labels


if __name__ == "__main__":
    CHOSEN_K = 5
    costumer_preprocessed, costumer_featured = load_data()

    #elbow_curve(costumer_preprocessed)
    #silhouette_scores(costumer_preprocessed)
    silhouette_plot(costumer_preprocessed, n_clusters=CHOSEN_K)

    model, labels =  run_kmeans(costumer_preprocessed, n_clusters=CHOSEN_K)

    plot_cluster_profile(costumer_preprocessed, costumer_featured, labels)
    plot_cluster_sizes(labels, title=f"Cluster sizes — K-Means")
    