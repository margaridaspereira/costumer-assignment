import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def load_data():
    costumer_preprocessed = pd.read_csv("costumer_preprocessed_combined.csv")
    costumer_raw = pd.read_csv("customer_info.csv")
    costumer_basket = pd.read_csv("customer_basket.csv")

    basket_agg = (
        costumer_basket
        .groupby("customer_id")
        .agg(total_transactions=("invoice_id", "count"))
        .reset_index()
    )
    costumer = (
        pd.merge(costumer_raw, basket_agg, on="customer_id", how="inner")
        .reset_index(drop=True)
    )
    return costumer_preprocessed, costumer


def elbow_curve(data, k_range=range(1, 20)):
    inertia = [
        KMeans(n_clusters=k, random_state=42, n_init="auto").fit(data).inertia_
        for k in k_range
    ]
    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), inertia, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.tight_layout()
    plt.show()
    return inertia


def silhouette_curve(data, k_range=range(2, 12)):
    scores = {
        k: silhouette_score(data, KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(data))
        for k in k_range
    }
    best_k = max(scores, key=scores.get)
    print(f"Best k by silhouette: {best_k}  (score={scores[best_k]:.4f})")

    plt.figure(figsize=(8, 4))
    plt.bar(scores.keys(), scores.values(), color="steelblue", edgecolor="black")
    plt.xlabel("k")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette scores — K-Means")
    plt.tight_layout()
    plt.show()
    return scores, best_k


def run_kmeans(data, costumer, n_clusters=6):
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = model.fit_predict(data)

    print(f"Silhouette        : {silhouette_score(data, labels):.4f}")
    print(f"Davies-Bouldin    : {davies_bouldin_score(data, labels):.4f}")
    print(f"Calinski-Harabasz : {calinski_harabasz_score(data, labels):.4f}")
    print(f"Cluster sizes:\n{pd.Series(labels).value_counts().sort_index()}")

    # Cluster sizes
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    sizes = pd.Series(labels).value_counts().sort_index()
    axes[0].bar(sizes.index.astype(str), sizes.values, color="steelblue", edgecolor="black")
    axes[0].set_title(f"Cluster sizes — K-Means (k={n_clusters})")
    axes[0].set_xlabel("Cluster")
    axes[0].set_ylabel("Customers")

    # Export
    output = costumer[["customer_id"]].copy()
    output["cluster_kmeans"] = labels
    output.to_csv("kmeans_cluster_assignments.csv", index=False)
    print(f"Saved {len(output)} rows → kmeans_cluster_assignments.csv")

    return model, labels


if __name__ == "__main__":
    CHOSEN_K = 6
    costumer_preprocessed, costumer = load_data()
    elbow_curve(costumer_preprocessed)
    silhouette_curve(costumer_preprocessed)
    run_kmeans(costumer_preprocessed, costumer, n_clusters=CHOSEN_K)
