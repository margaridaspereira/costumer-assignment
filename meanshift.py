import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import MeanShift, estimate_bandwidth
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


def run_meanshift(data, costumer):
    bandwidth = estimate_bandwidth(data, quantile=0.2, n_samples=500, random_state=42)
    print(f"Estimated bandwidth : {bandwidth:.4f}")

    model = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=-1)
    labels = model.fit_predict(data)
    n_clusters = len(set(labels))
    print(f"Clusters found      : {n_clusters}")

    print(f"Silhouette        : {silhouette_score(data, labels):.4f}")
    print(f"Davies-Bouldin    : {davies_bouldin_score(data, labels):.4f}")
    print(f"Calinski-Harabasz : {calinski_harabasz_score(data, labels):.4f}")
    print(f"Cluster sizes:\n{pd.Series(labels).value_counts().sort_index()}")

    # Cluster sizes
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    sizes = pd.Series(labels).value_counts().sort_index()
    axes[0].bar(sizes.index.astype(str), sizes.values, color="coral", edgecolor="black")
    axes[0].set_title(f"Cluster sizes — Mean Shift (k={n_clusters})")
    axes[0].set_xlabel("Cluster")
    axes[0].set_ylabel("Customers")

    # Export
    output = costumer[["customer_id"]].copy()
    output["cluster_meanshift"] = labels
    output.to_csv("meanshift_cluster_assignments.csv", index=False)
    print(f"Saved {len(output)} rows → meanshift_cluster_assignments.csv")

    return model, labels


if __name__ == "__main__":
    costumer_preprocessed, costumer = load_data()
    run_meanshift(costumer_preprocessed, costumer)
