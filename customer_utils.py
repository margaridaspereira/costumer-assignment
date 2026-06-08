import pandas as pd
import matplotlib.pyplot as plt


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
    pd.set_option('display.max_columns', None)
    print(costumer_featured[cols].groupby('cluster_id').mean())


# def load_data():
#     costumer_preprocessed = pd.read_csv("costumer_preprocessed_combined.csv")
#     costumer_raw = pd.read_csv("customer_info.csv")
#     costumer_basket = pd.read_csv("customer_basket.csv")

#     basket_agg = (
#         costumer_basket
#         .groupby("customer_id")
#         .agg(total_transactions=("invoice_id", "count"))
#         .reset_index()
#     )
#     costumer = (
#         pd.merge(costumer_raw, basket_agg, on="customer_id", how="inner")
#         .reset_index(drop=True)
#     )
#     return costumer_preprocessed, costumer


#acho melhor só ter no comparison
# print(f"Silhouette        : {silhouette_score(data, labels):.4f}")
# print(f"Davies-Bouldin    : {davies_bouldin_score(data, labels):.4f}")
# print(f"Calinski-Harabasz : {calinski_harabasz_score(data, labels):.4f}")
# print(f"Cluster sizes:\n{pd.Series(labels).value_counts().sort_index()}")

# Export
# output = costumer[["customer_id"]].copy()
# output["cluster_kmeans"] = labels
# output.to_csv("kmeans_cluster_assignments.csv", index=False)
# print(f"Saved {len(output)} rows → kmeans_cluster_assignments.csv")