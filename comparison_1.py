import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from customer_utils import load_data, plot_cluster_sizes, plot_cluster_profile
from kmeans_2 import  run_kmeans
from dbscan import run_dbscan
from hierarchical import run_hierarchical

# Metrics

def compute_metrics(data, labels_dict):
    rows = {}
    for name, labels in labels_dict.items():
        rows[name] = {
            "n_clusters"        : len(set(labels)),
            "silhouette"        : round(silhouette_score(data, labels), 4),
            "davies_bouldin"    : round(davies_bouldin_score(data, labels), 4),
            "calinski_harabasz" : round(calinski_harabasz_score(data, labels), 4),
        }
    return pd.DataFrame(rows).T

# Plots

def plot_umap_comparison(embedding, labels_dict):
    """One UMAP projection, one subplot per clustering solution."""
    n = len(labels_dict)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))

    for ax, (name, labels) in zip(axes, labels_dict.items()):
        scatter = ax.scatter(
            embedding[:, 0], embedding[:, 1],
            c=labels, cmap="tab10", s=10, alpha=0.7
        )
        plt.colorbar(scatter, ax=ax, label="Cluster")
        ax.set_title(f"UMAP — {name} (k={len(set(labels))})")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

    fig.suptitle("UMAP projection — clustering comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_pca_comparison(embedding, labels_dict):
    """One PCA projection, one subplot per clustering solution."""
    n = len(labels_dict)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))

    for ax, (name, labels) in zip(axes,labels_dict.items()):
        scatter = ax.scatter(
            embedding[:,0], embedding[:,1],
            c=labels, cmap="tab10", s=10, alpha=0.7
                            )
        plt.colorbar(scatter, ax=ax, label="Cluster")
        ax.set_title(f"PCA — {name} (k={len(set(labels))})")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
    fig.suptitle("PCA projection — clustering comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()



def plot_metrics(metrics_df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ["steelblue", "seagreen", "coral"]
    models = metrics_df.index.tolist()

    specs = [
        ("silhouette",        "higher", "Silhouette Score\n(higher = better)"),
        ("davies_bouldin",    "lower",  "Davies-Bouldin Score\n(lower = better)"),
        ("calinski_harabasz", "higher", "Calinski-Harabasz Score\n(higher = better)"),
    ]

    for ax, (col, better, title) in zip(axes, specs):
        bars = ax.bar(models, metrics_df[col], color=colors, edgecolor="black")
        ax.set_title(title)
        ax.set_ylabel(col)
        best = metrics_df[col].idxmax() if better == "higher" else metrics_df[col].idxmin()
        bars[models.index(best)].set_edgecolor("gold")
        bars[models.index(best)].set_linewidth(3)

    plt.tight_layout()
    plt.show()


def plot_cluster_sizes_comparison(labels_dict):
    n = len(labels_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    colors = ["steelblue", "seagreen", "coral"]

    for ax, (name, labels), color in zip(axes, labels_dict.items(), colors):
        plot_cluster_sizes(labels, title=f"{name} — Cluster Sizes", color=color, ax=ax)

    plt.tight_layout()
    plt.show()


def plot_heatmaps(data, labels_dict):
    n = len(labels_dict)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 6))

    for ax, (name, labels) in zip(axes, labels_dict.items()):
        tmp = data.copy()
        tmp["cluster"] = labels
        profile = tmp.groupby("cluster").mean()
        sns.heatmap(profile.T, cmap="RdBu_r", center=0, linewidths=0.5, ax=ax, annot=False, cbar=False)
        ax.set_title(f"{name} (k={len(set(labels))})")
        ax.set_xlabel("Cluster")

    fig.suptitle("Feature means per cluster (scaled) — all methods", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_crosstab(labels_a, labels_b, name_a="K-Means", name_b="Hierarchical"):
    crosstab = pd.crosstab(
        pd.Series(labels_a, name=name_a),
        pd.Series(labels_b, name=name_b)
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(crosstab, annot=True, fmt="d", cmap="Blues", ax=ax, linewidths=0.5)
    ax.set_title(f"{name_a} vs {name_b} — customer overlap")
    plt.tight_layout()
    plt.show()



def print_verdict(metrics_df):
    print("=" * 55)
    print("SUMMARY")
    print("=" * 55)
    print(metrics_df.to_string())
    print()

    best_sil = metrics_df["silhouette"].idxmax()
    best_db  = metrics_df["davies_bouldin"].idxmin()
    best_ch  = metrics_df["calinski_harabasz"].idxmax()

    print(f"Best Silhouette        : {best_sil}")
    print(f"Best Davies-Bouldin    : {best_db}")
    print(f"Best Calinski-Harabasz : {best_ch}")

    votes  = pd.Series([best_sil, best_db, best_ch]).value_counts()
    winner = votes.idxmax()
    print(f"\nRecommended algorithm  : {winner} ({votes[winner]}/3 metrics)")
    return winner


def export_clusters(costumer_featured, labels, preprocessed):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    # Clientes com cluster atribuído (os que passaram pelo pré-processamento)
    assignments = costumer_featured[["customer_id"]].copy()
    assignments["cluster"] = labels

    # Clientes sem transações no basket — excluídos pelo inner join no pré-processamento
    info = pd.read_csv("customer_info.csv")
    missing_ids = set(info["customer_id"]) - set(assignments["customer_id"])
    missing_df = info[info["customer_id"].isin(missing_ids)].copy()

    # Replicar o mesmo feature engineering do EDA
    current_year = 2024
    missing_df["age"] = current_year - pd.to_datetime(
        missing_df["customer_birthdate"], errors="coerce", format="mixed"
    ).dt.year
    missing_df["years_as_customer"] = current_year - missing_df["year_first_transaction"]
    missing_df["has_loyalty_card"] = missing_df["loyalty_card_number"].notna().astype(int)
    missing_df["total_transactions"] = 0
    missing_df["customer_gender"] = (missing_df["customer_gender"] == "M").astype(float)

    # Imputar NaNs com a mediana do preprocessed
    feature_cols = list(preprocessed.columns)
    missing_features = missing_df[feature_cols].copy()
    for col in feature_cols:
        if missing_features[col].isna().any():
            missing_features[col] = missing_features[col].fillna(preprocessed[col].median())

    # Escalar e prever cluster
    scaler = StandardScaler()
    scaler.fit(preprocessed)
    missing_scaled = scaler.transform(missing_features)

    kmeans = KMeans(n_clusters=len(set(labels)), random_state=16, n_init="auto")
    kmeans.fit(preprocessed)
    missing_labels = kmeans.predict(missing_scaled)

    missing_out = pd.DataFrame({
        "customer_id": missing_df["customer_id"].values,
        "cluster"    : missing_labels,
    })

    final = (
        pd.concat([assignments, missing_out], ignore_index=True)
        .sort_values("customer_id")
        .reset_index(drop=True)
    )
    final.to_csv("cluster_assignments.csv", index=False)
    print(f"Saved {len(final)} rows → cluster_assignments.csv")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    KMEANS_K            = 7
    HIERARCHICAL_K      = 7
    DBSCAN_EPS          = 0.8
    DBSCAN_MIN_SAMPLES  = 10
    COLS_TO_DROP        = ['customer_gender', 'has_loyalty_card']

    costumer_preprocessed, costumer_featured = load_data()

    # Fit UMAP once — shared projection for all models (sem variáveis binárias)
    print("Computing shared UMAP projection...")
    reducer = umap.UMAP(n_components=2, random_state=16)
    embedding = reducer.fit_transform(costumer_preprocessed.drop(columns=COLS_TO_DROP))

    # Fit PCA once — shared projection for all models
    print("Computing shared PCA projection...")
    pca = PCA(n_components=2)
    pca_embedding = pca.fit_transform(costumer_preprocessed)
    print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")

    print("\n── Fitting K-Means ──")
    _, labels_kmeans = run_kmeans(costumer_preprocessed, n_clusters=KMEANS_K)

    print("\n── Fitting Hierarchical (Ward) ──")
    _, labels_ward = run_hierarchical(costumer_preprocessed, n_clusters=HIERARCHICAL_K)

    print("\n── Fitting DBSCAN ──")
    _, labels_dbscan = run_dbscan(embedding, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)

    labels_dict = {
        "K-Means"     : labels_kmeans,
        "Hierarchical": labels_ward,
        "DBSCAN"      : labels_dbscan,
    }

    print("\n── Computing Metrics ──")
    metrics_df = compute_metrics(costumer_preprocessed, labels_dict)

    # All plots
    plot_umap_comparison(embedding, labels_dict)
    plot_pca_comparison(pca_embedding, labels_dict)
    plot_metrics(metrics_df)
    plot_cluster_sizes_comparison(labels_dict)
    plot_heatmaps(costumer_preprocessed, labels_dict)

    # Cluster overlap
    plot_crosstab(labels_kmeans, labels_ward,   "K-Means", "Hierarchical")
    plot_crosstab(labels_kmeans, labels_dbscan, "K-Means", "DBSCAN")
    plot_crosstab(labels_ward,   labels_dbscan, "Hierarchical", "DBSCAN")

    winner = print_verdict(metrics_df)
    final_labels = labels_dict[winner]

    plot_cluster_profile(costumer_preprocessed, costumer_featured, final_labels)

    # Export cluster assignments (from the recommended algorithm) — inclui todos os 33k clientes
    export_clusters(costumer_featured, final_labels, costumer_preprocessed)