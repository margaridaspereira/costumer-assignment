import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift

from customer_utils import load_data, plot_cluster_sizes, plot_cluster_profile


def bandwidth_exploration(data, bandwidths=[2.15, 2.16, 2.17, 2.18, 2.2]):
    for bw in bandwidths:
        model = MeanShift(bandwidth=bw, bin_seeding=True, n_jobs=-1)
        labels = model.fit_predict(data)
        print(f"bandwidth={bw} : clusters={len(set(labels))}")


def run_meanshift(data, bandwidth = 2.17):

    model = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=-1) #n_jobs=-1 uses all available CPU cores 
    labels = model.fit_predict(data)
    n_clusters = len(set(labels))
    print(f"Clusters found  : {n_clusters}")

    return model, labels


if __name__ == "__main__":
    costumer_preprocessed, costumer_featured = load_data()

    bandwidth_exploration(costumer_preprocessed)
    CHOSEN_BW = 2.17

    model, labels = run_meanshift(costumer_preprocessed, bandwidth=CHOSEN_BW)

    plot_cluster_profile(costumer_preprocessed, costumer_featured, labels)
    plot_cluster_sizes(labels, title="Cluster Sizes — MeanShift")