import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

def load_basket_data():
    basket = pd.read_csv("customer_basket.csv")
    clusters = pd.read_csv("cluster_assignments.csv")
    basket["items"] = basket["list_of_goods"].str.strip("[]").str.replace("'", "").str.split(", ")
    basket = basket.merge(clusters, on="customer_id", how="inner")
    return basket

def get_cluster_transactions(basket, cluster_id):
    cluster_basket = basket[basket["cluster"] == cluster_id]["items"].tolist()
    return cluster_basket

def get_association_rules(transactions, min_support=0.01, min_confidence=0.2):
    te = TransactionEncoder()
    te_array = te.fit_transform(transactions)
    df = pd.DataFrame(te_array, columns=te.columns_)
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules = rules.sort_values("lift", ascending=False)
    return rules

def get_all_cluster_rules(basket, min_support=0.01, min_confidence=0.2, top_n=5):
    results = {}
    for cluster_id in sorted(basket["cluster"].unique()):
        print(f"Processing cluster {cluster_id}...")
        transactions = get_cluster_transactions(basket, cluster_id)
        
        # Adjust min_support for small clusters
        n_transactions = len(transactions)
        support = min_support if n_transactions >= 1000 else 0.02
        
        rules = get_association_rules(transactions, support, min_confidence)
        results[cluster_id] = rules.head(top_n)
    return results

if __name__ == "__main__":
    basket = load_basket_data()
    print(basket.head())
    
    all_rules = get_all_cluster_rules(basket)
    for cluster_id, rules in all_rules.items():
        print(f"\n=== Cluster {cluster_id} ===")
        print(rules[["antecedents", "consequents", "support", "confidence", "lift"]])