import ast
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# --- PAGE CONFIGURATION (Supermarket/Retail Theme) ---
st.set_page_config(
    page_title="Cart&Cluster | Supermarket Insights",
    page_icon="🛒",
    layout="wide",
)

# Global Custom CSS Styling (Colors: Fresh Lettuce Green, Carrot Orange, Charcoal)
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        
        /* Fonts and Background */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        /* Custom Metric Cards */
        .metric-card {
            background-color: #f8f9fa;
            border-left: 5px solid #2E7D32;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            margin-bottom: 15px;
        }
        .metric-title {
            color: #555;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 600;
        }
        .metric-value {
            color: #1B5E20;
            font-size: 28px;
            font-weight: 700;
            margin-top: 5px;
        }
        
        /* Supermarket Receipt / Coupon Style Box */
        .coupon-box {
            background-color: #FFF3E0;
            border: 2px dashed #E65100;
            padding: 20px;
            border-radius: 10px;
            margin-top: 15px;
        }
        .coupon-title {
            color: #E65100;
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Coherent Graphics Styling
sns.set_theme(style="white")
plt.rcParams['font.family'] = 'sans-serif'
SUPER_PALETTE = ["#2E7D32", "#E65100", "#0277BD", "#1565C0", "#C62828", "#6A1B9A", "#4E342E"]


@st.cache_data
def parse_items(value):
    if pd.isna(value):
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
            return [parsed]
        except Exception:
            cleaned = value.strip()
            if cleaned.startswith("[") and cleaned.endswith("]"):
                cleaned = cleaned[1:-1]
            return [item.strip().strip("'\"") for item in cleaned.split(",") if item.strip()]
    return []


@st.cache_data
def load_data():
    preprocessed = pd.read_csv("costumer_preprocessed.csv")
    featured = pd.read_csv("costumer_featured.csv")
    customer_info = pd.read_csv("customer_info.csv")
    basket = pd.read_csv("customer_basket.csv")
    basket["items"] = basket["list_of_goods"].apply(parse_items)
    return preprocessed, featured, customer_info, basket


@st.cache_data
def fit_kmeans(preprocessed, n_clusters=7):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(preprocessed)
    model = KMeans(n_clusters=n_clusters, random_state=16, n_init="auto")
    labels = model.fit_predict(X_scaled)
    return model, labels, scaler


@st.cache_data
def compute_umap(preprocessed):
    reducer = umap.UMAP(n_components=2, random_state=16)
    embedding = reducer.fit_transform(preprocessed)
    return embedding


@st.cache_data
def compute_pca(preprocessed):
    pca = PCA(n_components=2)
    embedding = pca.fit_transform(preprocessed)
    return embedding, pca


@st.cache_data
def build_rules(transactions, min_support=0.02, min_confidence=0.2, top_n=5):
    if len(transactions) == 0:
        return pd.DataFrame()

    te = TransactionEncoder()
    te_array = te.fit_transform(transactions)
    df = pd.DataFrame(te_array, columns=te.columns_)
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        return pd.DataFrame()

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    if rules.empty:
        return pd.DataFrame()

    rules = rules.sort_values(["lift", "confidence"], ascending=False)
    rules["antecedent"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(list(x))))
    rules["consequent"] = rules["consequents"].apply(lambda x: ", ".join(sorted(list(x))))
    return rules.head(top_n)


@st.cache_data
def build_cluster_rules(basket, customer_info, labels):
    customer_clusters = customer_info[["customer_id"]].copy()
    customer_clusters["cluster_id"] = labels
    merged = basket.merge(customer_clusters, on="customer_id", how="inner")
    results = {}
    for cluster_id in sorted(customer_clusters["cluster_id"].unique()):
        cluster_transactions = merged.loc[merged["cluster_id"] == cluster_id, "items"].tolist()
        support = 0.02 if len(cluster_transactions) < 6000 else 0.01
        results[cluster_id] = build_rules(cluster_transactions, min_support=support, min_confidence=0.25, top_n=5)
    return results


def summarize_cluster_profiles(featured, labels):
    df = featured.copy()
    df["cluster_id"] = labels
    spend_cols = ["lifetime_spend_groceries", "lifetime_spend_electronics"]
    pct_cols = [
        "pct_groceries", "pct_electronics", "pct_vegetables", 
        "pct_nonalcohol_drinks", "pct_alcohol_drinks", "pct_meat", 
        "pct_fish", "pct_hygiene", "pct_videogames", "pct_petfood"
    ]

    profile = df.groupby("cluster_id").agg(
        customers=("cluster_id", "count"),
        avg_age=("age", "mean"),
        avg_years_as_customer=("years_as_customer", "mean"),
        avg_total_spend=("total_spend", "mean"),
        avg_promotions=("percentage_of_products_bought_promotion", "mean"),
        pct_with_loyalty=("has_loyalty_card", "mean"),
        avg_dependents=("dependents_home", "mean"),
    )

    for col in spend_cols:
        profile[f"avg_{col}"] = df.groupby("cluster_id")[col].mean()
    for col in pct_cols:
        profile[f"avg_{col}"] = df.groupby("cluster_id")[col].mean()

    return profile.sort_index().round(2)


def cluster_description(row):
    categories = {
        "🥬 Groceries & Fresh Produce": row["avg_lifetime_spend_groceries"],
        "📺 Appliances & Tech": row["avg_lifetime_spend_electronics"],
        "🥩 Butcher (Meat)": row.get("avg_pct_meat", 0),
        "🐟 Fishmonger (Fish)": row.get("avg_pct_fish", 0),
        "🎮 Video Games & Leisure": row.get("avg_pct_videogames", 0),
        "🐶 Pet Food & Supplies": row.get("avg_pct_petfood", 0),
    }
    favorite = max(categories, key=categories.get)
    
    description = [
        f"**Top Spending Category:** {favorite}  \n",
        f"**Average Age:** {row['avg_age']:.1f} years old | ",
        f"**Loyalty Card Adoption:** {row['pct_with_loyalty']:.1%} | ",
        f"**Average Lifetime Spend:** €{row['avg_total_spend']:.0f}  \n",
        f"**Promotion Sensitivity:** {row['avg_promotions']:.1%} of items bought on sale.  \n"
    ]
    
    if row["avg_years_as_customer"] > 12:
        description.append("🎯 *Profile:* Highly loyal customers who have been shopping with us for over a decade.")
    if row["avg_promotions"] > 0.45:
        description.append("🎯 *Profile:* Active bargain hunters. They primarily convert when orange promo tags are visible!")
    if row["avg_lifetime_spend_electronics"] > row["avg_lifetime_spend_groceries"]:
        description.append("🎯 *Profile:* Tech-first segment focused on upgrading home equipment and gadgets over daily groceries.")
    return "".join(description)


def campaign_from_rule(row, cluster_profile):
    if row is None or row.empty:
        return "No strong association rule found with the current thresholds."

    antecedent = row["antecedent"]
    consequent = row["consequent"]
    antecedent_lower = antecedent.lower()
    consequent_lower = consequent.lower()

    if "eggs" in consequent_lower or "eggs" in antecedent_lower:
        return f"🍳 **Family Breakfast Combo:** Get 1 pack of XL Eggs at a 20% direct discount when purchasing {antecedent}!"
    if any(item in consequent_lower for item in ["salad", "tomatoes"]):
        return f"🥗 **Healthy Living Campaign:** When buying {antecedent}, get the accompanying {consequent} at half price!"
    if "airpods" in antecedent_lower or "airpods" in consequent_lower:
        return f"⚡ **Premium Tech Bundle:** Buy {antecedent} and get an immediate discount voucher for {consequent} to equip your home."
    if "energy drink" in antecedent_lower or "energy drink" in consequent_lower:
        return f"🚀 **Maximum Energy Pack:** Grab {antecedent} and {consequent} together for a special bundled shelf price."
    if any(item in consequent_lower for item in ["pet food", "dog food", "chicken"]):
        return f"🦴 **Four-Legged Friend Special:** Enjoy a 20% discount on {consequent} when adding {antecedent} to your shopping cart."
    if cluster_profile.get("avg_promotions", 0) > 0.45:
        return f"🏷️ **Super 50% Voucher:** Half off on {consequent} when purchased together with {antecedent}."

    return f"🛒 **Cross-Selling Suggestion:** Display {consequent} prominently on the same promotional endcap or island as {antecedent} to stimulate impulse buys."


def main():
    preprocessed, featured, customer_info, basket = load_data()
    model, labels, scaler = fit_kmeans(preprocessed, n_clusters=7)
    umap_embedding = compute_umap(preprocessed)
    pca_embedding, pca = compute_pca(preprocessed)
    cluster_profiles = summarize_cluster_profiles(featured, labels)
    rules_by_cluster = build_cluster_rules(basket, customer_info, labels)

    # --- THEMED SIDEBAR ---
    st.sidebar.image("https://img.icons8.com/fluent/96/000000/shopping-cart.png", width=80)
    st.sidebar.title("Cart&Cluster Portal")
    st.sidebar.subheader("Campaign Management")
    
    sections = [
        "🏪 Executive Summary",
        "📊 Data Insights (EDA)",
        "🎯 Customer Profiles (Clustering)",
        "🏷️ Campaigns & Promotions",
        "📈 Strategic Recommendations",
    ]
    section = st.sidebar.radio("Navigate Sections:", sections)
    
    st.sidebar.markdown("---")
    st.sidebar.caption("💡 *Tip:* Filter different clusters in sections 3 and 4 to design targeted weekly discount flyers.")

    # --- SECTION 1: EXECUTIVE SUMMARY ---
    if section == "🏪 Executive Summary":
        st.subheader("Retail Control & Strategy Dashboard")
        st.title("🏪 Customer Intelligence & Shopping Basket Insights")
        
        # Clean Metric Card Grid
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Loyal Customers</div><div class="metric-value">{len(featured):,}</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Shopping Baskets Analisadas</div><div class="metric-value">{len(basket):,}</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Strategic Segments</div><div class="metric-value">7 Groups</div></div>', unsafe_allow_html=True)
        
        st.markdown("""
        ### 🎯 Core Business Objectives Achieved
        * **Lifestyle Mapping:** Successfully divided our customer base into 7 clear behavioral profiles (ranging from pet-owning families to young tech-focused segments).
        * **Margin Optimization:** Instead of issuing generic 50% store-wide discounts, we pinpointed exactly which groups require incentives to convert versus who buys out of brand loyalty.
        * **Affinity Campaigns (Cross-Selling):** Extracted market basket association rules from `customer_basket` to bridge disparate item categories (e.g., cross-promotions between the Butcher and Fishmonger sections).
        """)

    # --- SECTION 2: EDA ---
    elif section == "📊 Data Insights (EDA)":
        st.title("📊 Supermarket Consumer Behavior Analysis")
        
        tab1, tab2, tab3 = st.tabs(["📋 Data Audit", "👥 Demographics & Frequency", "🥩 Spending by Category"])
        
        with tab1:
            st.subheader("Database Record Volumes")
            overview = {
                "Total Registered Customers (Info)": len(customer_info),
                "Processed Customers for Modeling": len(preprocessed),
                "Unique Items in Cart Catalog": basket["items"].explode().nunique(),
            }
            st.table(pd.DataFrame.from_dict(overview, orient="index", columns=["Records"]))
            
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                sns.histplot(featured["age"].dropna(), bins=20,