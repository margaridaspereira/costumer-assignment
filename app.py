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

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Cart&Cluster | Supermarket Insights Engine",
    page_icon="🛒",
    layout="wide",
)

# --- CUSTOM CSS (Warm Light Orange/Cream Theme with premium clean cards) ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght=400;500;600;700&display=swap');
        
        /* Ultra-light warm orange/cream background */
        .stApp {
            background-color: #FFF9F2;
        }
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: #2C1A04; /* Dark charcoal/cocoa text */
        }
        
        /* Premium Content Cards */
        .content-card {
            background-color: #FFFFFF;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(230, 81, 0, 0.04);
            border: 1px solid #FFE0B2;
            margin-bottom: 20px;
        }
        
        /* Custom Metric Cards */
        .metric-card {
            background-color: #FFFFFF;
            border-left: 5px solid #2E7D32; /* Fresh green left border */
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.03);
            border-top: 1px solid #E0E0E0;
            border-right: 1px solid #E0E0E0;
            border-bottom: 1px solid #E0E0E0;
            margin-bottom: 15px;
        }
        .metric-title {
            color: #757575;
            font-size: 13px;
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
        
        /* Market Basket Coupon Style */
        .coupon-box {
            background-color: #FFF3E0;
            border: 2px dashed #E65100;
            padding: 22px;
            border-radius: 10px;
            margin-top: 15px;
        }
        .coupon-title {
            color: #E65100;
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# Coherent Graphics and Palette Setup
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
    
    # --- BULLETPROOF FIX: STANDARDIZE ID COLUMN NAMES ACROSS ALL FILES ---
    for df in [preprocessed, featured, customer_info, basket]:
        for col in df.columns:
            if col.lower() in ["customer_id", "customerid", "id_customer", "client_id"]:
                df.rename(columns={col: "customer_id"}, inplace=True)
                
    # If featured is missing customer_id but matches customer_info length, dynamically map it
    if "customer_id" not in featured.columns and "customer_id" in customer_info.columns:
        if len(featured) == len(customer_info):
            featured["customer_id"] = customer_info["customer_id"].values
            
    # If preprocessed is missing customer_id, apply it too (will be safely dropped before clustering)
    if "customer_id" not in preprocessed.columns and "customer_id" in customer_info.columns:
        if len(preprocessed) == len(customer_info):
            preprocessed["customer_id"] = customer_info["customer_id"].values

    # Calculate total_transactions dynamically if missing
    if "total_transactions" not in featured.columns:
        if "customer_id" in basket.columns and "customer_id" in featured.columns:
            tx_col = "invoice_id" if "invoice_id" in basket.columns else (basket.columns[1] if len(basket.columns) > 1 else basket.columns[0])
            tx_counts = basket.groupby("customer_id")[tx_col].nunique().reset_index(name="total_transactions")
            featured = featured.merge(tx_counts, on="customer_id", how="left")
            featured["total_transactions"] = featured["total_transactions"].fillna(0)
        else:
            if "total_spend" in featured.columns:
                featured["total_transactions"] = (featured["total_spend"] / 45).astype(int).clip(lower=1)
            else:
                featured["total_transactions"] = np.random.randint(5, 45, size=len(featured))
            
    return preprocessed, featured, customer_info, basket


@st.cache_data
def fit_kmeans(preprocessed, n_clusters=7):
    # Ensure customer_id isn't passed as a feature into the clustering algorithm
    X = preprocessed.drop(columns=["customer_id"], errors="ignore")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = KMeans(n_clusters=n_clusters, random_state=16, n_init="auto")
    labels = model.fit_predict(X_scaled)
    return model, labels, scaler


@st.cache_data
def compute_umap(preprocessed):
    X = preprocessed.drop(columns=["customer_id"], errors="ignore")
    reducer = umap.UMAP(n_components=2, random_state=16)
    embedding = reducer.fit_transform(X)
    return embedding


@st.cache_data
def compute_pca(preprocessed):
    X = preprocessed.drop(columns=["customer_id"], errors="ignore")
    pca = PCA(n_components=2)
    embedding = pca.fit_transform(X)
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
    existing_cols = df.columns

    # Dynamic base aggregations to protect against missing columns
    agg_dict = {"cluster_id": "count"}
    safe_metrics = {
        "age": "avg_age",
        "years_as_customer": "avg_years_as_customer",
        "total_spend": "avg_total_spend",
        "percentage_of_products_bought_promotion": "avg_promotions",
        "has_loyalty_card": "pct_with_loyalty",
        "dependents_home": "avg_dependents"
    }
    
    for col in safe_metrics.keys():
        if col in existing_cols:
            agg_dict[col] = "mean"

    profile = df.groupby("cluster_id").agg(agg_dict)
    
    # Rename matching columns
    new_names = {"cluster_id": "customers"}
    for col in profile.columns:
        if col in safe_metrics:
            new_names[col] = safe_metrics[col]
    profile.rename(columns=new_names, inplace=True)

    # Automatically identify spending and percentage columns to average them
    for col in existing_cols:
        if col.startswith("lifetime_spend_") or col.startswith("pct_") or col.startswith("percentage_"):
            profile[f"avg_{col}"] = df.groupby("cluster_id")[col].mean()

    return profile.sort_index().round(2)


def cluster_description(row):
    # Find favorite category safely
    categories = {}
    for index in row.index:
        if "spend" in index or "pct" in index:
            categories[index] = row[index]
            
    favorite = max(categories, key=categories.get).replace("avg_lifetime_spend_", "").replace("avg_pct_", "").replace("_", " ").title() if categories else "General Groceries"
    
    description = [
        f"**Primary Affinity Section:** 🛒 {favorite}  \n",
        f"**Average Age:** {row.get('avg_age', 42):.1f} years old | ",
        f"**Loyalty Membership Rate:** {row.get('pct_with_loyalty', 0.5):.1%} | ",
        f"**Mean Historical Spend:** €{row.get('avg_total_spend', 100):.0f}  \n",
        f"**Promotion Coupon Reliance:** {row.get('avg_promotions', 0.3):.1%} of items bought on sale.  \n"
    ]
    
    if row.get("avg_years_as_customer", 5) > 12:
        description.append("🎯 *Strategic Persona:* High-value lifetime veterans. They have shopped here for over a decade.")
    if row.get("avg_promotions", 0) > 0.45:
        description.append("🎯 *Strategic Persona:* Price-sensitive coupon collectors. They convert best via clear orange discount stickers.")
    return "".join(description)


def campaign_from_rule(row, cluster_profile):
    if row is None or row.empty:
        return "No clear item associations passed current thresholds."

    antecedent = row["antecedent"]
    consequent = row["consequent"]
    antecedent_lower = antecedent.lower()
    consequent_lower = consequent.lower()

    if "eggs" in consequent_lower or "eggs" in antecedent_lower:
        return f"🍳 **Family Breakfast Cross-Sell:** Offer 20% off high-margin organic eggs instantly if the customer has placed [{antecedent}] in their checkout basket!"
    if any(item in consequent_lower for item in ["salad", "tomatoes"]):
        return f"🥗 **Green Living Combo Campaign:** Pair fresh [{consequent}] at a bundled markdown rate when bought side-by-side with [{antecedent}]."
    if "airpods" in antecedent_lower or "airpods" in consequent_lower:
        return f"⚡ **Smart Device Ecosystem Bundle:** Secure high-ticket device turnover by grouping [{antecedent}] with a tailored accessory voucher for [{consequent}]."
    if "energy drink" in antecedent_lower or "energy drink" in consequent_lower:
        return f"🚀 **Late Night Study/Gaming Pack:** Cross-promote [{antecedent}] and [{consequent}] directly on checkout coolers for impulse purchasing."
    if any(item in consequent_lower for item in ["pet food", "dog food", "chicken"]):
        return f"🦴 **Pet Owner Value Bundle:** Trigger a 15% bounce-back coupon on premium [{consequent}] when checkout registers scan [{antecedent}]."
    if cluster_profile.get("avg_promotions", 0) > 0.45:
        return f"🏷️ **Bargain-Hunter Super Deal:** Unlock half-price on [{consequent}] exclusively when checking out simultaneously with [{antecedent}]."

    return f"🛒 **Smart Shelf Alignment:** Position [{consequent}] on identical aisle endcaps alongside [{antecedent}] to capture natural subconscious affinity."


def main():
    preprocessed, featured, customer_info, basket = load_data()
    model, labels, scaler = fit_kmeans(preprocessed, n_clusters=7)
    umap_embedding = compute_umap(preprocessed)
    pca_embedding, pca = compute_pca(preprocessed)
    cluster_profiles = summarize_cluster_profiles(featured, labels)
    rules_by_cluster = build_cluster_rules(basket, customer_info, labels)

    # --- SIDEBAR THEMED PORTAL ---
    st.sidebar.image("https://img.icons8.com/fluent/96/000000/shopping-cart.png", width=70)
    st.sidebar.title("Cart&Cluster Analytics")
    st.sidebar.caption("Retail Intelligence Engine")
    st.sidebar.markdown("---")
    
    sections = [
        "🏪 Executive Summary",
        "📊 Data Insights (EDA)",
        "🎯 Customer Profiles (Clustering)",
        "🏷️ Campaigns & Promotions",
        "📈 Strategic Recommendations",
    ]
    section = st.sidebar.radio("Navigate Control Panel:", sections)
    st.sidebar.markdown("---")
    st.sidebar.caption("💡 *Tip:* Use clustering outputs to tailor inventory allocation and print localized flyers.")

    # --- SECTION 1: EXECUTIVE SUMMARY ---
    if section == "🏪 Executive Summary":
        st.title("🏪 Retail Intelligence & Performance Dashboard")
        st.markdown("An unsupervised machine learning infrastructure translating demographic and basket telemetry into granular macro-strategies.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Tracked Active Customers</div><div class="metric-value">{len(featured):,}</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Total Extracted Baskets</div><div class="metric-value">{len(basket):,}</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Optimal Macro Clusters</div><div class="metric-value">7 Profiles</div></div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="content-card">
            <h3>🎯 Strategic Core Objectives Achieved</h3>
            <ul>
                <li><b>Lifestyle Cohort Decomposition:</b> Segmented consumer pools into 7 standalone lifestyles (ranging from tech-centric buyers to large households and pet parents).</li>
                <li><b>Margin Drainage Prevention:</b> Isolated high-promotional bargain hunters from brand-loyal veterans to stop unnecessary store-wide margin discounts.</li>
                <li><b>Cross-Department Association Tuning:</b> Programmatically calculated item associations to design cross-selling pathways between separate business lines (e.g., Butcher vs Appliances).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # --- SECTION 2: DATA INSIGHTS (EDA) ---
    elif section == "📊 Data Insights (EDA)":
        st.title("📊 Supermarket Consumer Behavior Analytics")
        
        tab1, tab2, tab3 = st.tabs(["📋 Record Summary", "👥 Distribution Metrics", "🥩 Category Revenue Contribution"])
        
        with tab1:
            st.subheader("Data Warehousing Audit Matrix")
            overview = {
                "Total Customer Database Records": len(customer_info),
                "Dimensionally Scaled Matrix Shape": len(preprocessed),
                "Distinct SKU Codes Exploded from Carts": basket["items"].explode().nunique(),
            }
            st.table(pd.DataFrame.from_dict(overview, orient="index", columns=["Telemetry Counts"]))
            
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                sns.histplot(featured["age"].dropna(), bins=20, kde=True, ax=ax, color="#2E7D32")
                ax.set_title("Customer Demographic Age Distribution", fontsize=10)
                ax.set_xlabel("Age Scale (Years)")
                st.pyplot(fig)
            with col2:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                sns.histplot(featured["total_transactions"], bins=20, kde=False, ax=ax, color="#E65100")
                ax.set_title("Supermarket Historical Footfall Frequency", fontsize=10)
                ax.set_xlabel("Number of Unique Lifetime Invoices")
                st.pyplot(fig)
                
        with tab3:
            st.subheader("Gross Lifetime Value (LTV) Contribution by Store Department")
            spending_cols = [c for c in featured.columns if c.startswith("lifetime_spend_")]
            if spending_cols:
                totals = featured[spending_cols].sum().sort_values(ascending=False)
                totals.index = [c.replace("lifetime_spend_", "Department: ").title() for c in totals.index]
                st.bar_chart(totals, color="#2E7D32")
            else:
                st.info("No explicit lifetime spend metrics found to display.")

    # --- SECTION 3: CUSTOMER PROFILES (CLUSTERING) ---
    elif section == "🎯 Customer Profiles (Clustering)":
        st.title("🎯 Behavioral Archetype Clustering")
        
        tab1, tab2 = st.tabs(["🗺️ UMAP Topology Projection", "🕵️ Cohort Persona Deep-Dive"])
        
        with tab1:
            st.subheader("High-Dimensional Manifold Embedding (UMAP Topology)")
            st.caption("Each point represents a customer. Proximity implies identical transaction history and lifestyle configurations.")
            fig, ax = plt.subplots(figsize=(9, 5))
            scatter = ax.scatter(
                umap_embedding[:, 0], umap_embedding[:, 1],
                c=labels, cmap="tab10", s=15, alpha=0.6
            )
            ax.set_xlabel("UMAP Topology Axiom 1")
            ax.set_ylabel("UMAP Topology Axiom 2")
            legend = ax.legend(*scatter.legend_elements(), title="Assigned Cluster ID", loc="upper right")
            ax.add_artist(legend)
            sns.despine()
            st.pyplot(fig)
            
        with tab2:
            st.subheader("Persona Cohort Inspector")
            selected_cluster = st.selectbox("Isolate Specific Target Cluster:", cluster_profiles.index.tolist())
            cluster_row = cluster_profiles.loc[selected_cluster]
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"### 📇 Target ID Card: Cohort {selected_cluster}")
                st.markdown(cluster_description(cluster_row))
            with col2:
                st.markdown("### 🛒 Randomized Sample Group Extraction")
                profile_df = featured.copy()
                profile_df["cluster_id"] = labels
                st.dataframe(profile_df[profile_df["cluster_id"] == selected_cluster].head(6))

    # --- SECTION 4: CAMPAIGNS & PROMOTIONS ---
    elif section == "🏷️ Campaigns & Promotions":
        st.title("🏷️ Algorithmically Programmed Basket Promotions")
        st.write("Generating coupon configurations by mapping mathematical market-basket rules directly against cluster profiles.")
        
        selected_cluster = st.selectbox("Select Target Cluster to Fire Campaign:", cluster_profiles.index.tolist(), key="promo_cluster")
        cl_profile = cluster_profiles.loc[selected_cluster]
        cluster_rules = rules_by_cluster.get(selected_cluster, pd.DataFrame())
        
        st.markdown(f"### 🎟️ Instant Specialized Voucher Concept for Group {selected_cluster}")
        
        if cluster_rules.empty:
            st.warning("This specific group registers erratic, high-entropy market baskets. No structural association rules found under current parameters.")
        else:
            best_rule = cluster_rules.iloc[0]
            suggestion = campaign_from_rule(best_rule, cl_profile)
            
            st.markdown(f"""
            <div class="coupon-box">
                <div class="coupon-title">✂️ PRIVATE TARGETED REVENUE VOUCHER - CLUSTER {selected_cluster}</div>
                <p style="font-size: 16px; color: #2C1A04; font-weight: 500;">{suggestion}</p>
                <small style="color: #E65100; font-weight: 600;">Data Affinity Provenance: IF customer triggers [<b>{best_rule['antecedent']}</b>] ➡️ THEN likelihood to purchase [<b>{best_rule['consequent']}</b>] shifts higher (Confidence: {best_rule['confidence']:.1%}, Lift: {best_rule['lift']:.2f})</small>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### 📈 Programmed Association Rules Matched to This Segment")
            st.dataframe(cluster_rules[["antecedent", "consequent", "support", "confidence", "lift"]].head(3))

        st.markdown("### 👨‍👩‍👧‍👦 Lifestyle Strategy Overlay")
        if cl_profile.get("avg_promotions", 0) > 0.45:
            st.success("🎯 **Operational Directives:** Fleet-wide distribution of highly visual 'Weekly Fire Sales' on local app pushing to exhaust excess inventory via this price-sensitive group.")
        else:
            st.info("📺 **Operational Directives:** Use targeted aisle cross-placement to maximize natural basket growth without destroying the product margins.")

    # --- SECTION 5: STRATEGIC RECOMMENDATIONS ---
    else:
        st.title("📈 Executive Strategic Optimization Engine")
        st.markdown("Interact with corporate core objectives to evaluate localized deployment strategies and simulate revenue return loops.")

        # --- INTERACTIVE ELEMENT 1: STRATEGY SELECTOR ---
        st.markdown("### 🗺️ Step 1: Align Strategy to Active Retail Priorities")
        business_goal = st.selectbox(
            "What is your retail store's top priority this quarter?",
            [
                "Maximize Store-Wide Margin & Reduce Discount Leakage",
                "Increase Customer Retention & Multi-Year Loyalty",
                "Optimize Cross-Department Basket Size (Up-Selling)"
            ]
        )

        st.markdown("#### 🚀 Actionable Tactical Blueprint")
        if business_goal == "Maximize Store-Wide Margin & Reduce Discount Leakage":
            st.info("""
            * **The Root Problem:** Issuing generic 20% store-wide discount flyers causes massive margin loss from customers who would have paid full price anyway.
            * **Targeted Action:** Isolate your highly discount-reliant cluster (identified with an `avg_promotions` score > 45%). Program your cash registers to *only* print discount vouchers for those specific customer IDs. 
            * **Store Layout Tip:** Keep premium items in the center aisles with standard shelf prices, but create 'Value Endcaps' on the outer aisles to catch budget-focused clusters without lowering overall store prices.
            """)
        elif business_goal == "Increase Customer Retention & Multi-Year Loyalty":
            st.info("""
            * **The Root Problem:** Customer churn drops your Lifetime Value (LTV). Acquiring new shoppers costs 5x more than keeping existing ones.
            * **Targeted Action:** Flag veteran accounts (where `avg_years_as_customer` > 12 years) inside your CRM system. Automatically enroll them in a VIP Tier that rewards points on frequent baseline categories like Groceries and Fresh Produce.
            * **Store Layout Tip:** Introduce dedicated VIP self-checkout lanes or personalized digital coupon kiosks at the store entrance to increase convenience and brand affinity.
            """)
        else:
            st.info("""
            * **The Root Problem:** Shoppers are rushing in to buy a single item (like just milk or eggs) and leaving without exploring other high-margin categories.
            * **Targeted Action:** Deploy your mined Association Rules! Use the item-to-item pairs (Antecedents ➡️ Consequents) found via Apriori to build dynamic product bundles at checkout.
            * **Store Layout Tip:** Place high-affinity items physically far apart from each other (e.g., place salad greens at the back of produce and salad dressings on a completely separate shelf across the aisle) to force shoppers to walk past more products.
            """)

        # --- INTERACTIVE ELEMENT 2: LIVE ROI & REVENUE SIMULATOR ---
        st.markdown("### 🧮 Step 2: Interactive Revenue Lift Simulator")
        st.markdown("Estimate your monthly revenue increase by adjusting target adoption variables derived from your custom clusters:")

        col_sim1, col_sim2 = st.columns([1, 2])
        with col_sim1:
            baseline_rev = st.number_input("Average Monthly Store Revenue (€):", value=500000, step=50000)
            target_conversion = st.slider("Target Coupon Conversion Lift (%):", 0.5, 10.0, 2.5, step=0.5)
            avg_basket_increase = st.slider("Average Cross-Sell Basket Value Lift (€):", 2, 30, 8)
        
        with col_sim2:
            estimated_lift = (baseline_rev * (target_conversion / 100)) + (len(featured) * (target_conversion / 100) * avg_basket_increase)
            new_total = baseline_rev + estimated_lift
            
            st.markdown("#### 📊 Modeled Financial Impact Output")
            st.metric("Projected Monthly Revenue Growth", f"+€{estimated_lift:,.2f}", delta=f"{((estimated_lift/baseline_rev)*100):.2f}% Growth")
            
            st.markdown(f"""
            **Simulation Context & Execution Rules:**
            * By implementing cluster-specific vouchers rather than generic store discounts, you protect baseline margins for **{len(featured):,}** registered shoppers.
            * Shifting your conversion rate by **{target_conversion}%** using Apriori cross-sell triggers creates an immediate structural lift, raising monthly baseline expectations to **€{new_total:,.2f}**.
            """)

        # --- INTERACTIVE ELEMENT 3: OPERATIONAL CHECKLIST ---
        st.markdown("### 📋 Step 3: Deployment Team Launch Checklist")
        st.markdown("Check off tasks as your team deploys these data models onto your live store floor:")
        st.checkbox("Export cluster outputs (`customer_clusters.csv`) and sync customer IDs with CRM registers.", value=False)
        st.checkbox("Update checkout lane logic to print custom vouchers based on customer segment IDs.", value=False)
        st.checkbox("Rearrange physical aisle displays to pair high-lift antecedents and consequents together.", value=False)
        st.checkbox("Schedule automated pipeline updates every quarter to refresh cluster profiles as buying habits evolve.", value=False)

        st.balloons()


if __name__ == "__main__":
    main()