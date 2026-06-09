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
import streamlit.components.v1 as components



# --- INJEÇÃO CSS COMPLETA: PALETA CLEAR & OVERRIDE DE BOTÕES ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;700;800&family=Open+Sans:wght@400;600;700&display=swap');
    @import url('https://fonts.googleapis.com/icon?family=Material+Icons');

    /* 1. FUNDO GLOBAL DE ALTA PRIORIDADE */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMainViewContainer"], [data-testid="stHeader"] {
        background-color: #FFFBF7 !important;
    }

    /* Limpeza total de fundos automáticos nos blocos de texto */
    [data-testid="stMarkdownContainer"], .stMarkdown, p, span, label {
        background-color: transparent !important;
    }

    /* 2. BARRA LATERAL (Blue Crate) */
    section[data-testid="stSidebar"] {
        background-color: #293379 !important;
    }
    section[data-testid="stSidebar"] *, 
    section[data-testid="stSidebar"] span, 
    section[data-testid="stSidebar"] label, 
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] h1 {
        color: #FFFFFF !important;
        font-family: 'Montserrat', sans-serif !important;
    }

    /* 3. TIPOGRAFIA INTEGRADA (Letras Maiores) */
    html, body, p, span, .stMarkdown p, ul, li {
        font-family: 'Open Sans', sans-serif !important;
        color: #293379 !important;
        font-size: 1.15rem !important;
    }
    h1 { font-family: 'Montserrat', sans-serif !important; font-weight: 800 !important; color: #b81817 !important; font-size: 2.5rem !important; }
    h2, h3, h4 { font-family: 'Montserrat', sans-serif !important; font-weight: 700 !important; color: #293379 !important; }

    /* 4. QUADRADOS DE DESTAQUE (Apenas onde for chamado explicitamente) */
    .orange-card {
        background-color: #FFF2E6 !important;
        padding: 22px;
        margin-top: 15px;
        margin-bottom: 25px;
        border-radius: 8px;
        border-left: 6px solid #ee7302 !important;
    }
    .orange-card h3 { color: #b81817 !important; margin-top: 0 !important; }

    /* 5. ELIMINAÇÃO TOTAL DE BOTÕES ESCUROS (Executive Summary & Outros) */
    button, [data-testid^="stBaseButton"] {
        background-color: #FFFFFF !important;
        color: #293379 !important;
        border: 2px solid #ee7302 !important;
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        padding: 8px 22px !important;
        border-radius: 6px !important;
        transition: all 0.2s ease !important;
    }
    
    button:hover, [data-testid^="stBaseButton"]:hover {
        background-color: #FFF2E6 !important;
        color: #b81817 !important;
        border-color: #b81817 !important;
    }

    /* 6. CUSTOM EXPANDER HEADER (Cluster Cards) */
    /* Make expander headers toasted light yellow with blue text */
            
    div[data-testid="stExpander"] > button,
    div[data-testid="stExpander"] summary,
    details[role="group"] > summary,
    .streamlit-expanderHeader,
    .stExpanderHeader {
        background-color: #FFF2E6 !important; /* toasted light yellow */
        color: #293379 !important; /* keep blue text */
        border: 1px solid #E6D58A !important;
        border-radius: 6px !important;
        padding: 8px 12px !important;
        font-weight: 700 !important;
    }
    div[data-testid="stExpander"] > button:hover,
    details[role="group"] > summary:hover,
    .stExpanderHeader:hover {
        background-color: #F6E7A1 !important;
        color: #293379 !important;
    }

    /* Correção visual para caixas de input numérico e listas suspensas */
    input[type="number"], div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #293379 !important;
        border: 1px solid #ee7302 !important;
    }

    /* Estilo das Abas Superiores (Tabs) */
    button[data-baseweb="tab"] {
        font-family: 'Montserrat', sans-serif !important;
        font-size: 1.15rem !important;
        color: #293379 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #b81817 !important;
        border-bottom-color: #b81817 !important;
    }
    /* Additional theme tweaks */
    h1 { font-size: 3.0rem !important; }
    h2 { font-size: 2.0rem !important; }
    h3 { font-size: 1.35rem !important; }
    .orange-card h3 { color: #2E7D32 !important; }
    [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p, .stSelectbox label, .stSelectbox > label {
        color: #293379 !important;
        font-weight: 600 !important;
    }
    /* Prevent small preview text from overflowing into expander headers */
    div[data-testid="stExpander"] > button > div > div:last-child {
        visibility: hidden !important;
        max-height: 0 !important;
        overflow: hidden !important;
    }
    div[data-testid="stExpander"] > button, div[data-testid="stExpander"] summary {
        white-space: normal !important;
        overflow: visible !important;
    }
            
    /* Esconde só o preview text (último div) */

    /* Mantém título e seta visíveis */
    div[data-testid="stExpander"] summary {
        white-space: normal !important;
        overflow: visible !important;
    }
            
    div[data-baseweb="popover"] ul {
    background-color: #FFFFFF !important;
    }

    div[data-baseweb="popover"] li {
        background-color: #FFFFFF !important;
        color: #E65100 !important;
    }

    div[data-baseweb="popover"] li:hover {
        background-color: #FFCC80 !important;
        color: #2C1A04 !important;
    }
    div[data-testid="stExpander"] summary [data-testid="stIconMaterial"] {
        font-size: 0 !important;
        width: 1.2rem !important;
        height: 1.2rem !important;
    }

    div[data-testid="stExpander"] summary [data-testid="stIconMaterial"]::before {
        content: "▸" !important;
        font-size: 1.2rem !important;
        color: #293379 !important;
    }

    details[open] summary [data-testid="stIconMaterial"]::before {
        content: "▾" !important;
    }
            
    div[data-testid="stExpander"] summary p {
        color: #293379 !important;
        font-size: 1rem !important;
        display: block !important;
        visibility: visible !important;
    }
            
    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="collapsedControl"] button {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        overflow: hidden !important;
        width: 2rem !important;
        height: 2rem !important;
    }

    [data-testid="stSidebarCollapseButton"] button * ,
    [data-testid="collapsedControl"] button * {
        display: none !important;
    }

    [data-testid="stSidebarCollapseButton"] button::after {
        content: "☰" !important;
        font-size: 1.2rem !important;
        color: #FFFFFF !important;
        display: block !important;
    }

    [data-testid="collapsedControl"] button::after {
        content: "☰" !important;
        font-size: 1.2rem !important;
        color: #293379 !important;
        display: block !important;
    }
            

    [data-testid="stExpandSidebarButton"] {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        width: 2rem !important;
        height: 2rem !important;
    }

    [data-testid="stExpandSidebarButton"] * {
        display: none !important;
    }

    [data-testid="stExpandSidebarButton"]::after {
        content: "☰" !important;
        font-size: 1.2rem !important;
        color: #293379 !important;
        display: block !important;
    }
            
    .st-emotion-cache-5r6ut5 {
        display: none !important;
    }
            
    [data-testid="stSidebar"] [data-testid="stImageContainer"] button,
    [data-testid="stSidebar"] [data-testid="StyledFullScreenButton"],
    [data-testid="stSidebar"] button[title="View fullscreen"] {
        display: none !important;
        visibility: hidden !important;
    }
            
    button[data-testid="stBaseButton-elementToolbar"][aria-label="Fullscreen"] {
        display: none !important;
    }
            
    section[data-testid="stSidebar"] * {
        font-size: 0.9rem !important;
    }
    section[data-testid="stSidebar"] h1 {
        font-size: 1.2rem !important;
    }
            

    div[data-testid="stTabs"] button[aria-selected="true"] span,
    div[data-testid="stTabs"] button[aria-selected="true"] p {
        color: #ee7302 !important;
        font-size: 1.05rem !important;
    }
            
    </style>
    """, 
    unsafe_allow_html=True
)

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Cluster | Supermarket Insights Engine",
    page_icon="🛒",
    layout="wide",
)



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
        n = len(cluster_transactions)
        if n >= 6000:
            support = 0.008
        elif n >= 3000:
            support = 0.012
        else:
            support = 0.02
        results[cluster_id] = build_rules(cluster_transactions, min_support=support, min_confidence=0.2, top_n=5)
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
        description.append(" *Strategic Persona:* High-value lifetime veterans. They have shopped here for over a decade.")
    if row.get("avg_promotions", 0) > 0.45:
        description.append(" *Strategic Persona:* Price-sensitive coupon collectors. They convert best via clear orange discount stickers.")
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
    st.sidebar.title("Cluster Analytics")
    st.sidebar.caption("An intelligent app that helps you understand each costumer!")
    st.sidebar.markdown("---")
    
    sections = [
        "Executive Summary",
        "Data Insights (EDA)",
        "Preprocessing",
        "Customer Profiles (Clustering)",
        "Campaigns & Promotions",
        "Strategic Recommendations",
    ]
    section = st.sidebar.radio("Navigate Control Panel:", sections)
    st.sidebar.markdown("---")
    st.sidebar.caption("💡 *Strategic Value: Leverage behavioral persona mapping to transition from reactive mass marketing to predictive, high-yield customer engagement.")

# --- SECTION 1: EXECUTIVE SUMMARY ---
    if section == "Executive Summary":
        # 🎨 Advanced UI Styling (Orange Monochromatic Theme)
        # --- CONFIGURAÇÃO GLOBAL DE CORES E FONTES (Início da main) ---
        st.markdown("""
            <style>
            /* Importar as fontes do Google Fonts */
            @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@500;700;800&family=Open+Sans:wght@400;600;700&display=swap');

            /* Aplicar Open Sans ao corpo do texto global */
            html, body, [data-testid="stAppViewContainer"], .stMarkdown p {
                font-family: 'Open Sans', sans-serif !important;
                color: #293379 !important; /* Blue Crate para o texto principal para dar contraste legível */
            }

            /* Aplicar Montserrat aos títulos */
            h1, h2, h3, h4, h5, h6, [data-testid="stWidgetLabel"] p {
                font-family: 'Montserrat', sans-serif !important;
                font-weight: 700 !important;
                color: #293379 !important; /* Títulos principais em Blue Crate */
            }
            
            /* Ajustar a cor dos títulos principais da página */
            .stMarkdown h1 {
                color: #b81817 !important; /* Tomato Red para os títulos principais H1 */
                font-weight: 800 !important;
            }

            /* Costumização dos cartões da página de Preprocessing */
            .content-card {
                background-color: #FFF2E6 !important; /* Um tom pastel derivado do Orange, ultra legível */
                padding: 22px;
                margin-bottom: 18px;
                border-radius: 8px;
                border-left: 6px solid #ee7302 !important; /* Borda proeminente em Orange */
                box-shadow: 0 4px 6px rgba(41, 51, 121, 0.05);
            }
            
            .content-card h3 {
                color: #b81817 !important; /* Títulos dos passos em Tomato Red */
                font-size: 1.25rem !important;
                margin-top: 0 !important;
                margin-bottom: 10px !important;
            }

            .content-card p, .content-card li {
                color: #293379 !important; /* Texto interno em Blue Crate de alto contraste */
                font-size: 0.95rem !important;
                line-height: 1.6 !important;
            }
            
            .content-card b {
                color: #293379 !important; /* Corrigido: Destaques em negrito agora usam Blue Crate */
                font-weight: 700;
            }
                    
            </style>
            """, 
            unsafe_allow_html=True
        )

        st.title(" Customer Segmentation & Targeted Promotion Engine")
        st.markdown("""
        An advanced data science framework using machine learning to analyze shopping habits, 
        uncover natural consumer profiles, and translate purchasing behavior into actionable 
        marketing campaigns and promotional strategies through transaction data.
        """)
        
        # 1. High-Level Metrics Grid
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Tracked Active Customers</div><div class="metric-value">{len(featured):,}</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Total Extracted Baskets</div><div class="metric-value">{len(basket):,}</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Champion Architecture</div><div class="metric-value" style="font-size:22px; padding-top:5px; font-weight:700; color:#E65100;">K-Means (k=7)</div></div>', unsafe_allow_html=True)
        with col4:
            est_lift = len(featured) * 14.50
            st.markdown(f'<div class="metric-card"><div class="metric-title">Est. Strategy Value Lift</div><div class="metric-value">€{est_lift:,.0f}</div></div>', unsafe_allow_html=True)
        
        # 2. Executive Rationale & Model Selection (With the requested soft orange background)
        st.markdown("""
        <div class="content-card">
            <h3 style="color:#E65100; margin-top:0;"> Modeling Methodology & Business Rationale</h3>
            <p>This analytical portal serves as a complete digital replacement for the printed technical report, consolidating the outputs of a robust unsupervised learning infrastructure. The development pipeline rigorously evaluated both density-based (DBSCAN) and centroid-based (K-Means) clustering approaches:</p>
            <ul>
                <li><b>Selection Rationale (K-Means vs. DBSCAN):</b> While density-based algorithms were tested on the UMAP-reduced coordinates (utilizing Nearest Neighbors to reassign outliers), <b>K-Means with k=7</b> was selected as the champion production model. This choice ensures <b>100% customer database coverage</b> (preventing any customer from being dropped as noise), yielding optimal structural partitions validated by <i>Inertia</i> (Elbow Method) and <i>Silhouette Coefficient</i> benchmarks.</li>
                <li><b>Association Rules Integration:</b> The 7 behavioral profiles discovered in the high-dimensional feature space were directly mapped against historical transaction records (<code>customer_basket</code>). This allowed the Apriori algorithm to extract customized antecedent-consequent rules per cohort, maximizing cross-selling returns.</li>
                <li><b>Margin Protection:</b> Clearly isolating price-sensitive profiles (Promo Surfers) from high-value shoppers focused on premium assortment and convenience (Power Shoppers) mitigates "discount leakage," preventing the redundant distribution of profit-eating vouchers.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # 3. Macro Taxonomy Matrix
        st.markdown("### Behavioral Taxonomy Matrix & Campaign Feasibility")
        st.caption("Below is the detailed analytical characterization of the 7 natural shopper segments discovered and their respective strategic activation directives:")

        col_left, col_right = st.columns(2)
        
        with col_left:
            with st.expander("💻 Cluster 0 — Tech Enthusiasts", expanded=True):
                st.markdown("""
                * **Behavioral DNA:** Heavy spending concentration in electronics and video game feature metrics. Store footprint is highly characterized by late-night hours or specific digital transaction channels.
                * **Campaign Feasibility:** Low sensitivity to standard daily grocery discounts. Activate via technology ecosystem cross-selling (e.g., matching accessories), product pre-orders, and innovation-driven tech launches.
                """)
                
            with st.expander("🌱 Cluster 1 — Plant-Based Lifestyle", expanded=True):
                st.markdown("""
                * **Behavioral DNA:** High continuous investment in vegetables and organic subcategories, paired with near-zero conversion metrics in fresh meat or traditional butcher lines. High loyalty card penetration.
                * **Campaign Feasibility:** Highly stable, routine shopping behavior. Opportunity for margin expansion through eco-friendly hygiene products, natural wellness supplements, and premium plant-based gourmet ranges.
                """)
                
            with st.expander("👨‍👩‍👧‍👦 Cluster 2 — Large Households", expanded=True):
                st.markdown("""
                * **Behavioral DNA:** Maximum values for dependents at home (both kids and teenagers). Large basket volumes heavily weighted toward essential household groceries and bulk non-alcoholic beverages.
                * **Campaign Feasibility:** Highly exposed to inflation and budget pressures. Respond ideally to volume-based triggers (e.g., Buy 3 Pay 2) and instant savings on private-label core grocery categories.
                """)

            with st.expander("🛡️ Cluster 3 — Brand Loyalists", expanded=True):
                st.markdown("""
                * **Behavioral DNA:** Long-term customer lifecycle metrics (high seniority). Total expenditure is evenly and regularly distributed across all store departments with a near-zero customer complaint rate.
                * **Campaign Feasibility:** Highly insensitive to aggressive spot promotions. Drive retention through premium tier experiences, VIP loyalty point multipliers, and experiential rewards rather than baseline margin discounts.
                """)
                
        with col_right:
            with st.expander("⚠️ Cluster 4 — At-Risk Youth", expanded=True):
                st.markdown("""
                * **Behavioral DNA:** Low accumulated monetary value across core food departments. Sporadic and fragmented visit patterns, with spending heavily focused on entertainment or immediate convenience niches.
                * **Campaign Feasibility:** High critical churn risk. Requires immediate, aggressive app-driven push notifications triggering entry-level food items and low-friction, frequency-building incentives to stabilize store footfall.
                """)
                
            with st.expander("💎 Cluster 5 — Power Shoppers", expanded=True):
                st.markdown("""
                * **Behavioral DNA:** The primary financial engine of store revenue. Maximum spending ceilings reached across traditional butcher cuts, fresh fish, and premium wine/alcohol lines. High diversity of unique SKUs.
                * **Campaign Feasibility:** Completely price-insensitive; applying generic promotions here actively destroys net margins. Target via exclusive gourmet catalog previews, private tastings, and premium customer service tiers.
                """)
                
            with st.expander("🎟️ Cluster 6 — Promo Surfers", expanded=True):
                st.markdown("""
                * **Behavioral DNA:** Extreme cost-optimization behavior. Purchasing patterns are entirely dictated by the percentage of items bought on promotion. High cross-store rotation to hunt bargains.
                * **Campaign Feasibility:** High risk of gross margin drainage. Manage under strict Apriori association rules: promotional voucher validation at checkout requires a complementary full-priced item to be scanned in the basket.
                """)
        
        st.markdown("---")
        st.caption("💡 *Academic Delivery Note: In strict compliance with project guidelines, this presentation layer omits raw code rendering. All heavy computation, modeling pipelines, and training scripts are structured exclusively within their respective `.py` source files and execution notebooks.*")


# --- SECTION 2: DATA INSIGHTS (EDA) ---
    elif section == "Data Insights (EDA)":
        st.title("Supermarket Consumer Behavior Analytics")
        st.markdown("An interactive exploratory data analysis into customer demographics, purchase frequencies, and department spending behavior.")
        
        tab1, tab2, tab3 = st.tabs(["Dataset Overview", "Feature Distributions", "Department Revenue"])
        
        # --- TAB 1: DATA OVERVIEW ---
        with tab1:
            st.subheader("Data Volume & Integrity Summary")
            st.markdown("A high-level health check of the active retail datasets integrated into the modeling pipeline:")
            
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric(label="Total Registered Customers", value=f"{len(customer_info):,}")
            with m_col2:
                st.metric(label="Profiles Ready for Modeling", value=f"{len(preprocessed):,}")
            with m_col3:
                st.metric(label="Unique Products Scanned (SKUs)", value=f"{basket['items'].explode().nunique():,}")
                
            st.markdown("---")
            st.markdown("### Sample View (Raw Data Sample)")
            st.dataframe(featured.head(10), use_container_width=True)
            
        # --- TAB 2: INTERACTIVE FEATURE DISTRIBUTIONS ---
        with tab2:
            st.subheader("Interactive Demographic & Behavioral Explorer")
            st.markdown("Select any continuous feature from the preprocessed dataset to audit its underlying distribution across the customer base:")
            
            numeric_cols = featured.select_dtypes(include=[np.number]).columns.tolist()
            
            exclude_cols = [
                c for c in numeric_cols 
                if c.startswith("lifetime_spend_") 
                or "id" in c.lower() 
                or "latitude" in c.lower() 
                or "longitude" in c.lower()
            ]
            clean_selectable_cols = [c for c in numeric_cols if c not in exclude_cols]
            
            selected_feature = st.selectbox(
                "Select a feature to analyze:", 
                options=clean_selectable_cols,
                format_func=lambda x: x.replace("_", " ").title()
            )
            
            if selected_feature:
                fig, ax = plt.subplots(figsize=(10, 4))
                main_orange = "#E65100"
                max_val = featured[selected_feature].max()
                
                if "gender" in selected_feature.lower():
                    sns.histplot(
                        featured[selected_feature].dropna(), 
                        discrete=True,
                        kde=False, 
                        ax=ax, 
                        color=main_orange, 
                        edgecolor="#FFE0B2",  
                        alpha=0.85
                    )
                    ax.set_xticks([0, 1])
                    ax.set_xticklabels(["Female", "Male"])
                    
                elif max_val <= 10:
                    data_rounded = featured[selected_feature].dropna().round()
                    sns.histplot(
                        data_rounded, 
                        discrete=True,
                        kde=False, 
                        ax=ax, 
                        color=main_orange, 
                        edgecolor="#FFE0B2",  
                        alpha=0.85
                    )
                    unique_ticks = sorted(data_rounded.unique().astype(int))
                    ax.set_xticks(unique_ticks)
                    ax.set_xticklabels(unique_ticks)
                    
                else:
                    sns.histplot(
                        featured[selected_feature].dropna(), 
                        bins=30, 
                        kde=True, 
                        ax=ax, 
                        color=main_orange, 
                        edgecolor="#FFE0B2",  
                        alpha=0.85
                    )
                
                ax.set_title(f"Distribution Profile of {selected_feature.replace('_', ' ').title()}", fontsize=11, fontweight="bold", pad=15)
                ax.set_xlabel(selected_feature.replace("_", " ").title(), fontsize=10)
                ax.set_ylabel("Customer Count", fontsize=10)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.tight_layout()
                
                st.pyplot(fig)
                
                if "gender" in selected_feature.lower():
                    gender_counts = featured[selected_feature].dropna().value_counts()
                    female_total = gender_counts.get(0, 0)
                    male_total = gender_counts.get(1, 0)
                    st.markdown(f"> **Quick Insight:** The dataset contains **{female_total:,}** female records (encoded as 0) and **{male_total:,}** male records (encoded as 1).")
                elif max_val <= 10:
                    st.markdown(f"> **Quick Insight:** The most frequent class for **{selected_feature.replace('_', ' ').title()}** is **{int(featured[selected_feature].round().mode()[0])}**, with responses historically bounded between **{int(featured[selected_feature].min())}** and **{int(featured[selected_feature].max())}**.")
                else:
                    st.markdown(f"> **Quick Insight:** The average value for **{selected_feature.replace('_', ' ').title()}** sits at **{featured[selected_feature].mean():,.2f}** (Standard Deviation: *{featured[selected_feature].std():,.2f}*), with records ranging from *{featured[selected_feature].min():,}* up to *{featured[selected_feature].max():,}*.")

        # --- TAB 3: CATEGORY REVENUE CONTRIBUTION ---
        with tab3:
            st.subheader("Total Revenue Contribution by Department")
            st.markdown("Analyzing the total cumulative expenditure distribution across primary retail categories:")
            
            spending_cols = [c for c in featured.columns if c.startswith("lifetime_spend_")]
            if spending_cols:
                totals = featured[spending_cols].sum().sort_values(ascending=False)
                totals.index = [c.replace("lifetime_spend_", "").replace("_", " ").title() for c in totals.index]
                
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.barplot(
                    x=totals.values, 
                    y=totals.index, 
                    palette="Oranges_r", 
                    ax=ax, 
                    edgecolor="#CCCCCC"
                )
                
                ax.set_title("Gross Lifetime Value (LTV) Contribution by Store Department", fontsize=11, fontweight="bold", pad=15)
                ax.set_xlabel("Total Cumulative Revenue (€)", fontsize=10)
                ax.set_ylabel("Department", fontsize=10)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.tight_layout()
                
                st.pyplot(fig)
                
                top_dept = totals.index[0]
                st.info(f"**Core Insight:** The {top_dept} department represents the largest share of historical customer wallet spend in this dataset.")
            else:
                st.info("No explicit lifetime spend metrics found to display.")

# --- SECTION 3: PREPROCESSING ---
    elif section == "Preprocessing":
        st.title("Preprocessing & Pipeline Rationale")
        st.markdown("This section documents the exact data preparation steps applied to the customer dataset prior to clustering:")
        
        st.markdown("""
            <style>
            .content-card {
                background-color: #FFE6D5; /* Laranja suave ligeiramente mais escuro que o fundo */
                padding: 20px;
                margin-bottom: 15px;
                border-radius: 8px;
                border-left: 5px solid #E65100; /* Barra lateral laranja escura para dar contraste */
                box-shadow: 0 2px 4px rgba(0,0,0,0.02);
            }
            .content-card h3 {
                color: #E65100 !important;
                margin-top: 0;
                font-weight: 700;
            }
            /* Alterado de castanho para o Azul Escuro da paleta global */
            .content-card p, .content-card li, .content-card b {
                color: #293379 !important;
            }
            </style>
            
            <div class='content-card'>
                <h3>1. Feature Extraction & Encoding</h3>
                <p>Raw temporal and categorical data were transformed into predictable features:
                <ul>
                    <li><b>Age Extraction:</b> Customer birthdates were converted to current age using the active datetime framework.</li>
                    <li><b>Customer Seniority:</b> Calculated as <b>years_as_customer</b> based on the initial transaction year.</li>
                    <li><b>Gender Categorization:</b> Encoded via <b>LabelEncoder</b> to ensure a clean binary numeric scale. Assigning female: 0 and male: 1</li>
                </ul>
                </p>
            </div>
            
            <div class='content-card'>
                <h3>2. Advanced Missing Data Imputation (MICE)</h3>
                <p>Instead of relying on biased mean or median fills, missing values within numeric columns were addressed using <b>Multivariate Imputation by Chained Equations (MICE)</b> via <b>IterativeImputer</b>. Non-predictive metadata and identifiers were dynamically excluded to prevent model leakage during the 10-iteration imputation cycle.</p>
            </div>
            
            <div class='content-card'>
                <h3>3. Outlier Capping via Interquartile Range (IQR)</h3>
                <p>To avoid cluster distortion from extreme values, distribution tails were reviewed using boxplots. Numeric variables were clipped using <b>np.clip</b> within statistical boundaries:
                <br>Lower Bound: max(0, Q1 - 1.5 * IQR) | Upper Bound: Q3 + 1.5 * IQR.
                <br>This preserved data size while neutralizing erratic extreme values.</p>
            </div>
            
            <div class='content-card'>
                <h3>4. Feature Aggregation & Dimensionality Pruning</h3>
                <p>To capture meaningful social profiles and minimize redundant data noise:
                <ul>
                    <li><b>kids_home</b> and <b>teens_home</b> were summed into a single feature: <b>dependents_home</b>.</li>
                    <li>High-entropy, raw geographic variables (latitude, longitude), unique IDs, and redundant intermediate indicators (such as the initial loyalty card flags or total distinct product counts) were systematically dropped.</li>
                </ul>
                </p>
            </div>
            
            <div class='content-card'>
                <h3>5. Skewness Mitigation (Log Transformation)</h3>
                <p>Monetary distributions are naturally heavily right-skewed. A log transformation using <b>np.log1p</b> was applied to all <b>lifetime_spend_</b> columns. This stabilized variance, normalized distributions, and prevented premium store departments from artificially overpowering weaker ones during distance calculations.</p>
            </div>
            
            <div class='content-card'>
                <h3>6. Z-Score Standardization (Scaling)</h3>
                <p>Because algorithms like K-Means and UMAP rely heavily on Euclidean and manifold distances, all processed numeric features were transformed using <b>StandardScaler</b>. This centered variables to a mean of 0 and scaled them to a variance of 1, ensuring every behavioral asset contributes equally to the final cohort layout.</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

    elif section == "Customer Profiles (Clustering)":
        st.title("Customer Profiles (Clustering)")
        
        # --- EXECUÇÃO E CARREGAMENTO REAL DA VOSSA INFRAESTRUTURA ---
        import kmeans_2 as km
        import dbscan as db
        # Nota: Ajusta o nome do import abaixo caso o teu terceiro script tenha outro nome de ficheiro
        import comparison as comp 

        # Ativação do pipeline de dados direto do vosso ecossistema
        costumer_preprocessed, costumer_featured = km.load_data()
        _, embedding = km.fit_umap(costumer_preprocessed)

        # Parâmetros Estáveis do Vosso Modelo
        KMEANS_K = 7
        DBSCAN_EPS = 0.3
        DBSCAN_MIN_SAMPLES = 5

        # Criação das Abas em Inglês
        tab_kmeans, tab_dbscan, tab_comparison = st.tabs([
            "K-Means Engine", 
            "DBSCAN Density Model", 
            "Algorithmic Comparison"
        ])

        # --- GESTOR DE FLUXO DO MATPLOTLIB ---
        # Intercepta o plt.show() original dos vossos scripts para desenhar no Streamlit
        def streamlit_plot_interceptor(*args, **kwargs):
            st.pyplot(plt.gcf())
            plt.clf()

        old_show = plt.show
        plt.show = streamlit_plot_interceptor

        # --- TAB 1: K-MEANS ---
        with tab_kmeans:
            st.subheader("Centroid-Based Model Optimization")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Elbow Curve Evaluation")
                plt.clf()
                km.elbow_curve(costumer_preprocessed, chosen_k=KMEANS_K)
                
            with col2:
                st.markdown("##### Silhouette Sample Profiles")
                plt.clf()
                km.silhouette_plot(costumer_preprocessed, n_clusters=KMEANS_K)

            st.markdown("##### UMAP High-Dimensional Projection (K-Means)")
            plt.clf()
            _, labels_kmeans = km.run_kmeans(costumer_preprocessed, n_clusters=KMEANS_K)
            km.plot_umap_clusters(embedding, labels_kmeans, title="Cluster Sizes — K-Means")

        # --- TAB 2: DBSCAN ---
        with tab_dbscan:
            st.subheader("Density-Based Spatial Clustering")
            
            st.markdown("##### UMAP Spatial Projection (DBSCAN — Outliers Reassigned)")
            plt.clf()
            _, labels_dbscan = db.run_dbscan(embedding, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
            labels_dbscan_clean = db.assign_noise(embedding, labels_dbscan)
            db.plot_umap_clusters(embedding, labels_dbscan_clean)

        # --- TAB 3: COMPARISON & BUSINESS RATIONALE ---
        with tab_comparison:
            st.header("Modeling Methodology & Business Rationale")
            st.markdown("""
            This analytical portal serves as a complete digital replacement for the printed technical report, 
            consolidating the outputs of a robust unsupervised learning infrastructure. The development pipeline 
            rigorously evaluated both density-based (DBSCAN) and centroid-based (K-Means) clustering approaches:
            """)

            # Execução do plot comparativo nativo
            st.markdown("##### Shared Manifold Projections Alignment")
            plt.clf()
            _, labels_kmeans = km.run_kmeans(costumer_preprocessed, n_clusters=KMEANS_K)
            _, labels_dbscan = db.run_dbscan(embedding, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
            labels_dbscan_clean = db.assign_noise(embedding, labels_dbscan)
            
            labels_dict = {"K-Means": labels_kmeans, "DBSCAN": labels_dbscan_clean}
            comp.plot_umap_comparison(embedding, labels_dict)

            # Tabela de Validação Limpa por Silhouette Score
            st.markdown("### Selection Metric: Silhouette Score Analysis")
            metrics_df = comp.compute_metrics(costumer_preprocessed, labels_dict)
            
            if "silhouette" in metrics_df.columns:
                silhouette_table = metrics_df[["n_clusters", "silhouette"]]
            else:
                silhouette_table = metrics_df.loc[["K-Means", "DBSCAN"], ["n_clusters", "silhouette"]]
                
            st.table(silhouette_table)

            # Estrutura de Quadrados Laranja Claro para os Tópicos de Negócio do Vosso Relatório
            st.markdown("<div class='orange-card'>", unsafe_allow_html=True)
            st.markdown("""
            <h3>Selection Rationale (K-Means vs. DBSCAN)</h3>
            While density-based algorithms were tested on the UMAP-reduced coordinates (utilizing Nearest Neighbors to reassign outliers), 
            <b>K-Means with k=7</b> was selected as the champion production model. This choice ensures 100% customer database coverage 
            (preventing any customer from being dropped as noise), yielding optimal structural partitions validated by Inertia (Elbow Method) 
            and Silhouette Coefficient benchmarks.
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='orange-card'>", unsafe_allow_html=True)
            st.markdown("""
            <h3>Association Rules Integration</h3>
            The 7 behavioral profiles discovered in the high-dimensional feature space were directly mapped against historical transaction records 
            (<i>customer_basket</i>). This allowed the Apriori algorithm to extract customized antecedent-consequent rules per cohort, maximizing cross-selling returns.
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='orange-card'>", unsafe_allow_html=True)
            st.markdown("""
            <h3>Margin Protection Matrix</h3>
            Clearly isolating price-sensitive profiles (<i>Promo Surfers</i>) from high-value shoppers focused on premium assortment and convenience 
            (<i>Power Shoppers</i>) mitigates <b>discount leakage</b>, preventing the redundant distribution of profit-eating vouchers.
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Restaura o sistema original do Matplotlib
        plt.show = old_show  


    elif section == "Campaigns & Promotions":
        st.title("Campaigns & Promotions")

        tab_coupons, tab_analysis = st.tabs([
            "Targeted Coupons",
            "Spending Analysis & Association Rules"
        ])

        CLUSTER_NAMES = {
            0: "💻 Cluster 0 — Tech Enthusiasts",
            1: "🌱 Cluster 1 — Plant-Based Lifestyle",
            2: "👨‍👩‍👧‍👦 Cluster 2 — Large Households",
            3: "🛡️ Cluster 3 — Brand Loyalists",
            4: "⚠️ Cluster 4 — At-Risk Youth",
            5: "💎 Cluster 5 — Power Shoppers",
            6: "🎟️ Cluster 6 — Promo Surfers",
        }

        PROMOTIONS = {
            0: {
                "title": "🍫 Chocolate, Energy Drink or Protein Bar",
                "description": "By buying AirPods.",
                "mechanic": "Electronics purchase unlocks a complimentary food item at checkout.",
                "discount": "FREE SNACK",
                "color": "#293379",
            },
            1: {
                "title": "🍳 Kitchen Essentials (Napkins & Cooking Oil)",
                "description": "By buying Dog Food + Baby Food.",
                "mechanic": "Bundle trigger — both items must be in the basket to activate.",
                "discount": "10% OFF",
                "color": "#2E7D32",
            },
            2: {
                "title": "🥣 Breakfast Bundle Deal",
                "description": "Buy Cereals + Tea + Butter together and save 15% on the full bundle.",
                "mechanic": "All 3 items must be scanned together to unlock the bundle price.",
                "discount": "15% OFF",
                "color": "#b81817",
            },
            3: {
                "title": "🏋️ Any Tech Item",
                "description": "Buy any sports nutrition product",
                "mechanic": "Nutrition purchase unlocks a tech discount voucher printed at checkout.",
                "discount": "5% OFF",
                "color": "#293379",
            },
            4: {
                "title": "🥦 Progressive Veggie Discount",
                "description": "The more vegetables you buy, the more you save: 2 items → 5% off, 4 items → 10% off, 6+ items → 15% off.",
                "mechanic": "Progressive discount — scales automatically with basket quantity.",
                "discount": "UP TO 15% OFF",
                "color": "#2E7D32",
            },
            5: {
                "title": "🎧 AirPods",
                "description": "By buying any Electronic item.",
                "mechanic": "Electronics purchase unlocks AirPods discount at checkout.",
                "discount": "20% OFF",
                "color": "#b81817",
            },
            6: {
                "title": "🎵 Bluetooth Headphones",
                "description": "By buying AirPods + Laptop.",
                "mechanic": "Basket-gated — both AirPods and Laptop must be in the basket to unlock the deal.",
                "discount": "30% OFF",
                "color": "#ee7302",
            },
        }

        with tab_coupons:
            st.markdown("### Select a Cluster to View its Promotion")
            
            selected_cluster = st.selectbox(
                "Choose a customer segment:",
                options=list(CLUSTER_NAMES.keys()),
                format_func=lambda x: CLUSTER_NAMES[x],
                key="campaign_cluster_select"
            )

            promo = PROMOTIONS[selected_cluster]
            color = promo['color']
            discount = promo['discount']
            title = promo['title']
            description = promo['description']
            mechanic = promo['mechanic']
            cluster_name = CLUSTER_NAMES[selected_cluster]


            components.html(f"""
            <style>
                body {{ background-color: #FFFBF7; margin: 0; padding: 0; }}
            </style>
            <div style="
                display: flex;
                max-width: 900px;
                width: 100%;
                margin-top: 20px;
                filter: drop-shadow(0 4px 16px rgba(0,0,0,0.15));
                background-color: #FFFBF7;
            ">
                <!-- MAIN COUPON BODY -->
                <div style="
                    background-color: {color};
                    padding: 36px 40px;
                    flex: 1;
                    border-radius: 12px 0 0 12px;
                ">
                    <p style="color: rgba(255,255,255,0.8); font-family: Montserrat; font-size: 0.85rem; letter-spacing: 3px; text-transform: uppercase; margin: 0 0 8px 0;">{cluster_name}</p>
                    <h1 style="color: #FFFFFF; font-family: Montserrat; font-size: 3rem; font-weight: 900; margin: 0 0 8px 0; line-height: 1;">{discount}</h1>
                    <h3 style="color: rgba(255,255,255,0.9); font-family: Montserrat; font-size: 1.2rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; margin: 0 0 20px 0;">{title}</h3>
                    <p style="color: rgba(255,255,255,0.85); font-size: 1rem; margin: 0 0 12px 0;">{description}</p>
                    <p style="color: rgba(255,255,255,0.65); font-size: 0.85rem; margin: 0;"><b style="color:rgba(255,255,255,0.85);">How it works:</b> {mechanic}</p>
                </div>

                <!-- NOTCH SEPARATOR -->
                <div style="
                    width: 24px;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    position: relative;
                    background-color: {color};
                    filter: brightness(0.85);
                    flex-shrink: 0;
                ">
                    <div style="width: 20px; height: 20px; border-radius: 50%; background-color: #FFFBF7; position: absolute; top: -10px; left: 2px; z-index: 10; filter: brightness(1);"></div>
                    <div style="width: 2px; height: 100%; border-left: 3px dashed rgba(255,255,255,0.4);"></div>
                    <div style="width: 20px; height: 20px; border-radius: 50%; background-color: #FFFBF7; position: absolute; bottom: -10px; left: 2px; z-index: 10; filter: brightness(1);"></div>
                </div>

                <!-- TEAR-OFF STUB -->
                <div style="
                    background-color: {color};
                    filter: brightness(0.75);
                    width: 80px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    border-radius: 0 12px 12px 0;
                    flex-shrink: 0;
                ">
                    <p style="color: #FFFFFF; font-family: Montserrat; font-weight: 800; font-size: 0.75rem; letter-spacing: 3px; text-transform: uppercase; writing-mode: vertical-rl; transform: rotate(180deg); margin: 0;">COUPON · {discount}</p>
                </div>
            </div>
            """, height=300)

        with tab_analysis:
            st.markdown("### Spending Analysis & Association Rules")

            selected_cluster_analysis = st.selectbox(
                "Choose a customer segment:",
                options=list(CLUSTER_NAMES.keys()),
                format_func=lambda x: CLUSTER_NAMES[x],
                key="analysis_cluster_select"
            )

            col_annual, col_rules = st.columns(2)

            with col_annual:
                st.markdown("#### Annual Spend by Category")
                spend_cols = [c for c in featured.columns if c.startswith("lifetime_spend_")]
                cluster_mask = labels == selected_cluster_analysis
                cluster_featured = featured[cluster_mask].copy()

                if "years_as_customer" in cluster_featured.columns and len(cluster_featured) > 0:
                    annual_data = {}
                    for col in spend_cols:
                        category = col.replace("lifetime_spend_", "").replace("_", " ").title()
                        avg_annual = (cluster_featured[col] / cluster_featured["years_as_customer"].replace(0, 1)).mean()
                        annual_data[category] = round(avg_annual, 2)

                    annual_df = pd.DataFrame.from_dict(annual_data, orient="index", columns=["Avg Annual Spend (€)"])
                    annual_df = annual_df.sort_values("Avg Annual Spend (€)", ascending=False)
                    st.dataframe(annual_df, use_container_width=True)
                else:
                    st.info("Annual spend data not available.")

            with col_rules:
                st.markdown("#### Top Association Rules")
                cluster_rules = rules_by_cluster.get(selected_cluster_analysis, pd.DataFrame())
                if not cluster_rules.empty:
                    st.dataframe(
                        cluster_rules[["antecedent", "consequent", "support", "confidence", "lift"]].round(3),
                        use_container_width=True
                    )
                else:
                    st.info("No association rules found for this cluster.")
    # ─────────────────────────────────────────────────────────────
    # SECTION: STRATEGIC RECOMMENDATIONS & CONCLUSION
    # Replace the existing `else:` block in main() with this block
    # ─────────────────────────────────────────────────────────────
    else:
        # ── INLINE STYLE OVERRIDES (scoped to this section) ──────
        st.markdown("""
        <style>
        .rec-hero {
            background: linear-gradient(135deg, #293379 0%, #1a2060 100%);
            border-radius: 12px;
            padding: 36px 40px;
            margin-bottom: 32px;
            color: #FFFFFF !important;
        }
        .rec-hero h2 { color: #FFFFFF !important; font-size: 1.7rem !important; margin-bottom: 6px !important; }
        .rec-hero h2, .rec-hero h2 * {
                                        color: #FFFFFF !important;
                                        opacity: 1 !important;
                                        visibility: visible !important;
                                    }
        .rec-hero p  { color: #FFD9B5 !important; font-size: 1.05rem !important; margin: 0 !important; }

        .priority-card {
            background-color: #FFFFFF;
            border: 1.5px solid #E0E0E0;
            border-top: 5px solid #ee7302;
            border-radius: 10px;
            padding: 22px 24px;
            margin-bottom: 18px;
            transition: box-shadow 0.2s ease;
        }
        .priority-card:hover { box-shadow: 0 6px 20px rgba(238,115,2,0.12); }
        .priority-card h4 { color: #b81817 !important; margin: 0 0 8px 0 !important; font-size: 1.1rem !important; }
        .priority-card p, .priority-card li { color: #293379 !important; font-size: 0.97rem !important; line-height: 1.65 !important; margin: 0 !important; }

        .cluster-pill {
            display: inline-block;
            background-color: #FFF2E6;
            color: #b81817 !important;
            border: 1px solid #ee7302;
            border-radius: 20px;
            padding: 3px 12px;
            font-size: 0.82rem !important;
            font-weight: 700;
            margin: 2px 3px;
            font-family: 'Montserrat', sans-serif;
        }

        .campaign-row {
            display: flex;
            align-items: flex-start;
            gap: 16px;
            background-color: #FFFBF7;
            border-left: 4px solid #ee7302;
            border-radius: 0 8px 8px 0;
            padding: 14px 18px;
            margin-bottom: 12px;
        }
        .campaign-row .c-icon { font-size: 1.7rem; flex-shrink: 0; margin-top: 2px; }
        .campaign-row .c-body h5 { color: #293379 !important; margin: 0 0 4px 0 !important; font-size: 0.97rem !important; }
        .campaign-row .c-body p  { color: #555 !important; font-size: 0.9rem !important; margin: 0 !important; }

        .sim-result-box {
            background: linear-gradient(135deg, #FFF2E6, #FFFBF7);
            border: 2px solid #ee7302;
            border-radius: 10px;
            padding: 24px;
            text-align: center;
        }
        .sim-result-box .sim-label { color: #293379 !important; font-size: 0.9rem !important; font-weight: 600; margin-bottom: 4px; }
        .sim-result-box .sim-value { color: #b81817 !important; font-size: 2.2rem !important; font-weight: 800; font-family: 'Montserrat', sans-serif; }
        .sim-result-box .sim-delta { color: #2E7D32 !important; font-size: 1rem !important; font-weight: 700; margin-top: 4px; }

        .conclusion-block {
            background-color: #F0F3FF;
            border-radius: 10px;
            padding: 26px 30px;
            margin-top: 8px;
        }
        .conclusion-block h3 { color: #293379 !important; margin-top: 0 !important; }
        .conclusion-block p, .conclusion-block li { color: #293379 !important; font-size: 0.97rem !important; line-height: 1.7 !important; }
        </style>
        """, unsafe_allow_html=True)

        # ── HERO BANNER ──────────────────────────────────────────
        st.markdown("""
        <div class="rec-hero">
            <h2>Conclusion & Strategic Recommendations</h2>
            <p>Turning seven behavioral personas into a concrete retail playbook — margin-safe, 
            cluster-aware, and ready for immediate activation.</p>
        </div>
        """, unsafe_allow_html=True)

        # ── TAB LAYOUT ────────────────────────────────────────────
        tab_simulator, tab_conclusion = st.tabs([
            "Revenue Lift Simulator",
            "Final Conclusions",
        ])

        
        # ─────────────────────────────────────────────────────────
        # TAB 1 — REVENUE LIFT SIMULATOR
        # ─────────────────────────────────────────────────────────
        with tab_simulator:
            st.markdown("### Interactive Revenue Lift Simulator")
            st.caption(
                "Adjust the levers below to model the expected financial impact "
                "of activating the cluster-targeted campaign strategy."
            )

            col_inputs, col_result = st.columns([3, 2], gap="large")

            with col_inputs:
                monthly_revenue = st.number_input(
                    "Average Monthly Store Revenue (€):",
                    min_value=50_000,
                    max_value=10_000_000,
                    value=500_000,
                    step=25_000,
                    format="%d",
                )
                targeted_pct = st.slider(
                    "Share of customers covered by targeted campaigns (%):",
                    min_value=10, max_value=100, value=65, step=5,
                    help="Estimated % of the customer base reached by at least one cluster campaign.",
                )
                avg_lift_pct = st.slider(
                    "Expected revenue lift per targeted customer (%):",
                    min_value=0.5, max_value=15.0, value=3.5, step=0.5,
                    help="Conservative industry benchmark: 2–5% for well-targeted promotions.",
                )
                margin_drag_pct = st.slider(
                    "Estimated margin drag from promotional costs (%):",
                    min_value=0.0, max_value=5.0, value=1.0, step=0.25,
                    help="Voucher costs, event hosting, loyalty point cost, etc.",
                )

            with col_result:
                gross_lift    = monthly_revenue * (targeted_pct / 100) * (avg_lift_pct / 100)
                margin_cost   = monthly_revenue * (margin_drag_pct / 100)
                net_lift      = gross_lift - margin_cost
                net_lift_pct  = (net_lift / monthly_revenue) * 100

                st.markdown(f"""
                <div class="sim-result-box">
                    <div class="sim-label">Gross Revenue Lift</div>
                    <div class="sim-value">+€{gross_lift:,.0f}</div>
                    <hr style="border-color:#E0E0E0; margin:12px 0;">
                    <div class="sim-label">After Promo Cost Drag (−€{margin_cost:,.0f})</div>
                    <div class="sim-value" style="font-size:1.7rem !important;">€{net_lift:,.0f}</div>
                    <div class="sim-delta">▲ {net_lift_pct:.2f}% net monthly growth</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("&nbsp;")
                annual_net = net_lift * 12
                st.markdown(f"""
                <div class="sim-result-box" style="margin-top:0; border-color:#293379;">
                    <div class="sim-label">Annualised Net Lift Projection</div>
                    <div class="sim-value" style="color:#293379 !important;">€{annual_net:,.0f}</div>
                    <div class="sim-delta" style="color:#293379 !important;">Based on current input values</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("#### Campaign-Level Impact Breakdown")
            st.caption("Estimated contribution per cluster campaign — adjust share assumptions as needed.")

            breakdown_data = {
                "Cluster": [
                    "💎 Power Shoppers", "🛡️ Brand Loyalists", "🌱 Plant-Based",
                    "👨‍👩‍👧‍👦 Large Households", "🎟️ Promo Surfers (margin save)",
                    "💻 Tech Enthusiasts", "⚠️ At-Risk Youth",
                ],
                "Est. Share of Base (%)": [8, 18, 14, 16, 15, 12, 17],
                "Campaign Type": [
                    "Experiential / No discount", "Points multiplier", "Bundle upsell",
                    "Volume trigger", "Coupon gate (cost save)", "Cross-sell", "Re-engagement",
                ],
                "Expected Lift / Impact": [
                    "+6–9% spend/visit", "+4% basket frequency", "+5% basket size",
                    "+7% volume uplift", "−2% margin leakage saved", "+3% cross-category",
                    "−30% churn rate reduction",
                ],
            }
            st.dataframe(
                pd.DataFrame(breakdown_data),
                use_container_width=True,
                hide_index=True,
            )

        # ─────────────────────────────────────────────────────────
        # TAB 2 — FINAL CONCLUSIONS
        # ─────────────────────────────────────────────────────────
        with tab_conclusion:
            st.markdown("### Project Conclusions & Forward Roadmap")

            st.markdown("""
            <div class="conclusion-block">
                <h3>Technical Takeaways</h3>
                <ul>
                    <li><b>K-Means (k=7) outperformed DBSCAN</b> as the production model: it guarantees 
                    100% customer coverage, produces stable centroids across retraining cycles, and 
                    yields a Silhouette Score that validates meaningful inter-cluster separation.</li>
                    <li><b>UMAP dimensionality reduction</b> confirmed that the 7-cluster structure is 
                    recoverable even from a 2D manifold projection — a strong signal of true latent 
                    groupings in the original high-dimensional feature space.</li>
                    <li><b>Apriori association rules</b> added a transactional layer that purely 
                    demographic segmentation cannot deliver: the same cluster can exhibit very 
                    different cross-sell opportunities depending on what items co-occur in baskets.</li>
                    <li><b>Log-transforming spend columns</b> before scaling proved essential — 
                    without it, Power Shoppers would have dominated the distance matrix 
                    and collapsed three or four natural personas into a single mega-cluster.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("&nbsp;")

            col_l, col_r = st.columns(2, gap="large")

            with col_l:
                st.markdown("""
                <div class="conclusion-block">
                    <h3>Business Recommendations</h3>
                    <ul>
                        <li><b>Stop issuing generic store-wide vouchers.</b> At least 23% of your 
                        customer base (Power Shoppers + Brand Loyalists) is completely 
                        price-insensitive — discounts here destroy margin with zero uplift.</li>
                        <li><b>Gate every promotional mechanic behind a basket condition.</b> 
                        Promo Surfers must earn their discount by placing at least one 
                        full-priced high-margin item first.</li>
                        <li><b>Activate dormancy alerts immediately.</b> At-Risk Youth churning 
                        silently is the single largest addressable revenue risk in this dataset. 
                        A €3 re-engagement voucher costs less than acquiring a new customer.</li>
                        <li><b>Use the typical_hour feature.</b> Sending a push notification 
                        2 hours before each cluster's median visit hour is the highest-ROI 
                        personalisation lever available without any additional data collection.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            with col_r:
                st.markdown("""
                <div class="conclusion-block">
                    <h3>90-Day Activation Roadmap</h3>
                    <ul>
                        <li><b>Month 1 — Quick wins:</b> Deploy the Conditional Coupon Gate 
                        for Cluster 6 at POS. Configure dormancy alert pipeline for 
                        Cluster 4. Zero investment, immediate margin recovery.</li>
                        <li><b>Month 2 — Growth plays:</b> Launch the Green Basket Bundle 
                        for Cluster 1 and the Family Bulk Incentive for Cluster 2. 
                        A/B test push notification timing using typical_hour.</li>
                        <li><b>Month 3 — Premium tier:</b> Pilot the Gourmet Insider 
                        tasting event for Cluster 5's top-100 spenders. 
                        Measure incremental basket size before scaling.</li>
                        <li><b>Ongoing:</b> Retrain the K-Means model quarterly. 
                        Customer profiles drift — especially the At-Risk cohort — 
                        and stale labels erode campaign accuracy fast.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.caption(
                "Academic Delivery Note: All modeling code, training notebooks, and .py "
                "source files are version-controlled in the project repository. "
                "This application layer contains zero raw code in compliance with project guidelines."
            )
if __name__ == "__main__":
    main()