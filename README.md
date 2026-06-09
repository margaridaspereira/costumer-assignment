# Costumer Assignment
We have been provided two datasets containing information on customer demographics, spending habits, purchasing behavior and historical transitions of the customer.
Our task was to perform customer segmentation and identify distinct groups of customers based on their shared characteristics. The two datasets are named - customer_info and customer_basket.

# Customer Segmentation & Targeted Promotion Engine

## How to Run

1. Install dependencies:

pip install -r requirements.txt

2. Run the app:

streamlit run app.py

To ensure all text and visualization contrasts render correctly, please click the three dots (menu) in the top right corner of the webpage, and choose Light mode.

**Note on Execution:**
As this project is developed entirely using standard Python scripts (.py files) rather than Jupyter Notebooks (except for the EDA phase), the code must be executed locally to generate and view the analytical outputs.

## Project Structure
- `app.py` — Main Streamlit application
- `eda.ipynb` — Exploratory Data Analysis and Preprocessing notebook
- `comparison.py` — Algorithm comparison (K-Means, DBSCAN, Hierarchical)
- `kmeans_2.py` — K-Means clustering
- `dbscan.py` — DBSCAN clustering
- `hierarchical.py` — Hierarchical clustering
- `customer_utils.py` — Shared utility functions
- `association_rules.py` — Association rules analysis
- `cluster_assignments.csv` — Final cluster assignments for all 33,038 customers
- `cluster_profile.csv` — Cluster mean profiles by segment

## Data Files Required
Place the following files in the root directory before running:
- `customer_info.csv`
- `customer_basket.csv`

## Notes
- The app runs on http://localhost:8501 by default
- First load may take a few minutes due to UMAP and association rules computation
