# ======================================================================
# Combined Visualizations using Seaborn + Plotly + Bokeh
# ======================================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from itertools import combinations

import plotly.express as px
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.models import ColumnDataSource

output_notebook()

# --------------------------------------------------------------
# Load Data
# --------------------------------------------------------------
df = pd.read_csv("transactions.csv")
df['Date'] = pd.to_datetime(df["Date"])

# ==============================================================
# MARKET BASKET - APRIORI (Custom)
# ==============================================================

def get_support(itemset, transactions):
    return sum(1 for t in transactions if itemset.issubset(set(t))) / len(transactions)

def apriori(transactions, min_support=0.05):
    items = set(item for t in transactions for item in t)
    itemsets = {frozenset([i]): get_support(frozenset([i]), transactions) for i in items}
    frequent = {k: v for k, v in itemsets.items() if v >= min_support}
    all_freq = dict(frequent)

    k = 2
    while frequent:
        candidates = []
        keys = list(frequent.keys())

        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                u = keys[i] | keys[j]
                if len(u) == k:
                    candidates.append(u)

        frequent = {}

        for c in candidates:
            sup = get_support(c, transactions)
            if sup >= min_support:
                frequent[c] = sup

        all_freq.update(frequent)
        k += 1

    return all_freq

def generate_rules(frequent_itemsets, transactions, min_confidence=0.3):
    rules = []
    for itemset, support in frequent_itemsets.items():
        if len(itemset) < 2:
            continue

        for i in range(1, len(itemset)):
            from itertools import combinations
            for antecedent in combinations(itemset, i):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                conf = support / frequent_itemsets.get(antecedent, 1e-9)
                if conf >= min_confidence:
                    lift = conf / frequent_itemsets.get(consequent, 1e-9)
                    rules.append({
                        "antecedent": list(antecedent),
                        "consequent": list(consequent),
                        "support": support,
                        "confidence": conf,
                        "lift": lift
                    })
    return pd.DataFrame(rules)

# Prepare transactions
transactions = df.groupby("Transaction_Id")["Item"].apply(list).tolist()

freq_sets = apriori(transactions, min_support=0.05)
rules_df = generate_rules(freq_sets, transactions)

# ==============================================================
# RFM ANALYSIS
# ==============================================================

current_day = df['Date'].max() + pd.Timedelta(days=1)

rfm = df.groupby("Customer_Id").agg({
    "Date": lambda x: (current_day - x.max()).days,
    "Transaction_Id": "nunique",
    "Amount": "sum"
}).reset_index()

rfm.columns = ["Customer_Id", "Recency", "Frequency", "Monetary"]

# Normalize and cluster
sc = StandardScaler()
rfm_scaled = sc.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

# ==============================================================
# 1️⃣ SEABORN VISUALIZATIONS
# ==============================================================

plt.figure(figsize=(10,5))
sns.countplot(y=df["Item"], order=df["Item"].value_counts().head(10).index, palette="coolwarm")
plt.title("Top 10 Purchased Items (Seaborn)", fontweight="bold")
plt.tight_layout()
plt.show()

# RFM Histograms
for col in ["Recency", "Frequency", "Monetary"]:
    plt.figure(figsize=(10,5))
    sns.histplot(rfm[col], bins=15, kde=True, color="purple")
    plt.title(f"{col} Distribution (Seaborn)")
    plt.show()

# KMeans cluster scatter
plt.figure(figsize=(8,6))
sns.scatterplot(x=rfm["Frequency"], y=rfm["Monetary"], hue=rfm["Cluster"], palette="tab10", s=100)
plt.title("Customer Clusters (Seaborn)")
plt.show()

# ==============================================================
# 2️⃣ PLOTLY VISUALIZATIONS
# ==============================================================

# Top Items
item_counts = df["Item"].value_counts().reset_index()
item_counts.columns = ["Item", "Count"]

fig = px.bar(
    item_counts.head(10),
    x="Count",
    y="Item",
    orientation="h",
    title="Top 10 Items (Plotly)",
    text="Count",
)
fig.update_traces(textposition="outside")
fig.show()

# RFM Plotly
for col in ["Recency", "Frequency", "Monetary"]:
    fig = px.histogram(rfm, x=col, nbins=20, title=f"{col} Distribution (Plotly)")
    fig.show()

# Cluster Scatter
fig = px.scatter(
    rfm,
    x="Frequency",
    y="Monetary",
    color="Cluster",
    title="Customer Clusters (Plotly)",
    size_max=12
)
fig.show()

# Association Rules (Top 10)
if len(rules_df) > 0:
    top_rules = rules_df.nlargest(10, "lift")
    fig = px.bar(
        top_rules,
        x="lift",
        y=top_rules.apply(lambda r: " + ".join(r["antecedent"]) + " → " + " + ".join(r["consequent"]), axis=1),
        orientation="h",
        title="Top 10 Association Rules (Plotly)",
        text="lift"
    )
    fig.show()

# ==============================================================
# 3️⃣ BOKEH VISUALIZATIONS
# ==============================================================

# Top items
source = ColumnDataSource(item_counts.head(10))

p = figure(
    y_range=item_counts["Item"].head(10)[::-1],
    height=400,
    title="Top 10 Purchased Items (Bokeh)"
)

p.hbar(y="Item", right="Count", height=0.5, source=source, color="orange")

show(p)

# RFM Bokeh
for col in ["Recency", "Frequency", "Monetary"]:
    hist, edges = np.histogram(rfm[col], bins=20)
    p = figure(title=f"{col} Distribution (Bokeh)", height=350)
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], color="teal", alpha=0.7)
    show(p)

# Cluster Scatter
p = figure(title="Customer Clusters (Bokeh)", height=400)
p.circle(rfm["Frequency"], rfm["Monetary"], size=10, color="navy", alpha=0.5)
p.xaxis.axis_label = "Frequency"
p.yaxis.axis_label = "Monetary"
show(p)
