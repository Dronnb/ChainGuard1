
import pandas as pd
import numpy as np
import networkx as nx

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

print("Loading ChainGuard graph data...")

# 1. Load existing files
nodes = pd.read_csv("data/graph_nodes.csv")
edges = pd.read_csv("data/graph_edges.csv")
predictions = pd.read_csv("data/cnn_wallet_predictions.csv")

# Normalize wallet addresses
for df, columns in [
    (nodes, ["wallet_address"]),
    (edges, ["from_address", "to_address"]),
    (predictions, ["wallet_address"]),
]:
    for column in columns:
        df[column] = df[column].astype(str).str.lower().str.strip()

# Keep one prediction per wallet
predictions = predictions.drop_duplicates(
    subset="wallet_address", keep="first"
)

# 2. Build the directed transaction graph
G = nx.DiGraph()

for row in nodes.itertuples(index=False):
    G.add_node(row.wallet_address)

for row in edges.itertuples(index=False):
    G.add_edge(
        row.from_address,
        row.to_address,
        transaction_count=float(row.transaction_count),
        total_value_eth=float(row.total_value_eth),
    )

print(f"Graph nodes: {G.number_of_nodes()}")
print(f"Graph edges: {G.number_of_edges()}")

# 3. Calculate graph features
print("Calculating graph features...")

in_degree = dict(G.in_degree())
out_degree = dict(G.out_degree())
pagerank = nx.pagerank(G, max_iter=200)

self_loops = {
    wallet: int(G.has_edge(wallet, wallet))
    for wallet in G.nodes()
}

graph_features = pd.DataFrame({
    "wallet_address": list(G.nodes()),
    "graph_in_degree": [
        in_degree[w] for w in G.nodes()
    ],
    "graph_out_degree": [
        out_degree[w] for w in G.nodes()
    ],
    "pagerank": [
        pagerank[w] for w in G.nodes()
    ],
    "has_self_loop": [
        self_loops[w] for w in G.nodes()
    ],
})

# 4. Combine graph features with node statistics
results = nodes.merge(
    graph_features,
    on="wallet_address",
    how="left"
)

# Add CNN scores and predictions
results = results.merge(
    predictions[
        ["wallet_address", "cnn_fraud_score", "cnn_prediction"]
    ],
    on="wallet_address",
    how="left"
)

results["cnn_fraud_score"] = pd.to_numeric(
    results["cnn_fraud_score"], errors="coerce"
)

results["cnn_score_missing"] = (
    results["cnn_fraud_score"].isna().astype(int)
)

# Keep missing CNN scores distinguishable from real scores.
# Use the median only as a neutral imputation for the detector.
median_score = results["cnn_fraud_score"].median()

if pd.isna(median_score):
    median_score = 0.5

results["cnn_fraud_score"] = (
    results["cnn_fraud_score"].fillna(median_score)
)

# 5. Prepare features for unsupervised anomaly detection
feature_columns = [
    "outgoing_transactions",
    "outgoing_eth",
    "incoming_transactions",
    "incoming_eth",
    "graph_in_degree",
    "graph_out_degree",
    "pagerank",
    "has_self_loop",
    "cnn_fraud_score",
    "cnn_score_missing",
]

X = results[feature_columns].replace(
    [np.inf, -np.inf], np.nan
)

X = X.fillna(X.median()).fillna(0)

# Log transformation reduces the influence of very large values
for column in [
    "outgoing_transactions",
    "outgoing_eth",
    "incoming_transactions",
    "incoming_eth",
    "graph_in_degree",
    "graph_out_degree",
]:
    X[column] = np.log1p(X[column].clip(lower=0))

X_scaled = StandardScaler().fit_transform(X)

# 6. Detect unusual wallets
print("Running Isolation Forest...")

detector = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)

labels = detector.fit_predict(X_scaled)

results["graph_anomaly"] = np.where(
    labels == -1, "Anomaly", "Normal"
)

# Higher score means more unusual relative to this dataset
results["graph_anomaly_score"] = -detector.score_samples(
    X_scaled
)

# 7. Create a review priority, not a confirmed-fraud label
results["review_priority_score"] = (
    0.5 * results["cnn_fraud_score"]
    + 0.5 * (
        results["graph_anomaly_score"].rank(pct=True)
    )
)

results = results.sort_values(
    "review_priority_score", ascending=False
)

# 8. Save results
output_path = "data/graph_anomaly_results.csv"
results.to_csv(output_path, index=False)

print("\n========== GRAPH ANALYSIS RESULTS ==========")
print("Wallets analyzed:", len(results))
print(
    "\nAnomaly counts:\n",
    results["graph_anomaly"].value_counts().to_string()
)

print("\nTop 10 wallets for review:")
print(
    results[
        [
            "wallet_address",
            "cnn_fraud_score",
            "graph_anomaly",
            "graph_anomaly_score",
            "review_priority_score",
        ]
    ].head(10).to_string(index=False)
)

print(f"\nSaved results to: {output_path}")
print(
    "\nReminder: anomalies and CNN scores are risk signals, "
    "not verified fraud labels."
)
