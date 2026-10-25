
import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "transactions1.csv"
OUTPUT_DIR = BASE_DIR / "data"

# Load transaction dataset
df = pd.read_csv(DATA_FILE)

print("Dataset loaded successfully")
print("Total transactions:", len(df))

# Check required columns
required = ["from_address", "to_address", "value_eth"]

missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Clean wallet addresses
df["from_address"] = (
    df["from_address"].astype("string").str.strip().str.lower()
)
df["to_address"] = (
    df["to_address"].astype("string").str.strip().str.lower()
)

# Keep valid wallet-to-wallet connections
edges = df.dropna(
    subset=["from_address", "to_address"]
).copy()

edges = edges[
    (edges["from_address"] != "") &
    (edges["to_address"] != "")
]

# Convert transaction value to numeric
edges["value_eth"] = pd.to_numeric(
    edges["value_eth"], errors="coerce"
).fillna(0)

# Aggregate transactions between the same pair of wallets
edges = (
    edges.groupby(
        ["from_address", "to_address"],
        as_index=False
    )
    .agg(
        transaction_count=("value_eth", "size"),
        total_value_eth=("value_eth", "sum"),
        average_value_eth=("value_eth", "mean")
    )
)

# Build a node list containing all participating wallets
wallets = pd.Index(
    pd.concat([
        edges["from_address"],
        edges["to_address"]
    ]).unique(),
    name="wallet_address"
)

nodes = pd.DataFrame({"wallet_address": wallets})

# Calculate basic incoming and outgoing statistics
outgoing = edges.groupby("from_address").agg(
    outgoing_transactions=("transaction_count", "sum"),
    outgoing_eth=("total_value_eth", "sum")
)

incoming = edges.groupby("to_address").agg(
    incoming_transactions=("transaction_count", "sum"),
    incoming_eth=("total_value_eth", "sum")
)

nodes = nodes.join(
    outgoing, on="wallet_address"
).join(
    incoming, on="wallet_address"
)

stat_columns = [
    "outgoing_transactions",
    "outgoing_eth",
    "incoming_transactions",
    "incoming_eth"
]

nodes[stat_columns] = nodes[stat_columns].fillna(0)

# Save graph data
nodes.to_csv(OUTPUT_DIR / "graph_nodes.csv", index=False)
edges.to_csv(OUTPUT_DIR / "graph_edges.csv", index=False)

print("\n========== GRAPH SUMMARY ==========")
print("Unique wallets (nodes):", len(nodes))
print("Unique wallet connections (edges):", len(edges))
print("Saved:", OUTPUT_DIR / "graph_nodes.csv")
print("Saved:", OUTPUT_DIR / "graph_edges.csv")
print("\nSample wallet nodes:")
print(nodes.head())
print("\nSample wallet connections:")
print(edges.head())
