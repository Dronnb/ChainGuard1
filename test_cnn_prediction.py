
import pandas as pd
import numpy as np

from cnn_inference import CNNWalletRisk


# Load historical transactions
DATA_PATH = "data/transactions1.csv"

tx = pd.read_csv(DATA_PATH)

tx.columns = tx.columns.str.strip()

tx["from_address"] = (
    tx["from_address"].astype("string").str.lower().str.strip()
)

tx["to_address"] = (
    tx["to_address"].astype("string").str.lower().str.strip()
)

tx["timestamp_unix"] = pd.to_numeric(
    tx["timestamp_unix"], errors="coerce"
)

tx["value_eth"] = pd.to_numeric(
    tx["value_eth"], errors="coerce"
)

tx = tx.dropna(subset=["value_eth"])
tx = tx[tx["value_eth"] >= 0].copy()


def truthy(series):
    return series.astype(str).str.lower().isin(
        ["true", "1", "yes"]
    )


tx["_receiver_contract"] = truthy(
    tx["receiver_is_contract"]
) if "receiver_is_contract" in tx.columns else False

tx["_created_contract"] = truthy(
    tx["contract_created"]
) if "contract_created" in tx.columns else False


def wallet_features(wallet):

    sent = tx[tx["from_address"] == wallet]
    received = tx[tx["to_address"] == wallet]

    sent_contract = sent[sent["_receiver_contract"]]

    all_times = pd.concat([
        sent["timestamp_unix"],
        received["timestamp_unix"]
    ]).dropna()

    def mean_interval(frame):
        times = frame["timestamp_unix"].dropna().sort_values()

        if len(times) < 2:
            return np.nan

        return times.diff().dropna().mean() / 60

    def safe_min(series):
        return series.min() if len(series) else np.nan

    def safe_max(series):
        return series.max() if len(series) else np.nan

    def safe_mean(series):
        return series.mean() if len(series) else np.nan

    return {
        "Avg min between sent tnx": mean_interval(sent),
        "Avg min between received tnx": mean_interval(received),

        "Time Diff between first and last (Mins)": (
            (all_times.max() - all_times.min()) / 60
            if len(all_times) >= 2 else np.nan
        ),

        "Sent tnx": len(sent),
        "Received Tnx": len(received),

        "Number of Created Contracts":
            int(sent["_created_contract"].sum()),

        "Unique Received From Addresses":
            received["from_address"].nunique(),

        "Unique Sent To Addresses":
            sent["to_address"].nunique(),

        "min value received": safe_min(received["value_eth"]),
        "max value received": safe_max(received["value_eth"]),
        "avg val received": safe_mean(received["value_eth"]),

        "min val sent": safe_min(sent["value_eth"]),
        "max val sent": safe_max(sent["value_eth"]),
        "avg val sent": safe_mean(sent["value_eth"]),

        "min value sent to contract":
            safe_min(sent_contract["value_eth"]),

        "max val sent to contract":
            safe_max(sent_contract["value_eth"]),

        "avg value sent to contract":
            safe_mean(sent_contract["value_eth"]),

        "total Ether sent": sent["value_eth"].sum(),
        "total ether received": received["value_eth"].sum(),

        "total ether sent contracts":
            sent_contract["value_eth"].sum()
    }


# Select a wallet with historical activity
wallet = tx["from_address"].dropna().iloc[0]

# Extract wallet features
features = wallet_features(wallet)

# Load trained CNN
predictor = CNNWalletRisk()

# Predict wallet risk
result = predictor.predict(features)


print("\n========== CHAINGUARD WALLET ANALYSIS ==========")

print("Wallet:", wallet)
print("Transactions sent:", features["Sent tnx"])
print("Transactions received:", features["Received Tnx"])

print("\nCNN Result:")
print("Risk Score:", result["wallet_risk_score"])
print("Risk Percentage:", result["risk_percentage"], "%")
print("Risk Category:", result["risk_category"])

print("\nWarning:", result["warning"])
