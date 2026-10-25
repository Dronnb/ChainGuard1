
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from pathlib import Path
import joblib

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
MODELS = BASE / "models"
MODELS.mkdir(exist_ok=True)

TRAIN_FILE = DATA / "transaction_dataset.csv"
TX_FILE = DATA / "transactions1.csv"

# These features must exist in the labeled wallet dataset
FEATURES = [
    "Avg min between sent tnx",
    "Avg min between received tnx",
    "Time Diff between first and last (Mins)",
    "Sent tnx",
    "Received Tnx",
    "Number of Created Contracts",
    "Unique Received From Addresses",
    "Unique Sent To Addresses",
    "min value received",
    "max value received",
    "avg val received",
    "min val sent",
    "max val sent",
    "avg val sent",
    "min value sent to contract",
    "max val sent to contract",
    "avg value sent to contract",
    "total Ether sent",
    "total ether received",
    "total ether sent contracts",
]

# Same CNN family, adapted to the selected feature count
class WalletCNN(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        return self.classifier(self.conv(x.unsqueeze(1)))


# ---------- 1. Load labelled wallet dataset ----------
df = pd.read_csv(TRAIN_FILE)
df.columns = df.columns.str.strip()

required = FEATURES + ["FLAG"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Training dataset is missing columns: {missing}")

df["FLAG"] = pd.to_numeric(df["FLAG"], errors="coerce")
df = df[df["FLAG"].isin([0, 1])].copy()

if "Address" in df.columns:
    df = df.drop_duplicates(subset="Address")

X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)
X = X.dropna(axis=1, how="all")

# Keep only features with actual variation
X = X.loc[:, X.nunique(dropna=True) > 1]
used_features = X.columns.tolist()
y = df.loc[X.index, "FLAG"].astype(int)

if len(used_features) < 2:
    raise ValueError("Not enough usable features to train the model.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

X_train = scaler.fit_transform(imputer.fit_transform(X_train))
X_test = scaler.transform(imputer.transform(X_test))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WalletCNN(len(used_features)).to(device)

counts = np.bincount(y_train.to_numpy(), minlength=2)
weights = len(y_train) / (2 * np.maximum(counts, 1))
criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(weights, dtype=torch.float32, device=device)
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_ds = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train.to_numpy(), dtype=torch.long),
)
loader = DataLoader(train_ds, batch_size=128, shuffle=True)

print("Training compatible CNN...")
print("Features:", len(used_features))
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

for epoch in range(30):
    model.train()
    total_loss = 0.0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 5 == 0:
        print(
            f"Epoch {epoch + 1}/30 - "
            f"Loss: {total_loss / len(loader):.4f}"
        )

# ---------- 2. Evaluate on held-out labelled wallets ----------
model.eval()
with torch.no_grad():
    logits = model(
        torch.tensor(X_test, dtype=torch.float32, device=device)
    )
    probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    predictions = (probabilities >= 0.5).astype(int)

print("\n========== COMPATIBLE CNN RESULTS ==========")
print(classification_report(
    y_test, predictions,
    labels=[0, 1],
    target_names=["Not fraud", "Fraud"],
    zero_division=0
))

if y_test.nunique() == 2:
    print("Fraud ROC-AUC:", round(
        roc_auc_score(y_test, probabilities), 4
    ))

# Save separately; do not overwrite the original CNN
torch.save(
    {
        "model_state_dict": model.cpu().state_dict(),
        "input_features": used_features,
        "epochs": 30,
    },
    MODELS / "compatible_wallet_cnn.pt",
)
joblib.dump(imputer, MODELS / "compatible_imputer.pkl")
joblib.dump(scaler, MODELS / "compatible_scaler.pkl")

# ---------- 3. Build wallet features from transactions ----------
tx = pd.read_csv(TX_FILE)
tx.columns = tx.columns.str.strip()

for col in ["from_address", "to_address"]:
    tx[col] = tx[col].astype("string").str.lower().str.strip()
    tx[col] = tx[col].replace("", pd.NA)

tx["timestamp_unix"] = pd.to_numeric(
    tx["timestamp_unix"], errors="coerce"
)
tx["value_eth"] = pd.to_numeric(
    tx["value_eth"], errors="coerce"
)

# Do not silently turn missing ETH values into zero
tx = tx.dropna(subset=["value_eth"])
tx = tx[tx["value_eth"] >= 0].copy()

def truthy(series):
    return series.astype(str).str.lower().isin(
        ["true", "1", "yes"]
    )

if "receiver_is_contract" in tx.columns:
    receiver_contract = truthy(tx["receiver_is_contract"])
else:
    receiver_contract = pd.Series(False, index=tx.index)

if "contract_created" in tx.columns:
    created_contract = truthy(tx["contract_created"])
else:
    created_contract = pd.Series(False, index=tx.index)

tx["_receiver_contract"] = receiver_contract
tx["_created_contract"] = created_contract

wallets = set(tx["from_address"].dropna()) | set(
    tx["to_address"].dropna()
)

def mean_interval_minutes(frame):
    times = frame["timestamp_unix"].dropna().sort_values()
    if len(times) < 2:
        return np.nan
    return times.diff().dropna().mean() / 60.0

def wallet_features(wallet):
    sent = tx[tx["from_address"] == wallet]
    received = tx[tx["to_address"] == wallet]

    sent_contract = sent[sent["_receiver_contract"]]
    all_times = pd.concat([
        sent["timestamp_unix"],
        received["timestamp_unix"],
    ]).dropna()

    span = (
        (all_times.max() - all_times.min()) / 60.0
        if len(all_times) >= 2 else np.nan
    )

    def min_or_nan(series):
        return series.min() if len(series) else np.nan

    def max_or_nan(series):
        return series.max() if len(series) else np.nan

    def mean_or_nan(series):
        return series.mean() if len(series) else np.nan

    return {
        "Avg min between sent tnx":
            mean_interval_minutes(sent),
        "Avg min between received tnx":
            mean_interval_minutes(received),
        "Time Diff between first and last (Mins)": span,
        "Sent tnx": len(sent),
        "Received Tnx": len(received),
        "Number of Created Contracts":
            int(sent["_created_contract"].sum()),
        "Unique Received From Addresses":
            received["from_address"].nunique(),
        "Unique Sent To Addresses":
            sent["to_address"].nunique(),
        "min value received": min_or_nan(received["value_eth"]),
        "max value received": max_or_nan(received["value_eth"]),
        "avg val received": mean_or_nan(received["value_eth"]),
        "min val sent": min_or_nan(sent["value_eth"]),
        "max val sent": max_or_nan(sent["value_eth"]),
        "avg val sent": mean_or_nan(sent["value_eth"]),
        "min value sent to contract":
            min_or_nan(sent_contract["value_eth"]),
        "max val sent to contract":
            max_or_nan(sent_contract["value_eth"]),
        "avg value sent to contract":
            mean_or_nan(sent_contract["value_eth"]),
        "total Ether sent": sent["value_eth"].sum(),
        "total ether received": received["value_eth"].sum(),
        "total ether sent contracts":
            sent_contract["value_eth"].sum(),
    }

wallet_df = pd.DataFrame(
    [wallet_features(w) for w in sorted(wallets)],
    index=sorted(wallets),
)
wallet_df.index.name = "wallet_address"

# Match training feature order exactly
target_X = wallet_df.reindex(columns=used_features)
target_X = target_X.replace([np.inf, -np.inf], np.nan)
target_scaled = scaler.transform(imputer.transform(target_X))

model = model.to(device)
model.eval()
with torch.no_grad():
    target_logits = model(
        torch.tensor(target_scaled, dtype=torch.float32, device=device)
    )
    target_probs = torch.softmax(target_logits, dim=1)[:, 1].cpu().numpy()

wallet_df["cnn_fraud_score"] = np.round(target_probs, 4)
wallet_df["cnn_prediction"] = np.where(
    target_probs >= 0.5, "Fraud", "Not Fraud"
)

output = DATA / "cnn_wallet_predictions.csv"
wallet_df.reset_index().to_csv(output, index=False)

print("\nWallets scored:", len(wallet_df))
print("Saved predictions to:", output)
print(
    "Warning: scores are model predictions, not verified fraud labels."
)
