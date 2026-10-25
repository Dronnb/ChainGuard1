
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    average_precision_score,
    roc_auc_score,
)

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "transaction_dataset.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cpu")

# --------------------------------------------------
# 2. Load and clean the labeled wallet dataset
# --------------------------------------------------
df = pd.read_csv(DATA_PATH)

if "FLAG" not in df.columns:
    raise ValueError("FLAG column is missing.")

df = df.dropna(subset=["FLAG"]).copy()
df["FLAG"] = pd.to_numeric(df["FLAG"], errors="coerce")
df = df.dropna(subset=["FLAG"])
df = df[df["FLAG"].isin([0, 1])].copy()
df["FLAG"] = df["FLAG"].astype(int)

# Remove duplicate wallet records to reduce leakage risk.
if "Address" in df.columns:
    before = len(df)
    df = df.drop_duplicates(subset=["Address"], keep="first")
    print("Duplicate wallet rows removed:", before - len(df))

# Keep numerical features for this first CNN experiment.
excluded = {
    "FLAG",
    "Address",
    "Index",
    "Unnamed: 0",
    "ERC20 most sent token type",
    "ERC20_most_rec_token_type",
}

feature_columns = [
    col for col in df.columns if col not in excluded
]

X = df[feature_columns].copy()

# Convert feature columns to numeric; invalid values become missing.
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

# Remove columns with no usable numerical values.
X = X.dropna(axis=1, how="all")
X = X.replace([np.inf, -np.inf], np.nan)

# Remove constant columns.
X = X.loc[:, X.nunique(dropna=True) > 1]

y = df.loc[X.index, "FLAG"]

if X.shape[1] == 0:
    raise ValueError("No usable numerical features were found.")

print("\nDataset ready")
print("Wallet records:", len(X))
print("Numerical features:", X.shape[1])
print("Label counts:")
print(y.value_counts().sort_index())

# --------------------------------------------------
# 3. Split BEFORE fitting preprocessing
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Fit preprocessing only on training data.
imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# CNN input shape: samples, channels, features.
X_train = torch.tensor(
    X_train, dtype=torch.float32
).unsqueeze(1)

X_test = torch.tensor(
    X_test, dtype=torch.float32
).unsqueeze(1)

y_train_tensor = torch.tensor(
    y_train.to_numpy(), dtype=torch.long
)
y_test_tensor = torch.tensor(
    y_test.to_numpy(), dtype=torch.long
)

# --------------------------------------------------
# 4. Define a 1D CNN
# --------------------------------------------------
class WalletCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
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
        x = self.features(x)
        return self.classifier(x)


model = WalletCNN().to(device)

# Give the minority fraud class greater weight.
class_counts = np.bincount(
    y_train.to_numpy(), minlength=2
)
class_weights = len(y_train) / (
    2.0 * np.maximum(class_counts, 1)
)

weights = torch.tensor(
    class_weights, dtype=torch.float32
)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.001
)

# --------------------------------------------------
# 5. Train the CNN
# --------------------------------------------------
epochs = 30
batch_size = 128

print("\nTraining CNN on", device)
print("Training rows:", len(X_train))
print("Test rows:", len(X_test))

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    order = torch.randperm(len(X_train))

    for start in range(0, len(order), batch_size):
        batch_indices = order[start:start + batch_size]

        xb = X_train[batch_indices].to(device)
        yb = y_train_tensor[batch_indices].to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch_indices)

    if (epoch + 1) % 5 == 0:
        avg_loss = total_loss / len(X_train)
        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- Loss: {avg_loss:.4f}"
        )

# --------------------------------------------------
# 6. Evaluate on the held-out test set
# --------------------------------------------------
model.eval()

with torch.no_grad():
    logits = model(X_test.to(device))
    probabilities = torch.softmax(logits, dim=1)
    predictions = probabilities.argmax(dim=1).cpu().numpy()
    fraud_scores = probabilities[:, 1].cpu().numpy()

y_true = y_test.to_numpy()

print("\n========== CNN RESULTS ==========")
print("Accuracy:", round(
    accuracy_score(y_true, predictions), 4
))
print("Fraud PR-AUC:", round(
    average_precision_score(y_true, fraud_scores), 4
))
print("ROC-AUC:", round(
    roc_auc_score(y_true, fraud_scores), 4
))

print("\nConfusion matrix:")
print(confusion_matrix(y_true, predictions))

print("\nClassification report:")
print(classification_report(
    y_true,
    predictions,
    labels=[0, 1],
    target_names=["Not fraud (0)", "Fraud (1)"],
    zero_division=0,
))

# --------------------------------------------------
# 7. Save model and preprocessing objects
# --------------------------------------------------
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_features": list(X.columns),
        "epochs": epochs,
    },
    MODEL_DIR / "wallet_cnn.pt",
)

import joblib

joblib.dump(imputer, MODEL_DIR / "imputer.pkl")
joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

print("\nSaved CNN and preprocessing files in:", MODEL_DIR)
