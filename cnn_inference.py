
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

CHECKPOINT_PATH = MODEL_DIR / "compatible_wallet_cnn.pt"
IMPUTER_PATH = MODEL_DIR / "compatible_imputer.pkl"
SCALER_PATH = MODEL_DIR / "compatible_scaler.pkl"


# ============================================================
# 2. FEATURE NAMES
# Must match train_compatible_cnn.py exactly
# ============================================================

FEATURE_NAMES = [
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


# ============================================================
# 3. CNN ARCHITECTURE
# Exact architecture from train_compatible_cnn.py
# ============================================================

class WalletCNN(nn.Module):

    def __init__(self, n_features):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(
                1, 16,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(
                16, 32,
                kernel_size=3,
                padding=1
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
        x = x.unsqueeze(1)
        x = self.conv(x)
        return self.classifier(x)


# ============================================================
# 4. CNN INFERENCE CLASS
# ============================================================

class CNNWalletRisk:

    def __init__(self):

        # Check whether all model files exist
        required_files = [
            CHECKPOINT_PATH,
            IMPUTER_PATH,
            SCALER_PATH,
        ]

        for path in required_files:
            if not path.exists():
                raise FileNotFoundError(
                    f"Required model file not found: {path}"
                )

        # Select CPU or GPU
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print("Loading compatible CNN...")

        # Load trained checkpoint
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=self.device,
            weights_only=False,
        )

        # Retrieve exact feature order from checkpoint
        checkpoint_features = checkpoint.get("input_features")

        if not isinstance(checkpoint_features, (list, tuple)):
            raise ValueError(
                "Checkpoint does not contain the expected "
                "list of input feature names."
            )

        self.feature_names = list(checkpoint_features)

        # Verify feature count
        if len(self.feature_names) != len(FEATURE_NAMES):
            raise ValueError(
                f"Checkpoint expects {len(self.feature_names)} "
                f"features, but inference defines "
                f"{len(FEATURE_NAMES)}."
            )

        # Verify feature order
        if self.feature_names != FEATURE_NAMES:
            raise ValueError(
                "Feature names or order do not match "
                "the trained CNN checkpoint."
            )

        # Build exact trained architecture
        self.model = WalletCNN(
            n_features=len(self.feature_names)
        ).to(self.device)

        # Load trained weights
        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

        # Load preprocessing objects
        self.imputer = joblib.load(IMPUTER_PATH)
        self.scaler = joblib.load(SCALER_PATH)

        print("Compatible CNN loaded successfully!")
        print(f"Device: {self.device}")
        print(f"Input features: {len(self.feature_names)}")
        print("Feature order verified.")
        print("Imputer loaded.")
        print("Scaler loaded.")


    # ========================================================
    # 5. WALLET RISK PREDICTION
    # ========================================================

    def predict(self, wallet_features):

        """
        Input:
            Dictionary containing the 20 wallet features.

        Output:
            Wallet-level fraud risk probability,
            preliminary risk category and warning.
        """

        # Check for missing feature names
        missing_features = [
            name
            for name in self.feature_names
            if name not in wallet_features
        ]

        if missing_features:
            raise ValueError(
                f"Missing wallet features: {missing_features}"
            )

        # Construct feature array in training order
        values = np.array(
            [[
                wallet_features[name]
                for name in self.feature_names
            ]],
            dtype=np.float64,
        )

        # Replace infinite values with NaN
        # The trained imputer will handle missing values.
        values = np.where(
            np.isfinite(values),
            values,
            np.nan
        )

        # Preserve feature names during preprocessing
        feature_df = pd.DataFrame(
            values,
            columns=self.feature_names
        )

        # Apply trained imputer
        imputed_values = self.imputer.transform(feature_df)

        # Restore feature names for the scaler
        imputed_df = pd.DataFrame(
            imputed_values,
            columns=self.feature_names
        )

        # Apply trained scaler
        scaled_values = self.scaler.transform(
        imputed_df.to_numpy()
        )

        # Convert to PyTorch tensor
        tensor = torch.tensor(
            scaled_values,
            dtype=torch.float32,
            device=self.device,
        )

        # Model inference
        with torch.no_grad():

            logits = self.model(tensor)

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            fraud_probability = float(
                probabilities[0, 1].item()
            )

        # Preliminary risk categorization
        if fraud_probability >= 0.70:
            category = "High Risk"

        elif fraud_probability >= 0.40:
            category = "Medium Risk"

        else:
            category = "Low Risk"

        return {
            "wallet_risk_score": round(
                fraud_probability, 4
            ),

            "risk_percentage": round(
                fraud_probability * 100, 2
            ),

            "risk_category": category,

            "warning": (
                "Model-generated wallet risk estimate. "
                "Not a confirmed fraud finding."
            ),
        }


# ============================================================
# 6. BASIC MODEL LOADING TEST
# ============================================================

if __name__ == "__main__":

    try:

        predictor = CNNWalletRisk()

        print("\n================================")
        print("CNN INFERENCE MODULE READY")
        print("================================")

    except Exception as error:

        print("\nCNN INITIALIZATION FAILED")
        print("Error:", error)
        raise
