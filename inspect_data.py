
import pandas as pd
from pathlib import Path

data_dir = Path(__file__).parent / "data"

datasets = {
    "TRAINING DATASET": data_dir / "transaction_dataset.csv",
    "TRANSACTION DATASET": data_dir / "transactions1.csv",
}

for title, file_path in datasets.items():
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    if not file_path.exists():
        print(f"File not found: {file_path}")
        continue

    df = pd.read_csv(file_path)

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nMissing values (top 10):")
    print(df.isnull().sum().sort_values(ascending=False).head(10))

    print("\nDuplicate rows:", df.duplicated().sum())

    if "FLAG" in df.columns:
        print("\nFraud label counts:")
        print(df["FLAG"].value_counts(dropna=False))

    if "status" in df.columns:
        print("\nTransaction status counts:")
        print(df["status"].value_counts(dropna=False))

    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))
