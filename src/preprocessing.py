import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess(input_path, output_path):
    print("Reading data...")
    df = pd.read_csv(os.path.join(input_path, "telco_churn.csv"))

    print(f"Raw shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Drop columns that aren't useful features
    drop_cols = [
        "CustomerID", "Count", "Country", "State", "City",
        "Zip Code", "Lat Long", "Latitude", "Longitude",
        "Churn Label", "Churn Score", "CLTV", "Churn Reason"
    ]
    df.drop(columns=drop_cols, inplace=True)

    # Fix Total Charges - coerce blanks to NaN and drop
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
    df.dropna(inplace=True)

    # Encode binary yes/no columns
    binary_cols = [
        "Partner", "Dependents", "Phone Service", "Paperless Billing",
        "Multiple Lines", "Online Security", "Online Backup",
        "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies"
    ]
    for col in binary_cols:
        df[col] = df[col].map({
            "Yes": 1, "No": 0,
            "No phone service": 0, "No internet service": 0
        })

    # Encode gender
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

    # Encode categorical columns
    cat_cols = ["Internet Service", "Contract", "Payment Method"]
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    # Use Churn Value (0/1) as label, move to first column
    cols = ["Churn Value"] + [c for c in df.columns if c != "Churn Value"]
    df = df[cols]

    print(f"Processed shape: {df.shape}")

    # Split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    # Save
    os.makedirs(output_path, exist_ok=True)
    train_df.to_csv(os.path.join(output_path, "train.csv"), index=False, header=False)
    test_df.to_csv(os.path.join(output_path, "test.csv"), index=False, header=False)
    print("Done. Files written to", output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, default="/opt/ml/processing/input")
    parser.add_argument("--output-path", type=str, default="/opt/ml/processing/output")
    args = parser.parse_args()
    preprocess(args.input_path, args.output_path)