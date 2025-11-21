import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTETomek
import os

def preprocess_data(input_path, output_path):
    print("=== Memulai Preprocessing ===")

    # 1. Load data
    df = pd.read_csv(input_path)
    print("Data Loaded:", df.shape)

    # 2. Drop kolom tidak penting
    if 'StudentID' in df.columns:
        df.drop(columns=['StudentID'], inplace=True)

    # 3. Replace mapping
    replacements = {
        'ExtracurricularActivities': {'No': 0, 'Yes': 1},
        'PlacementTraining': {'No': 0, 'Yes': 1},
        'PlacementStatus': {'NotPlaced': 0, 'Placed': 1},
    }
    df.replace(replacements, inplace=True)

    # 4. Split X - y
    X = df.drop(columns='PlacementStatus')
    y = df['PlacementStatus']

    # 5. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 6. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 7. SMOTE-Tomek
    smote_tomek = SMOTETomek(random_state=32)
    X_train_res, y_train_res = smote_tomek.fit_resample(X_train_scaled, y_train)

    # 8. Satukan kembali hasilnya
    processed_df = pd.DataFrame(X_train_res, columns=X.columns)
    processed_df['PlacementStatus'] = y_train_res

    # 9. Save hasil preprocessing
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "processed_dataset.csv")
    processed_df.to_csv(output_file, index=False)

    print("Preprocessing selesai.")
    print("Dataset disimpan di:", output_file)


if __name__ == "__main__":
    preprocess_data(
        input_path="placementdata_raw.csv",
        output_path="preprocessing"
    )
