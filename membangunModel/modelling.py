import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from imblearn.combine import SMOTETomek
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import pkg_resources

# Path untuk menyimpan output lokal
OUTPUT_PATH = Path(r"C:\sistemMachienlearning\membangunModel")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


df = pd.read_csv("membangunModel/processed_dataset.csv")

X = df.drop(columns="PlacementStatus")
y = df["PlacementStatus"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Resampling
smote_tomek = SMOTETomek(random_state=32)
X_train_res, y_train_res = smote_tomek.fit_resample(X_train_scaled, y_train)

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Placement_Modelling.py")

# Disable autolog untuk kontrol penuh
mlflow.sklearn.autolog(disable=True)

with mlflow.start_run(run_name="RF_Modelling") as run:
    # PARAMETER MODEL
    params = {
        'n_estimators': 300,
        'max_depth': 7,
        'min_samples_split': 5,
        'min_samples_leaf': 3,
        'max_features': 'sqrt',
        'class_weight': 'balanced',
        'bootstrap': True,
        'random_state': 42
    }
    
    # LOG PARAMETERS - akan muncul di kolom Parameters
    mlflow.log_params(params)
    
    # MODEL
    model_rf = RandomForestClassifier(**params)
    model_rf.fit(X_train_res, y_train_res)
    
    # PREDIKSI
    pred_train = model_rf.predict(X_train_res)
    pred_test = model_rf.predict(X_test_scaled)

    # EVALUASI - Training Set
    train_acc = accuracy_score(y_train_res, pred_train)
    train_precision = precision_score(y_train_res, pred_train, average='weighted')
    train_recall = recall_score(y_train_res, pred_train, average='weighted')
    train_f1 = f1_score(y_train_res, pred_train, average='weighted')
    
    # EVALUASI - Test Set
    test_acc = accuracy_score(y_test, pred_test)
    test_precision = precision_score(y_test, pred_test, average='weighted')
    test_recall = recall_score(y_test, pred_test, average='weighted')
    test_f1 = f1_score(y_test, pred_test, average='weighted')
    
    # LOG METRICS - akan muncul di kolom Metrics
    mlflow.log_metrics({
        'train_accuracy': train_acc,
        'train_precision': train_precision,
        'train_recall': train_recall,
        'train_f1_score': train_f1,
        'test_accuracy': test_acc,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1_score': test_f1
    })
    
    # LOG MODEL
    mlflow.sklearn.log_model(model_rf, "model")

    # Print hasil
    print("="*60)
    print("TRAINING METRICS:")
    print(f"  Accuracy:  {train_acc:.4f}")
    print(f"  Precision: {train_precision:.4f}")
    print(f"  Recall:    {train_recall:.4f}")
    print(f"  F1-Score:  {train_f1:.4f}")
    print("\nTEST METRICS:")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall:    {test_recall:.4f}")
    print(f"  F1-Score:  {test_f1:.4f}")
    print("="*60)
    
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, pred_test))

    # Confusion Matrix
    cm = confusion_matrix(y_test, pred_test)
    plt.figure(figsize=(7,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Random Forest Modelling")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    # Simpan Confusion Matrix ke LOKAL
    cm_path_local = OUTPUT_PATH / f"confusion_modelling.png"
    plt.savefig(cm_path_local, dpi=300, bbox_inches='tight')
    print(f"Confusion Matrix disimpan di: {cm_path_local}")
    
    # Simpan juga ke MLflow Artifact
    mlflow.log_artifact(str(cm_path_local))
    plt.close()
    
    # Feature Importance
    feature_importances = pd.DataFrame({
        'feature': X.columns,
        'importance': model_rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Plot Feature Importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importances.head(5), x='importance', y='feature', palette='viridis')
    plt.title('Top 5 Feature Importance - Random Forest', fontsize=14, fontweight='bold')
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.tight_layout()
    
    # Simpan Feature Importance ke LOKAL
    fi_path_local = OUTPUT_PATH / f"feature_coefficients_modelling.png"
    plt.savefig(fi_path_local, dpi=300, bbox_inches='tight')
    print(f"Feature Importance disimpan di: {fi_path_local}")
    
    # Simpan juga ke MLflow Artifact
    mlflow.log_artifact(str(fi_path_local))
    plt.close()
    

    # Simpan Classification Report ke file TXT di LOKAL
    report_path_local = OUTPUT_PATH / f"classification_report_modelling.txt"
    with open(report_path_local, 'w') as f:
        f.write("="*60 + "\n")
        f.write("CLASSIFICATION REPORT - Random Forest\n")
        f.write("="*60 + "\n\n")
        f.write("TRAINING METRICS:\n")
        f.write(f"  Accuracy:  {train_acc:.4f}\n")
        f.write(f"  Precision: {train_precision:.4f}\n")
        f.write(f"  Recall:    {train_recall:.4f}\n")
        f.write(f"  F1-Score:  {train_f1:.4f}\n\n")
        f.write("TEST METRICS:\n")
        f.write(f"  Accuracy:  {test_acc:.4f}\n")
        f.write(f"  Precision: {test_precision:.4f}\n")
        f.write(f"  Recall:    {test_recall:.4f}\n")
        f.write(f"  F1-Score:  {test_f1:.4f}\n\n")
        f.write("="*60 + "\n\n")
        f.write(classification_report(y_test, pred_test))
    
    print(f"Classification Report disimpan di: {report_path_local}")
    mlflow.log_artifact(str(report_path_local))
    
        # List library yang digunakan dalam script ini
    required_packages = [
        'pandas',
        'mlflow',
        'scikit-learn',
        'imbalanced-learn',
        'matplotlib',
        'seaborn',
        'joblib',
        'numpy'
    ]

    requirements_list = []
    for package in required_packages:
        try:
            version = pkg_resources.get_distribution(package).version
            requirements_list.append(f"{package}=={version}")
        except:
            requirements_list.append(package)

    requirements_path = OUTPUT_PATH / f"requirements_modelling.txt"
    with open(requirements_path, 'w') as f:
        f.write('\n'.join(requirements_list))

    print(f"Requirements.txt disimpan di: {requirements_path}")
    mlflow.log_artifact(str(requirements_path))

    # Cetak informasi run
    print(f"\n{'='*60}")
    print(f"Run ID: {run.info.run_id}")
    print(f"Artifact URI: {run.info.artifact_uri}")
    print(f"{'='*60}\n")
    print(f"\nMLflow UI: http://127.0.0.1:5000/")