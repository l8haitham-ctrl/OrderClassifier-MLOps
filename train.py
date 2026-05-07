"""
TP2 - MLOps avec MLflow
Pipeline d'entranement : 3 modles sur le dataset E-Commerce Orders
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "sample_orders.csv")
EXPERIMENT_NAME = "clf-v1"
TARGET_COLUMN = "status"  # Predict the order status
RANDOM_STATE = 42

# ============================================================================
# STEP 1: DATA PREPARATION
# ============================================================================
def load_and_prepare_data(data_path: str):
    """Load the e-commerce orders dataset and engineer features for ML."""
    print(" Loading dataset...")
    df = pd.read_csv(data_path)

    # Drop rows with nulls
    df = df.dropna()

    # Sample data to speed up training for the TP
    if len(df) > 5000:
        df = df.sample(n=5000, random_state=42)

    # Feature Engineering
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["day_of_week"] = df["order_date"].dt.dayofweek
    df["month"] = df["order_date"].dt.month
    df["quarter"] = df["order_date"].dt.quarter
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Encode categorical features
    le_category = LabelEncoder()
    le_region = LabelEncoder()
    df["product_category_enc"] = le_category.fit_transform(df["product_category"])
    df["region_enc"] = le_region.fit_transform(df["region"])

    # Define features and target
    feature_cols = [
        "amount", "customer_id", "product_category_enc", "region_enc",
        "day_of_week", "month", "quarter", "is_weekend"
    ]
    X = df[feature_cols]
    y = LabelEncoder().fit_transform(df[TARGET_COLUMN])

    print(f" Dataset ready: {X.shape[0]:,} rows, {X.shape[1]} features, {len(np.unique(y))} classes")
    print(f"   Classes: {dict(zip(np.unique(df[TARGET_COLUMN]), np.unique(y)))}")
    return X, y, feature_cols


def save_confusion_matrix(y_test, y_pred, model_name: str, labels) -> str:
    """Generate and save a confusion matrix plot as an artifact."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()

    artifact_path = f"confusion_matrix_{model_name.replace(' ', '_')}.png"
    plt.savefig(artifact_path, dpi=120)
    plt.close()
    return artifact_path


# ============================================================================
# STEP 2: TRAIN & LOG MODELS
# ============================================================================
def train_and_log_model(model, model_name, X_train, X_test, y_train, y_test,
                         params: dict, flavor="sklearn"):
    """Train a model, compute metrics, and log everything to MLflow."""
    print(f"\n Training: {model_name}")

    with mlflow.start_run(run_name=model_name):
        # Log parameters
        for key, value in params.items():
            mlflow.log_param(key, value)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))

        # Train
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Compute metrics
        accuracy  = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1        = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Log confusion matrix artifact
        labels = ["cancelled", "completed", "pending", "returned"]
        cm_path = save_confusion_matrix(y_test, y_pred, model_name, labels)
        mlflow.log_artifact(cm_path)
        os.remove(cm_path)  # Clean up local file

        # Log the model (but don't register it yet)
        signature = infer_signature(X_train, model.predict(X_train))
        if flavor == "xgboost":
            mlflow.xgboost.log_model(model, "model", signature=signature)
        else:
            mlflow.sklearn.log_model(model, "model", signature=signature)

        run_id = mlflow.active_run().info.run_id
        print(f"    accuracy={accuracy:.4f} | precision={precision:.4f} | "
              f"recall={recall:.4f} | f1={f1:.4f}")
        print(f"    Run ID: {run_id}")

        return {
            "model_name": model_name,
            "run_id": run_id,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "model": model,
        }


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    print("=" * 60)
    print("  TP2 - MLOps Pipeline : E-Commerce Order Status Classifier")
    print("=" * 60)

    # Setup MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"\n MLflow Experiment: {EXPERIMENT_NAME}")
    print(f" Tracking URI: http://localhost:5000")

    # Load data
    X, y, features = load_and_prepare_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Scale features (needed for SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    results = []

    #  Model 1: Random Forest 
    rf_params = {"n_estimators": 200, "max_depth": 15, "min_samples_split": 5,
                 "random_state": RANDOM_STATE}
    rf_model = RandomForestClassifier(**rf_params, n_jobs=-1)
    results.append(train_and_log_model(
        rf_model, "Random Forest",
        X_train, X_test, y_train, y_test,
        rf_params, flavor="sklearn"
    ))

    #  Model 2: XGBoost 
    xgb_params = {"n_estimators": 200, "max_depth": 8, "learning_rate": 0.1,
                  "subsample": 0.8, "random_state": RANDOM_STATE}
    xgb_model = XGBClassifier(**xgb_params, use_label_encoder=False,
                               eval_metric="mlogloss", n_jobs=-1)
    results.append(train_and_log_model(
        xgb_model, "XGBoost",
        X_train, X_test, y_train, y_test,
        xgb_params, flavor="xgboost"
    ))

    #  Model 3: SVM 
    svm_params = {"C": 1.0, "kernel": "rbf", "random_state": RANDOM_STATE}
    svm_model = SVC(**svm_params)
    results.append(train_and_log_model(
        svm_model, "SVM",
        X_train_scaled, X_test_scaled, y_train, y_test,
        svm_params, flavor="sklearn"
    ))

    #  Summary 
    print("\n" + "=" * 60)
    print("   RESULTS SUMMARY")
    print("=" * 60)
    results_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "model"} for r in results
    ])
    print(results_df[["model_name", "accuracy", "precision", "recall", "f1_score"]]
          .to_string(index=False))

    best = max(results, key=lambda r: r["accuracy"])
    print(f"\n Best Model: {best['model_name']} (accuracy={best['accuracy']:.4f})")
    print(f"   Run ID: {best['run_id']}")
    print(f"\n View results: http://localhost:5000/#/experiments")
    print("=" * 60)

    # Save best run id for CI/CD
    run_id_path = os.path.join(os.path.dirname(__file__), "best_run_id.txt")
    with open(run_id_path, "w") as f:
        f.write(best["run_id"])

    # Register Best Model and Transition to Staging
    client = MlflowClient()
    best_model_name = best["model_name"].replace(" ", "_")
    
    try:
        # Register the best model using its run_id
        print(f"\n Registering the best model: {best_model_name}...")
        model_uri = f"runs:/{best['run_id']}/model"
        mlflow.register_model(model_uri, best_model_name)
        
        # Get the newly registered model version
        versions = client.search_model_versions(f"name='{best_model_name}'")
        if versions:
            # Sort to get the latest version (it should be the only one, but just in case)
            best_version = max(versions, key=lambda v: int(v.version)).version
            print(f" Transitioning {best_model_name} version {best_version} to Staging...")
            client.transition_model_version_stage(
                name=best_model_name,
                version=best_version,
                stage="Staging",
                archive_existing_versions=True
            )
            client.update_registered_model(
                name=best_model_name,
                description="E-Commerce Order Status Classifier"
            )
            client.update_model_version(
                name=best_model_name,
                version=best_version,
                description="Best performing model from experiment clf-v1, promoted to Staging."
            )
            client.set_model_version_tag(
                name=best_model_name,
                version=best_version,
                key="accuracy",
                value=str(round(best["accuracy"], 4))
            )
            print(" Registration and Transition successful.")
    except Exception as e:
        print(f" Could not register/transition model: {e}")

    return results


if __name__ == "__main__":
    main()
