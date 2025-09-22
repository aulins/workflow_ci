# modelling.py
import argparse
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


def train(data_path: Path):
    """
    Train diabetes prediction model with MLflow tracking
    """
    # 1) Load data
    print(f"Loading data from: {data_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"Data loaded successfully. Shape: {df.shape}")
    
    # Basic data validation
    if "Outcome" not in df.columns:
        raise ValueError("Target column 'Outcome' not found in dataset")
    
    X = df.drop(columns=["Outcome"]).values
    y = df["Outcome"].astype(int).values
    
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")
    print(f"Target distribution: {np.bincount(y)}")

    # 2) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

    # 3) Set MLflow tracking URI (lebih fleksibel untuk CI/CD)
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(mlflow_uri)
    
    # Create experiment if not exists
    try:
        experiment = mlflow.get_experiment_by_name("diabetes-basic")
        if experiment is None:
            experiment_id = mlflow.create_experiment("diabetes-basic")
            print(f"Created new experiment with ID: {experiment_id}")
        else:
            print(f"Using existing experiment: {experiment.experiment_id}")
    except Exception as e:
        print(f"Warning: Could not setup experiment: {e}")
    
    mlflow.set_experiment("diabetes-basic")

    # 4) Enable autolog with proper configuration for CI
    mlflow.sklearn.autolog(
        log_model_signatures=True, 
        log_input_examples=False,
        log_models=True,
        disable=False,
        exclusive=False,
        disable_for_unsupported_versions=False,
        silent=False
    )

    # 5) Train model with comprehensive logging
    run_name = f"logreg_baseline_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name):
        print("Starting model training...")
        
        # Log parameters
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)
        
        # Train model
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        
        print("Model training completed. Evaluating...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)
        
        # Log additional metrics for better monitoring
        mlflow.log_metric("train_samples", len(X_train))
        mlflow.log_metric("test_samples", len(X_test))
        mlflow.log_metric("n_features", X.shape[1])
        
        print(f"Model Performance:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")

        # 6) Create and log confusion matrix
        try:
            cm = confusion_matrix(y_test, y_pred)
            
            # Use non-interactive backend for CI environments
            plt.switch_backend('Agg')
            fig = plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt="d", cbar=False, cmap='Blues')
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("Confusion Matrix")
            
            # Save in current directory
            fig_path = Path("confusion_matrix.png")
            plt.tight_layout()
            fig.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Log artifact
            mlflow.log_artifact(str(fig_path))
            print(f"Confusion matrix saved and logged: {fig_path}")
            
            # Clean up
            if fig_path.exists():
                fig_path.unlink()
                
        except Exception as e:
            print(f"Warning: Could not create confusion matrix: {e}")
        
        # 7) Log feature importance if available
        try:
            if hasattr(model, 'coef_'):
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                importance = abs(model.coef_[0])
                
                # Create feature importance plot
                plt.figure(figsize=(8, 6))
                indices = np.argsort(importance)[::-1][:10]  # Top 10 features
                plt.bar(range(len(indices)), importance[indices])
                plt.xlabel("Feature Index")
                plt.ylabel("Importance (|coefficient|)")
                plt.title("Top 10 Feature Importances")
                
                importance_path = Path("feature_importance.png")
                plt.tight_layout()
                plt.savefig(importance_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                mlflow.log_artifact(str(importance_path))
                print(f"Feature importance plot logged: {importance_path}")
                
                # Clean up
                if importance_path.exists():
                    importance_path.unlink()
                    
        except Exception as e:
            print(f"Warning: Could not create feature importance plot: {e}")
        
        # 8) Model validation check
        min_accuracy = float(os.getenv("MIN_ACCURACY", "0.70"))
        if acc >= min_accuracy:
            print(f"✅ Model validation PASSED (accuracy {acc:.4f} >= {min_accuracy})")
            mlflow.log_param("validation_status", "PASSED")
        else:
            print(f"❌ Model validation FAILED (accuracy {acc:.4f} < {min_accuracy})")
            mlflow.log_param("validation_status", "FAILED")
            # Don't exit with error in CI, just log the status
        
        # Get current run info
        current_run = mlflow.active_run()
        print(f"Run ID: {current_run.info.run_id}")
        print(f"Experiment ID: {current_run.info.experiment_id}")

    print("Training completed successfully!")
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print("Check your MLflow UI for detailed results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train diabetes prediction model")
    parser.add_argument(
        "--data",
        type=str,
        default="../dataset_preprocessing/diabetes_preprocessed.csv",
        help="Path to CSV file with preprocessed data",
    )
    args = parser.parse_args()
    
    print(f"Starting training with data: {args.data}")
    train(Path(args.data))