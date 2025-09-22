# modelling.py
import argparse
import os
import sys
import warnings
from pathlib import Path
from datetime import datetime

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_auc_score, classification_report
)
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend for CI
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def setup_mlflow():
    """Setup MLflow tracking with proper configuration"""
    # Force local file system tracking to avoid path issues
    mlflow_uri = "file:./mlruns"
    mlflow.set_tracking_uri(mlflow_uri)
    
    # Create experiment if not exists
    experiment_name = "diabetes-basic"
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location="./mlruns"  # Force local path
            )
            print(f"✅ Created new experiment '{experiment_name}' with ID: {experiment_id}")
        else:
            print(f"✅ Using existing experiment '{experiment_name}' (ID: {experiment.experiment_id})")
    except Exception as e:
        print(f"⚠️ Warning: Could not setup experiment: {e}")
    
    mlflow.set_experiment(experiment_name)
    return mlflow.get_experiment_by_name(experiment_name)


def load_and_validate_data(data_path: Path):
    """Load and validate dataset"""
    print(f"📂 Loading data from: {data_path}")
    
    if not data_path.exists():
        raise FileNotFoundError(f"❌ Data file not found: {data_path}")
    
    try:
        df = pd.read_csv(data_path)
        print(f"✅ Data loaded successfully. Shape: {df.shape}")
    except Exception as e:
        raise ValueError(f"❌ Error reading CSV file: {e}")
    
    # Validate required columns
    if "Outcome" not in df.columns:
        raise ValueError("❌ Target column 'Outcome' not found in dataset")
    
    # Check for missing values
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        print(f"⚠️ Warning: Dataset contains {missing_values} missing values")
    
    # Prepare features and target
    X = df.drop(columns=["Outcome"]).values
    y = df["Outcome"].astype(int).values
    
    print(f"📊 Features shape: {X.shape}, Target shape: {y.shape}")
    print(f"📈 Target distribution: {dict(zip(['Non-Diabetes', 'Diabetes'], np.bincount(y)))}")
    
    return X, y, df


def train_model(X_train, y_train):
    """Train logistic regression model"""
    print("🔄 Starting model training...")
    
    model = LogisticRegression(
        max_iter=1000, 
        random_state=42,
        solver='liblinear'  # More stable for small datasets
    )
    model.fit(X_train, y_train)
    
    print("✅ Model training completed")
    return model


def evaluate_model(model, X_test, y_test):
    """Comprehensive model evaluation"""
    print("📊 Evaluating model performance...")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    print("📈 Model Performance:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name.capitalize()}: {metric_value:.4f}")
    
    return metrics, y_pred, y_pred_proba


def create_visualizations(y_test, y_pred, model, X):
    """Create and save model visualization artifacts"""
    artifacts = []
    
    try:
        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cbar=True, cmap='Blues',
                   xticklabels=['Non-Diabetes', 'Diabetes'],
                   yticklabels=['Non-Diabetes', 'Diabetes'])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        
        cm_path = Path("confusion_matrix.png")
        plt.tight_layout()
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        artifacts.append(str(cm_path))
        print(f"✅ Confusion matrix saved: {cm_path}")
        
    except Exception as e:
        print(f"⚠️ Warning: Could not create confusion matrix: {e}")
    
    try:
        # 2. Feature Importance
        if hasattr(model, 'coef_'):
            feature_names = [f"Feature_{i+1}" for i in range(X.shape[1])]
            importance = np.abs(model.coef_[0])
            
            plt.figure(figsize=(10, 6))
            indices = np.argsort(importance)[::-1]
            plt.bar(range(len(importance)), importance[indices])
            plt.xlabel("Feature Index")
            plt.ylabel("Importance (|Coefficient|)")
            plt.title("Feature Importance")
            plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)
            
            importance_path = Path("feature_importance.png")
            plt.tight_layout()
            plt.savefig(importance_path, dpi=150, bbox_inches='tight')
            plt.close()
            artifacts.append(str(importance_path))
            print(f"✅ Feature importance plot saved: {importance_path}")
            
    except Exception as e:
        print(f"⚠️ Warning: Could not create feature importance plot: {e}")
    
    return artifacts


def train(data_path: Path, min_accuracy: float = 0.70):
    """
    Main training function with comprehensive MLflow tracking
    """
    print("🚀 Starting diabetes prediction model training")
    print("=" * 60)
    
    # Setup MLflow
    experiment = setup_mlflow()
    
    # Load and validate data
    X, y, df = load_and_validate_data(data_path)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"📋 Data split - Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    # Configure MLflow autolog
    mlflow.sklearn.autolog(
        log_model_signatures=True,
        log_input_examples=True,
        log_models=True,
        disable=False,
        exclusive=False,
        disable_for_unsupported_versions=False,
        silent=True
    )
    
    # Start MLflow run
    run_name = f"logreg_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name):
        print(f"🔄 Started MLflow run: {run_name}")
        
        # Log parameters
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("solver", "liblinear")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("min_accuracy_threshold", min_accuracy)
        mlflow.log_param("dataset_shape", f"{df.shape[0]}x{df.shape[1]}")
        mlflow.log_param("feature_count", X.shape[1])
        
        # Train model
        model = train_model(X_train, y_train)
        
        # Evaluate model
        metrics, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test)
        
        # Log metrics to MLflow
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log additional info
        mlflow.log_metric("train_samples", len(X_train))
        mlflow.log_metric("test_samples", len(X_test))
        mlflow.log_metric("n_features", X.shape[1])
        
        # Create and log visualizations
        try:
            artifacts = create_visualizations(y_test, y_pred, model, X)
            for artifact_path in artifacts:
                if Path(artifact_path).exists():
                    try:
                        mlflow.log_artifact(artifact_path)
                        print(f"✅ Artifact logged: {artifact_path}")
                        # Clean up local file
                        Path(artifact_path).unlink()
                    except Exception as e:
                        print(f"⚠️ Warning: Could not log artifact {artifact_path}: {e}")
                        # Try to clean up anyway
                        if Path(artifact_path).exists():
                            Path(artifact_path).unlink()
        except Exception as e:
            print(f"⚠️ Warning: Could not create visualizations: {e}")
        
        # Log classification report as text
        try:
            report = classification_report(y_test, y_pred, 
                                         target_names=['Non-Diabetes', 'Diabetes'])
            report_path = Path("classification_report.txt")
            with open(report_path, 'w') as f:
                f.write(report)
            
            try:
                mlflow.log_artifact(str(report_path))
                print("✅ Classification report logged")
            except Exception as e:
                print(f"⚠️ Warning: Could not log classification report: {e}")
            finally:
                # Always clean up
                if report_path.exists():
                    report_path.unlink()
                    
        except Exception as e:
            print(f"⚠️ Warning: Could not create classification report: {e}")
        
        # Model validation
        accuracy = metrics['accuracy']
        if accuracy >= min_accuracy:
            validation_status = "PASSED"
            print(f"✅ Model validation PASSED (accuracy {accuracy:.4f} >= {min_accuracy})")
        else:
            validation_status = "FAILED"
            print(f"❌ Model validation FAILED (accuracy {accuracy:.4f} < {min_accuracy})")
        
        mlflow.log_param("validation_status", validation_status)
        mlflow.log_metric("validation_passed", 1 if validation_status == "PASSED" else 0)
        
        # Log run info
        current_run = mlflow.active_run()
        print(f"📝 Run ID: {current_run.info.run_id}")
        print(f"🔬 Experiment ID: {current_run.info.experiment_id}")
        
        # Final summary
        print("=" * 60)
        print("🎉 Training completed successfully!")
        print(f"📊 Final Accuracy: {accuracy:.4f}")
        print(f"🎯 Validation: {validation_status}")
        print(f"📁 MLflow UI: {mlflow.get_tracking_uri()}")
        print("=" * 60)
        
        # Return validation status for CI/CD
        return validation_status == "PASSED"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train diabetes prediction model with MLflow")
    parser.add_argument(
        "--data",
        type=str,
        default="../dataset_preprocessing/diabetes_preprocessed.csv",
        help="Path to CSV file with preprocessed data"
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.70,
        help="Minimum required accuracy for model validation"
    )
    
    args = parser.parse_args()
    
    try:
        success = train(Path(args.data), args.min_accuracy)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)