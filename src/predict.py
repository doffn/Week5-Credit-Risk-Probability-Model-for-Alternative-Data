import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
)
import mlflow
import mlflow.sklearn
import joblib


class ModelTrainer:
    def __init__(self, agg_df, rfm_df, label_col='is_high_risk', id_col='CustomerId'):
        # Data
        self.agg_df = agg_df
        self.rfm_df = rfm_df
        self.label_col = label_col
        self.id_col = id_col
        
        # Prepare data on init
        self.X, self.y = self._prepare_data()
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()

        # Scale and impute full dataset upfront
        X_imputed = self.imputer.fit_transform(self.X)
        self.X_scaled = self.scaler.fit_transform(X_imputed)
        
        # Train/test split fixed for reproducibility
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, self.y, stratify=self.y, test_size=0.2, random_state=42
        )

        # Store results and best model info
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.best_f1 = -1

    def _prepare_data(self):
        merged = pd.merge(self.agg_df, self.rfm_df[[self.id_col, self.label_col]], on=self.id_col)
        X = merged.drop([self.id_col, self.label_col], axis=1)
        y = merged[self.label_col]
        return X, y

    def train_models(self, models_dict):
        for name, model in models_dict.items():
            with mlflow.start_run(run_name=name, nested=True):
                print(f"\nTraining {name}...")
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                report = classification_report(self.y_test, y_pred, output_dict=True)
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred)
                recall = recall_score(self.y_test, y_pred)
                f1 = f1_score(self.y_test, y_pred)

                print(f"--- {name} Evaluation ---")
                print(classification_report(self.y_test, y_pred))
                print("-" * 40)

                # Save results
                self.results[name] = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "report": report,
                    "model": model,
                    "y_pred": y_pred
                }

                # Log metrics and model
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_text(str(report), "classification_report.txt")
                mlflow.sklearn.log_model(model, name.replace(" ", "_").lower() + "_model")

                # Track best model by F1-score
                if f1 > self.best_f1:
                    self.best_f1 = f1
                    self.best_model = model
                    self.best_model_name = name

    def predict(self, features: list):
        """Predict on new single sample (features must be raw, unscaled)."""
        X_arr = np.array(features).reshape(1, -1)
        X_imputed = self.imputer.transform(X_arr)
        X_scaled = self.scaler.transform(X_imputed)
        pred = self.best_model.predict(X_scaled)
        proba = self.best_model.predict_proba(X_scaled)[:, 1] if hasattr(self.best_model, "predict_proba") else None
        return {
            "prediction": int(pred[0]),
            "probability": float(proba[0]) if proba is not None else None,
        }

    def save_best_model(self, filepath="best_model.pkl"):
        if self.best_model:
            joblib.dump(self.best_model, filepath)
            print(f"Saved best model '{self.best_model_name}' to '{filepath}'")
        else:
            print("No model trained yet to save.")

    def plot_confusion_matrix(self, model_name=None):
        if model_name is None:
            model_name = self.best_model_name
        if model_name not in self.results:
            print(f"Model '{model_name}' results not found.")
            return

        y_pred = self.results[model_name]["y_pred"]
        disp = ConfusionMatrixDisplay.from_predictions(self.y_test, y_pred, cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix - {model_name}")
        plt.show()

    def plot_feature_importance(self, model_name=None):
        if model_name is None:
            model_name = self.best_model_name
        if model_name not in self.results:
            print(f"Model '{model_name}' results not found.")
            return

        model = self.results[model_name]["model"]
        if not hasattr(model, "feature_importances_"):
            print(f"Model '{model_name}' does not have feature_importances_ attribute.")
            return

        importances = model.feature_importances_
        feature_names = self.X.columns
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title(f"Feature Importances - {model_name}")
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), feature_names[indices], rotation=90)
        plt.tight_layout()
        plt.show()
