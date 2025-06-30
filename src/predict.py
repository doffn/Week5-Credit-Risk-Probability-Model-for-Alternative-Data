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
