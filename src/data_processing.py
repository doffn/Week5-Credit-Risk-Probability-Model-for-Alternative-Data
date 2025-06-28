import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans



class DataProcessing:
    """
    Credit Risk Data Processing Class for EDA, Feature Engineering, and Target Creation.

    This class encapsulates the steps for preparing the customer transaction data
    for credit risk modeling, including:
    1. Exploratory Data Analysis (EDA)
    2. Feature Engineering (time-based and aggregate features)
    3. Proxy Target Variable Creation using RFM analysis

    Business Understanding Notes (Task 1):
    - Basel II requires interpretable models, favoring Logistic Regression with WoE.
    - A proxy for 'default' is created using RFM due to lack of explicit labels.
    - There's a trade-off between model interpretability (Logistic Regression) and
      potential higher performance (Gradient Boosting).
    """

    def __init__(self, df):
        """
        Initializes the DataProcessing class with the input DataFrame.

        Args:
            df (pd.DataFrame): The raw input transaction DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        self.df = df.copy()
        self.agg = None
        self.rfm = None
        self.preprocessor = None
        self.X = None
        self.y = None
        self._validate_initial_columns()

    def _validate_initial_columns(self):
        """Ensures essential columns are present in the initial DataFrame."""
        required_cols = ['CustomerId', 'TransactionStartTime', 'Amount', 'Value', 'TransactionId', 'ProductCategory']
        if not all(col in self.df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in self.df.columns]
            raise ValueError(f"Input DataFrame must contain columns: {required_cols}. Missing: {missing}")

    # ---------- Task 2: EDA ----------
    def perform_eda(self):
        """
        Performs exploratory data analysis and prints key insights.
        Includes basic info, missing values, summary stats, and plots.
        """
        print("--- Performing EDA ---")
        print("\nDataset Info:")
        self.df.info()
        print("\nSummary Statistics:")
        print(self.df.describe(include='all')) # include='all' for categorical summary too
        print("\nMissing Values:")
        print(self.df.isnull().sum())

        self._plot_distributions()
        self._plot_categorical_counts()
        self._plot_correlation_matrix()
        self._plot_outliers()
        print("--- EDA Complete ---")

    def _plot_distributions(self):
        """Plots histograms for numerical columns."""
        num_cols = self.df.select_dtypes(include=np.number).columns
        if not num_cols.empty:
            print("\nPlotting Numerical Distributions:")
            self.df[num_cols].hist(figsize=(15, min(len(num_cols)//3 * 4, 10)), bins=30)
            plt.tight_layout()
            plt.suptitle('Distribution of Numerical Features', y=1.02)
            plt.show()
        else:
            print("\nNo numerical columns to plot distributions.")

    def _plot_categorical_counts(self, top_n=10):
        """Prints value counts and plots bar charts for categorical columns."""
        cat_cols = self.df.select_dtypes(include='object').columns
        if not cat_cols.empty:
            print(f"\nPlotting Top {top_n} Categorical Value Counts:")
            for col in cat_cols:
                print(f"\n{col} value counts (Top {top_n}):")
                print(self.df[col].value_counts().head(top_n))
                plt.figure(figsize=(10, 6))
                sns.countplot(data=self.df, y=col, order=self.df[col].value_counts().index[:top_n], palette='viridis')
                plt.title(f"Distribution of {col} (Top {top_n})")
                plt.xlabel("Count")
                plt.ylabel(col)
                plt.show()
        else:
             print("\nNo categorical columns to plot counts.")


    def _plot_correlation_matrix(self):
        """Plots a heatmap of the correlation matrix for numerical columns."""
        num_cols = self.df.select_dtypes(include=np.number).columns
        if len(num_cols) > 1: # Need at least 2 numerical columns for correlation
            print("\nPlotting Correlation Matrix:")
            corr = self.df[num_cols].corr()
            plt.figure(figsize=(min(len(num_cols)*0.8, 12), min(len(num_cols)*0.8, 10)))
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
            plt.title("Correlation Matrix")
            plt.show()
        else:
            print("\nNot enough numerical columns to plot correlation matrix.")

    def _plot_outliers(self):
        """Plots box plots for numerical columns to visualize outliers."""
        num_cols = self.df.select_dtypes(include=np.number).columns
        if not num_cols.empty:
            print("\nPlotting Box Plots for Outlier Detection:")
            n_cols_plot = 3
            n_rows_plot = (len(num_cols) + n_cols_plot - 1) // n_cols_plot
            fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(15, n_rows_plot * 4))
            axes = axes.flatten()

            for i, col in enumerate(num_cols):
                sns.boxplot(x=self.df[col], ax=axes[i], palette='viridis')
                axes[i].set_title(f'Box Plot of {col}')
                axes[i].set_xlabel(col)

            # Hide any unused subplots
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()
            plt.suptitle('Outlier Visualization using Box Plots', y=1.02)
            plt.show()
        else:
             print("\nNo numerical columns to plot outliers.")


    # ---------- Task 3: Feature Engineering ----------
    def perform_feature_engineering(self):
        """
        Engineers time-based features and aggregate features per customer.
        Builds and fits the preprocessing pipeline.

        Returns:
            np.ndarray: The preprocessed feature matrix X.
        """
        print("--- Performing Feature Engineering ---")
        # Extract time features
        self.df['TransactionStartTime'] = pd.to_datetime(self.df['TransactionStartTime'])
        self.df['hour'] = self.df['TransactionStartTime'].dt.hour
        self.df['month'] = self.df['TransactionStartTime'].dt.month

        # Aggregate features per customer
        agg = self.df.groupby('CustomerId').agg(
            total_amt=('Amount', 'sum'),
            avg_amt=('Amount', 'mean'),
            txn_count=('Amount', 'count'),
            std_val=('Value', lambda x: x.std(skipna=True)) # Handle potential NaNs in Value
        ).reset_index()

        # Handle potential NaNs from std_val where count is 1 or less
        agg['std_val'] = agg['std_val'].fillna(0)


        # Merge categorical features (take the most frequent category per customer)
        # Or simply the first, depending on business logic. Let's take the most frequent.
        # If a customer has multiple categories, take the one they appeared in most often.
        # If there's a tie, take the first one in the sorted value_counts index.
        cat = self.df.groupby('CustomerId')['ProductCategory'].agg(lambda x: x.mode()[0] if not x.mode().empty else 'Unknown').reset_index()

        self.agg = agg.merge(cat, on='CustomerId', how='left')

        # Define features for pipeline
        # Use dynamically determined column names for robustness
        numerical_agg_features = [col for col in self.agg.columns if col not in ['CustomerId', 'ProductCategory']]
        categorical_agg_features = ['ProductCategory']

        # Build preprocessing pipeline
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline([
                    ("imputer", SimpleImputer(strategy='mean')),
                    ("scaler", StandardScaler())]), numerical_agg_features),
                ("cat", Pipeline([
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_agg_features)
            ],
            remainder='passthrough' # Keep other columns if any (though none expected here)
        )

        # Fit-transform preprocessing pipeline on the aggregated data
        self.X = self.preprocessor.fit_transform(self.agg.drop(columns=['CustomerId']))

        print("Feature Engineering Complete.")
        print(f"Shape of engineered features (X): {self.X.shape}")
        return self.X

    # ---------- Task 4: Proxy Target via RFM ----------
    def create_rfm_proxy_target(self, n_clusters=3):
        """
        Computes RFM metrics, clusters customers, and creates a high-risk proxy target.

        Args:
            n_clusters (int): The number of clusters for KMeans.

        Returns:
            pd.DataFrame: DataFrame with CustomerId and 'is_high_risk' flag.
        """
        print(f"--- Creating RFM Proxy Target with {n_clusters} Clusters ---")
        if 'TransactionStartTime' not in self.df.columns or not pd.api.types.is_datetime64_any_dtype(self.df['TransactionStartTime']):
             # Ensure TransactionStartTime is datetime, handle if feature_engineering wasn't run first
             self.df['TransactionStartTime'] = pd.to_datetime(self.df['TransactionStartTime'])


        # Snapshot date for recency calculation (day after last transaction)
        snapshot_date = self.df['TransactionStartTime'].max() + pd.Timedelta(days=1)

        # Calculate RFM metrics
        rfm = self.df.groupby('CustomerId').agg(
            Recency=('TransactionStartTime', lambda x: (snapshot_date - x.max()).days),
            Frequency=('TransactionId', 'count'),
            Monetary=('Amount', 'sum')
        ).reset_index()

        # Handle potential issues with customers having only one transaction (Recency = 0)
        # Recency of 0 is fine, but ensure no infinities or NaNs if snapshot calculation failed
        rfm['Recency'] = rfm['Recency'].clip(lower=0) # Recency cannot be negative

        # Scale RFM for clustering
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

        # KMeans clustering
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # Added n_init
            rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
        except ValueError as e:
             print(f"Could not perform KMeans clustering. Error: {e}")
             print(f"RFM DataFrame head:\n{rfm.head()}")
             print(f"Scaled RFM head:\n{pd.DataFrame(rfm_scaled, columns=['Recency_scaled', 'Frequency_scaled', 'Monetary_scaled']).head()}")
             print(f"Number of customers: {len(rfm)}")
             print(f"Number of non-null scaled RFM rows: {np.sum(np.isfinite(rfm_scaled).all(axis=1))}")
             self.rfm = rfm[['CustomerId']] # Return RFM without cluster/risk if clustering fails
             print("RFM proxy target creation failed.")
             return self.rfm

        # Identify high-risk cluster (e.g., lowest frequency, could be highest recency or lowest monetary)
        # Using lowest frequency as per the original code's logic
        cluster_means = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
        low_engagement_cluster = cluster_means['Frequency'].idxmin()
        print(f"\nCluster means:\n{cluster_means}")
        print(f"Identified low engagement cluster (by lowest frequency): {low_engagement_cluster}")


        # Label high-risk customers
        rfm['is_high_risk'] = (rfm['Cluster'] == low_engagement_cluster).astype(int)

        print("\nRFM Metrics with Cluster and Risk Flag:")
        display(rfm.head())

        # Store only CustomerId and the target variable
        self.rfm = rfm[['CustomerId', 'is_high_risk']]

        print("RFM Proxy Target Creation Complete.")
        return self.rfm

    def visualize_rfm_analysis(self):
        """
        Provides visualizations for the RFM analysis and the proxy target distribution.
        Requires compute_rfm_labels to have been run successfully.
        """
        if self.rfm is None or 'Cluster' not in self.rfm.columns:
            print("RFM analysis not available for visualization. Run create_rfm_proxy_target() first.")
            return

        print("--- Visualizing RFM Analysis ---")

        # Merge RFM metrics back for visualization
        if 'Recency' not in self.rfm.columns:
             # Need to re-calculate RFM metrics for visualization if they weren't stored
             snapshot_date = self.df['TransactionStartTime'].max() + pd.Timedelta(days=1)
             rfm_full = self.df.groupby('CustomerId').agg(
                 Recency=('TransactionStartTime', lambda x: (snapshot_date - x.max()).days),
                 Frequency=('TransactionId', 'count'),
                 Monetary=('Amount', 'sum')
             ).reset_index()
             rfm_full['Recency'] = rfm_full['Recency'].clip(lower=0)
             viz_df = rfm_full.merge(self.rfm[['CustomerId', 'Cluster', 'is_high_risk']], on='CustomerId', how='left')
        else:
             # If RFM metrics were stored on self.rfm (e.g., if compute_rfm_labels stored the full RFM df)
             viz_df = self.rfm.copy()


        # 1. Visualize RFM distributions
        rfm_metrics = ['Recency', 'Frequency', 'Monetary']
        if all(col in viz_df.columns for col in rfm_metrics):
            print("Plotting RFM Distributions:")
            viz_df[rfm_metrics].hist(figsize=(15, 5), bins=30)
            plt.tight_layout()
            plt.suptitle('Distribution of RFM Metrics')
            plt.show()
        else:
            print("RFM metric columns not found for distribution plot.")


        # 2. Visualize RFM clusters (Pairwise scatter plots using scaled data)
        try:
            scaler = StandardScaler()
            rfm_scaled_viz = scaler.fit_transform(viz_df[rfm_metrics])
            rfm_scaled_viz_df = pd.DataFrame(rfm_scaled_viz, columns=[f'{col}_scaled' for col in rfm_metrics])
            rfm_scaled_viz_df['Cluster'] = viz_df['Cluster']

            print("\nPlotting Pairwise Scatter Plots of Scaled RFM Metrics by Cluster:")
            g = sns.pairplot(
                rfm_scaled_viz_df,
                hue='Cluster',
                palette='coolwarm',
                vars=[f'{col}_scaled' for col in rfm_metrics],
                height=3,
                aspect=1
            )
            g.fig.suptitle('Pairwise Scatter Plots of Scaled RFM Metrics by Cluster', y=1.02)
            plt.show()
        except Exception as e:
             print(f"Could not plot RFM pairwise scatter plots. Ensure RFM metrics and Cluster are available. Error: {e}")


        # 3. Visualize the number of customers in each cluster
        if 'Cluster' in viz_df.columns:
            print("\nPlotting Number of Customers per Cluster:")
            plt.figure(figsize=(8, 5))
            sns.countplot(data=viz_df, x='Cluster', palette='viridis')
            plt.title('Number of Customers per Cluster')
            plt.xlabel('Cluster')
            plt.ylabel('Count')
            plt.show()
        else:
            print("Cluster information not available for plotting cluster counts.")

        # 4. Visualize the distribution of the high-risk flag
        if 'is_high_risk' in viz_df.columns:
             print("\nPlotting Distribution of High-Risk Customers:")
             plt.figure(figsize=(6, 4))
             sns.countplot(data=viz_df, x='is_high_risk', palette='viridis')
             plt.title('Distribution of High-Risk Customers')
             plt.xlabel('Is High Risk (0=No, 1=Yes)')
             plt.ylabel('Count')
             plt.show()
        else:
             print("High-risk target variable not available for plotting distribution.")

        print("--- RFM Visualization Complete ---")


    def prepare_final_data(self):
        """
        Merges the engineered features (agg) with the proxy target (rfm) and
        applies the final preprocessing steps.

        Returns:
            tuple: A tuple containing the preprocessed feature matrix X and
                   the target vector y (pd.Series).
        Raises:
            ValueError: If feature_engineering() or create_rfm_proxy_target()
                        have not been run successfully.
        """
        print("--- Preparing Final Data ---")
        if self.agg is None:
            raise ValueError("Run perform_feature_engineering() first.")
        if self.rfm is None or 'is_high_risk' not in self.rfm.columns:
            raise ValueError("Run create_rfm_proxy_target() successfully first.")
        if self.preprocessor is None:
             # This should not happen if feature_engineering ran, but as a safeguard
             raise ValueError("Preprocessor not fitted. Run perform_feature_engineering().")


        # Merge aggregated features with the RFM-based target
        # Use the original self.agg which contains CustomerId
        data = self.agg.merge(self.rfm[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')

        # Ensure 'is_high_risk' is not missing for any customers (should not happen with left merge)
        if data['is_high_risk'].isnull().any():
             print("Warning: Missing 'is_high_risk' values after merge. Filling with 0 (assuming not high risk if RFM failed/missing).")
             data['is_high_risk'] = data['is_high_risk'].fillna(0).astype(int)


        self.y = data['is_high_risk']

        # Reapply the fitted preprocessor on the features of the merged dataset
        # Ensure the columns passed to transform match those used during fit
        features_to_transform = self.agg.drop(columns=['CustomerId']).columns # Use column names from the fitted dataframe
        X_data = data[features_to_transform]

        self.X = self.preprocessor.transform(X_data)

        print("Final Data Preparation Complete.")
        print(f"Shape of final features (X): {self.X.shape}")
        print(f"Shape of target (y): {self.y.shape}")
        return self.X, self.y

    def get_preprocessor(self):
        """Returns the fitted preprocessing pipeline."""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Run perform_feature_engineering().")
        return self.preprocessor

    def get_aggregated_features(self):
        """Returns the DataFrame with aggregated features before final preprocessing."""
        if self.agg is None:
             raise ValueError("Aggregated features not created. Run perform_feature_engineering().")
        return self.agg

    def get_rfm_data(self):
        """Returns the DataFrame with CustomerId and the 'is_high_risk' target."""
        if self.rfm is None:
             raise ValueError("RFM proxy target not created. Run create_rfm_proxy_target().")
        return self.rfm

