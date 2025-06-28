# Credit Risk Model

This repository contains an end-to-end implementation of a Credit Risk Scoring model using alternative data provided by an eCommerce platform.

## 📁 Project Structure

```bash
credit-risk-model/
├── .github/workflows/ci.yml   # CI/CD workflow
├── data/                      # Raw and processed data
├── notebooks/                 # Exploratory notebooks
├── src/                       # Core source code
│   ├── data_processing.py     # Feature engineering
│   ├── train.py               # Training logic
│   ├── predict.py             # Inference logic
│   └── api/
│       ├── main.py            # FastAPI app
│       └── pydantic_models.py # Request schema
├── tests/                     # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
````

## 🚀 Credit Scoring Business Understanding

### 1. Basel II Influence on Model Requirements

The Basel II Accord emphasizes risk measurement and transparency, encouraging interpretable models for auditability and regulatory compliance. This impacts how financial models must be designed, explained, and validated.

### 2. Need for Proxy Variables

Without a direct "default" label, a proxy (e.g., based on RFM behavior) helps us simulate real-world risk patterns. However, mislabeling could lead to biased predictions, which can harm both the business and its customers.

### 3. Model Trade-Offs in Regulated Environments

* **Simple models (e.g., Logistic Regression + WoE)**: Interpretable, regulatory-friendly.
* **Complex models (e.g., XGBoost)**: Higher performance, but harder to justify in regulated settings.

A balance between performance and transparency is crucial.

## 🔧 Features

* Behavioral clustering using RFM
* Proxy target engineering
* Model training with MLflow
* FastAPI for deployment
* Docker containerization
* CI/CD with GitHub Actions

## 📜 License

MIT or as per 10 Academy’s project requirement.

# Week5-Credit-Risk-Probability-Model-for-Alternative-Data
