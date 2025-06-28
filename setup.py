import os
from pathlib import Path

project_name = ""

folders = [
    ".github/workflows",
    "data/raw",
    "data/processed",
    "notebooks",
    "src/api",
    "tests",
]

files = {
    ".gitignore": """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*.so
*.ipynb_checkpoints
.env
data/
mlruns/
""",
    "requirements.txt": """pandas
numpy
scikit-learn
matplotlib
seaborn
mlflow
fastapi
uvicorn
xverse
woe
pytest
flake8
black
jupyter
""",
    "Dockerfile": """FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
""",
    "docker-compose.yml": """version: '3.8'

services:
  credit-risk-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - .:/app
""",
    ".github/workflows/ci.yml": """name: CI

on:
  push:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.10

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt

    - name: Lint with flake8
      run: |
        flake8 src/ --max-line-length=88

    - name: Run tests
      run: |
        pytest tests/
""",
    "notebooks/1.0-eda.ipynb": "",
    "src/__init__.py": "",
    "src/data_processing.py": "# Data processing and feature engineering functions\n",
    "src/train.py": "# Training script for credit risk model\n",
    "src/predict.py": "# Inference script to use trained model for predictions\n",
    "src/api/main.py": """from fastapi import FastAPI
from .pydantic_models import CustomerData

app = FastAPI()

@app.get("/")
def home():
    return {"status": "Credit Risk API running"}

@app.post("/predict")
def predict_risk(data: CustomerData):
    # Dummy return for now
    return {"risk_score": 0.5}
""",
    "src/api/pydantic_models.py": """from pydantic import BaseModel

class CustomerData(BaseModel):
    customer_id: str
    transaction_count: int
    avg_transaction_amount: float
    frequency: float
    recency: int
    monetary: float
""",
    "tests/test_data_processing.py": """import pytest

def test_dummy():
    assert 1 + 1 == 2
""",
}

print(f"📁 Creating project: {project_name}")
Path(project_name).mkdir(exist_ok=True)

for folder in folders:
    path = Path(project_name) / folder
    path.mkdir(parents=True, exist_ok=True)
    print(f"📂 Created folder: {path}")

for filename, content in files.items():
    path = Path(project_name) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"📝 Created file: {path}")

print("\n✅ Project initialized successfully (without README).")
