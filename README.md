# 🚀 E-Commerce Order Status Classifier (TP2 MLOps)

Welcome to **JOUR 2** of the *Industrialisation de l'IA dans le Cloud* curriculum. 

This repository contains a fully automated, production-grade **MLOps Pipeline** designed to train, track, test, and deploy a Machine Learning model that predicts the status of e-commerce orders (`completed`, `pending`, `cancelled`, `returned`).

---

## 🏗️ System Architecture & Workflow

This project implements a complete CI/CD lifecycle for Machine Learning using **MLflow** and **GitHub Actions**.

### 1. Training & Experiment Tracking (`train.py`)
- We evaluate three distinct algorithms: **Random Forest**, **XGBoost**, and **SVM**.
- **MLflow Tracking** automatically logs hyper-parameters, metrics (`accuracy`, `precision`, `recall`, `f1_score`), and artifacts (Confusion Matrices).

### 2. Model Registry
- The script mathematically determines the **Best Model** based on accuracy.
- Only the winning model is registered into the **MLflow Model Registry**.
- The model is automatically tagged and transitioned to the **Staging** environment, keeping our registry clean and production-ready.

### 3. Automated Testing (`tests/`)
Before any deployment, the code must pass strict quality gates using `pytest`:
- **Unit Tests (`test_train.py`)**: Validates the data engineering pipeline (handling nulls, correct data types, expected columns).
- **Integration Tests (`test_model.py`)**: Queries the MLflow Tracking Server to ensure the registered model exceeds the required business threshold (Accuracy > 70%).

### 4. Continuous Integration / Continuous Deployment (CI/CD)
- Orchestrated via **GitHub Actions** (`.github/workflows/ml-pipeline.yml`).
- Triggered automatically `on: push` to the `main` branch.
- The pipeline spins up an Ubuntu runner, installs dependencies, sets up a local MLflow server, runs tests, executes training, and simulates deployment.

### 5. Canary Deployment Simulation (`deploy_canary.py`)
Instead of a risky "Big Bang" release, we simulate a **Canary Rollout Strategy**:
- Traffic is incrementally routed to the new model: **5% → 25% → 50% → 100%**.
- During each phase, "live" metrics are monitored.
- **Rollback Mechanism**: If accuracy drops below the threshold, the model is instantly aborted and marked as `Archived`.
- **Promotion**: If the model survives the 100% traffic phase, it is promoted from `Staging` to `Production` in the MLflow Model Registry.

---

## 🛠️ Tech Stack
- **Python 3.10+**
- **Scikit-Learn & XGBoost**: Machine Learning models
- **MLflow**: Experiment tracking and Model Registry
- **Pytest**: Unit and Integration testing
- **GitHub Actions**: CI/CD Orchestration
- **Pandas**: Data manipulation

---

## 🚀 How to Run Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the MLflow Server
Open a separate terminal and run:
```bash
python -m mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri "sqlite:///mlflow.db" --default-artifact-root "mlruns"
```

### 3. Run the Training Pipeline
```bash
python train.py
```
*You can view the results by opening `http://localhost:5000` in your browser.*

### 4. Run the Test Suite
```bash
pytest -v
```

### 5. Simulate the Canary Deployment
```bash
python deploy_canary.py
```
