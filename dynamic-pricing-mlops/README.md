# 🚀 Dynamic Surge Pricing Engine (MLOps Pipeline)

An end-to-end Machine Learning pipeline designed to predict optimal surge pricing multipliers for a Gig Marketplace (DoItForMe). This project focuses on **Reproducibility**, **Data Versioning**, and **Experiment Tracking**.

## 🛠 Tech Stack
* **Modeling:** XGBoost, Scikit-Learn
* **Data Versioning:** DVC (Data Version Control)
* **Experiment Tracking:** MLflow
* **Automation:** DVC Pipelines (`dvc.yaml`)
* **Language:** Python 3.9+

## 📊 The Problem
In a real-time gig economy, demand fluctuates based on weather, time, and supply. Hard-coded rules fail to capture these dynamics. This system trains an ML model to predict the `surge_multiplier` based on:
* Weather conditions (Rain/Storm)
* Active worker supply
* Time of day (Rush hours)

## ⚙️ How to Run locally

1.  **Clone the Repo**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/dynamic-pricing-mlops.git](https://github.com/YOUR_USERNAME/dynamic-pricing-mlops.git)
    cd dynamic-pricing-mlops
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Reproduce the Pipeline (The Magic Command)**
    This command generates data, preprocesses it, and trains the model using the exact locked versions.
    ```bash
    dvc repro
    ```

4.  **View Experiments**
    ```bash
    mlflow ui
    ```
    Open `http://localhost:5000` to see metrics (MAE, R2) and model artifacts.

## 🔄 MLOps Workflow
1.  **Data Generation:** Simulates realistic market conditions.
2.  **Versioning:** `dvc.lock` ensures that every model training run is linked to a specific hash of the dataset.
3.  **Tracking:** All hyperparameters (learning rate, trees) and metrics are logged to MLflow for comparison.