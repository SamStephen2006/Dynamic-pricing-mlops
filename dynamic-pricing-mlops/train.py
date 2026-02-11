import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Config
DATA_PATH = "data/gig_pricing.csv"
EXPERIMENT_NAME = "DoItForMe_Pricing_Engine"

# 2. Setup MLflow Experiment
mlflow.set_experiment(EXPERIMENT_NAME)

def main():
    with mlflow.start_run():
        print("Loading Data...")
        df = pd.read_csv(DATA_PATH)
        
        # Split Features (X) and Target (y)
        X = df.drop(columns=['surge_multiplier'])
        y = df['surge_multiplier']
        
        # 3. Build the Preprocessing Pipeline
        # We need to turn "weather" (Categorical) into numbers
        # We need to scale "workers" (Numerical) so it matches other features
        categorical_features = ['weather', 'exam_season']
        numerical_features = ['hour', 'workers']
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
        
        # 4. Define the Model (XGBoost)
        # We are using a Pipeline: Raw Data -> Preprocessor -> Model
        params = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 5
        }
        
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', xgb.XGBRegressor(**params))
        ])
        
        # 5. Train
        print(" Training Pricing Model...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model_pipeline.fit(X_train, y_train)
        
        # 6. Evaluate
        predictions = model_pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print(f"Model Performance: MAE={mae:.4f}, R2={r2:.4f}")
        
        # --- YOUR TASK: FILL IN THE MISSING MLFLOW LOGGING ---
        
        # Log the parameters "n_estimators" and "learning_rate"
        mlflow.log_param("n_estimators", params["n_estimators"])
        
        # <--- WRITE THE CODE TO LOG 'learning_rate' HERE
        mlflow.log_param("learning_rate", params["learning_rate"])
        
        # Log the metrics MAE and R2
        mlflow.log_metric("mae", mae)
        # <--- WRITE THE CODE TO LOG 'r2' HERE
        
        mlflow.log_metric("r2", r2)
        
        # Log the entire model pipeline so we can reuse it later
        mlflow.sklearn.log_model(model_pipeline, "model")
        # Hint: Use mlflow.sklearn.log_model(object, name)
        # <--- WRITE THE CODE TO LOG 'model_pipeline' AS "model"
        
        print(" Experiment logged to MLflow UI")

if __name__ == "__main__":
    main()