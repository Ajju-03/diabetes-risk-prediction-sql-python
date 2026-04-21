import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5,
    "target_column": "outcome",
    "zero_invalid_cols": [
    "glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"],

    "db_path": "diabates_db.db",
    "db_url": "mysql+pymysql://root:Ajju0323@localhost/medical_db",
    "table_name": "diabetes_data",

    "data_csv": "extracted_data/diabetes.csv",
    "output_dir": "./outputs",
    "preprocessor_path": "preprocessor.pkl",

    "experiment_name": "Diabetes_Risk_Prediction",
    "registry_name": "DiabetesRiskModel",
    "tracking_uri": "sqlite:///mlflow.db",

    "models": {
        "LogisticRegression": lambda: LogisticRegression(
            max_iter=2000, random_state=42, C=1.0, solver='lbfgs'
        ),
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=100, max_depth=6, random_state=42
        ),
        "GradientBoosting": lambda: GradientBoostingClassifier(
            n_estimators = 100, learning_rate = 0.1, random_state = 42
        ),
        "SupportVectorMachines": lambda: SVC(
            kernel="rbf", probability=True, random_state=42, C=1.0 
        )
    }
}