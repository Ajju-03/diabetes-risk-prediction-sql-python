import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from Database.databasemanager import logger
from config import CONFIG
import pandas as pd
import numpy as np


class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_names = []
        self.is_fitted = False
    
    def clean_data(self, df):
        df = df.copy()
        for col in CONFIG["zero_invalid_cols"]:
           if col in df.columns:
              df[col] = df[col].replace(0, np.nan)
        return df
    
    def feature_engineering(self, df):
        df = df.copy()

        if "glucose" in df.columns and "bmi" in df.columns:
            df["glucose_bmi"] = df["glucose"] * df["bmi"] / 1000

        if "insulin" in df.columns and "glucose" in df.columns:
            df["insulin_glucose_ratio"] = (df["insulin"] / (df["glucose"] + 1)).clip(upper=10)

        if "age" in df.columns:
            df["age_risk"] = pd.cut(
                df["age"],
                bins = [0, 30, 45, 60, 100],
                labels=[0, 1, 2, 3]
            ).astype(float)

        if "bmi" in df.columns:
            df["bmi_category"] = pd.cut(
                df["bmi"],
                bins=[0, 18.5, 25, 30, 100],
                labels=[0, 1, 2, 3]
            ).astype(float)          

        return df
        

    def prepare(self, df):

        df = self.clean_data(df)

        df = self.feature_engineering(df)

        logger.info("Preparing data...")
        target = CONFIG["target_column"]
        X = df.drop(columns=[target, "id", "created_at"], errors="ignore")
        y = df[target]
        self.feature_names = list(X.columns)

        logger.info(f"  Features: {self.feature_names}")
        logger.info(f"  Shape: {X.shape} | Diabetic: {y.sum()} / {len(y)}")

        X_train , X_test, y_train, y_test = train_test_split(X, y, 
            test_size=CONFIG["test_size"],
            random_state=CONFIG["random_state"],
            stratify=y)
        
        # Impute missing values
        X_train = pd.DataFrame(self.imputer.fit_transform(X_train), columns=self.feature_names)
        X_test = pd.DataFrame(self.imputer.transform(X_test), columns=self.feature_names)
        
        # scale features
        X_train = pd.DataFrame(self.scaler.fit_transform(X_train), columns=self.feature_names)
        X_test = pd.DataFrame(self.scaler.transform(X_test), columns=self.feature_names)

        self.is_fitted = True
        logger.info(f"  Train: {X_train.shape} |  Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    

    def save(self, path="preprocessor.pkl"):
        payload = {
            "imputer": self.imputer,
            "scaler": self.scaler,
            "feature_names": self.feature_names
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path="preprocessor.pkl"):
        obj = cls()
        data   = joblib.load(path)
        obj.scaler        = data["scaler"]
        obj.imputer       = data["imputer"]
        obj.feature_names = data["feature_names"]
        obj.is_fitted     = True
        logger.info(f"Preprocessor state loaded from {path}")
        return obj
