import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from fetch_data import fetch_clean_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data():
    df = fetch_clean_data()

    X = df.drop('outcome', axis=1)
    y = df['outcome']
    
    X = X.fillna(X.mean())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    smote = SMOTE(random_state=0)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train_resampled, y_test

if __name__ == "__main__":
   X_train_scaled, X_test_scaled, y_train_resampled, y_test = preprocess_data()
   print("Train shape :", X_train_scaled.shape)
   print("Test shape :", X_test_scaled.shape)