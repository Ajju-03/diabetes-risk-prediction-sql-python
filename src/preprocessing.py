import pandas as pd
import numpy as np
from fetch_data import fetch_clean_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data():
    df = fetch_clean_data()

    X = df.drop('outcome', axis=1)
    y = df['outcome']
    
    X = X.fillna(X.mean())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
   X_train, X_test, y_train, y_test = preprocess_data()
   print("Train shape :", X_train.shape)
   print("Test shape :", X_test.shape)