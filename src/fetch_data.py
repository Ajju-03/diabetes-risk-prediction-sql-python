import pandas as pd
from db_connection import get_engine

def fetch_clean_data():
    engine = get_engine()
    query = "SELECT * FROM clean_diabetes_data"
    df = pd.read_sql(query, engine)
    return df

if __name__ == "__main__":
    df = fetch_clean_data()
    print(df.head())
    print(df.isnull().sum())