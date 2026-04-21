import os

from sqlalchemy import create_engine, text, inspect
import logging
import pandas as pd
from config import CONFIG

# -------------------------------- Setup Logging ---------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------- Database Manager ---------------------------- #
class DatabaseManager:
    def __init__(self):
        url = CONFIG.get("db_url")
        self.engine = create_engine(url)

    def load_data(self):
        logger.info("Loading data from CSV...")
        df = pd.read_csv(CONFIG["data_csv"])
        self.engine = create_engine(CONFIG.get("db_url"))
        df = df.rename(columns={
            "Pregnancies":              "pregnancies",
            "Glucose":                  "glucose",
            "BloodPressure":            "blood_pressure",
            "SkinThickness":            "skin_thickness",
            "Insulin":                  "insulin",
            "BMI":                      "bmi",
            "DiabetesPedigreeFunction": "diabetes_pedigree",
            "Age":                      "age",
            "Outcome":                  "outcome",
        })
        insp = inspect(self.engine)
        if insp.has_table(CONFIG["table_name"]):
            with self.engine.connect() as conn:
                count = conn.execute(text(f"SELECT COUNT(*) FROM {CONFIG['table_name']}")).scalar()
                if count > 0:
                    logger.info("Data already exists in MySQL, skipping insert.")
                    return pd.read_sql(f"SELECT * FROM {CONFIG['table_name']}", self.engine)
        df.to_sql(CONFIG.get("table_name"), con=self.engine, if_exists="append", index=False)
        logger.info("CSV data inserted into MySQL.") 
        return df   
    
    # save predicted values into predictions table in db
    def save_predictions(self, patient_id, model_name, 
                         model_version, risk_score, prediction):
        risk_label = (
            "HIGH RISK" if risk_score >= 0.7 else
            "MEDIUM RISK" if risk_score >= 0.4 else
            "LOW RISK"
        )
        query = text("""
            INSERT INTO predictions
                (patient_id, model_name, model_version, risk_score, prediction, risk_label)
            VALUES
                (:pid, :mname, :mver, :rscore, :pred, :label)
        """)
        with self.engine.connect() as conn:
            conn.execute(query, {
                "pid": patient_id,
                "mname": model_name,
                "mver": model_version,
                "rscore": risk_score,
                "pred": prediction,
                "label": risk_label
            })
            conn.commit()

    # save values into experiment_log table in db
    def log_experiment(self, run_id, model_name, accuracy, f1, roc_auc):

        query = text("""
            INSERT INTO experiment_log
                (mlflow_run_id, model_name, accuracy, f1_score, roc_auc)
            VALUES
                (:run_id, :model_name, :accuracy, :f1, :roc_auc)         
        """) 
        with self.engine.connect() as conn:
            conn.execute(query, {
                "run_id": run_id,
                "model_name": model_name,
                "accuracy": accuracy,
                "f1": f1,
                "roc_auc": roc_auc 
            })
            conn.commit()

    # get best model from the database
    def get_best_model(self):

        query = text("""
            SELECT model_name, roc_auc, accuracy, f1_score, mlflow_run_id
            FROM experiment_log
            ORDER BY roc_auc DESC
            LIMIT 1;                           
        """)
        df = pd.read_sql(query, self.engine)
        if df.empty:
            return None
        return df.iloc[0].to_dict()
                          
