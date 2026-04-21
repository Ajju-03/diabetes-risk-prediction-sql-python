import os, json
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature
from config import CONFIG
from Database.databasemanager import logger, logging

class MLflowTracker:
    def __init__(self, experiment_name, tracking_uri):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient()
        self.experiment_name = experiment_name


    def log_run(self, model_name, model, params, metrics, feature_names, X_train, artifact_paths):
        with mlflow.start_run(run_name=model_name) as run:
            run_id = run.info.run_id 

            # log hyperparameters
            mlflow.log_params(params)

            # log evaluation metrics
            mlflow.log_metrics(metrics)

            # log all chart images as artifacts
            for path in artifact_paths:
                if path and os.path.exists(path):
                    mlflow.log_artifact(path, artifact_path="charts")

            # log feature names as JSON artifact
            fn_path = f"/tmp/features_{model_name}.json"
            with open (fn_path, "w") as f:
                json.dump({"features": feature_names}, f, indent=2)
            mlflow.log_artifact(fn_path, artifact_path="metadata")      

            signature = infer_signature(X_train, model.predict(X_train))

            mlflow.sklearn.log_model(sk_model=model,
                name="model",
                signature=signature,
                registered_model_name=CONFIG["registry_name"]) 
            
            mlflow.set_tags({
                "model_type": model_name,
                "dataset": "PIMA INDIANS DATASET",
                "features":    str(len(feature_names)),
                "developer": "Portfolio Project",
                "stage": "development"
            })
            logger.info(
                f"  MLflow run logged | ID: {run_id[:8]}... | "
                f"AUC: {metrics.get('roc_auc', 0):.4f}"
            )
        return run_id
    
    def promote_best_model(self):

        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if not experiment:
            logging.warning("No experiment found")
            return None
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.roc_auc DESC"],
            max_results=1
        )
        if runs.empty:
            logging.warning("No runs found")
            return None
        
        best_run = runs.iloc[0]
        best_run_id = best_run["run_id"]
        best_auc = best_run.get("metrics.roc_auc", 0)
        best_name = best_run.get("tags.model_type", "Unknown")

        logger.info(
            f"Best model: {best_name} | "
            f"ROC-AUC: {best_auc:.4f} | "
            f"Run: {best_run_id[:8]}...."
        )

        try: 
            versions = self.client.search_model_versions(
                f"name='{CONFIG['registry_name']}'"
            )

            best_version = None
            for v in versions:
                if v.run_id == best_run_id:
                    best_version = v.version
                    break

            if best_version:
                self.client.set_registered_model_alias(
                    name = CONFIG["registry_name"],
                    alias="Champion",
                    version= best_version
                )  
                logger.info(f"Promoted best version {best_version} to Production")
                return best_run_id
        except Exception as e:
            logger.warning(f"could not promote model: {e}")
            return best_run_id 


    def load_production_model(self):
        uri = f"models:/{CONFIG['registry_name']}@Champion"
        model = mlflow.sklearn.load_model(uri)
        logger.info(f"Loaded production model from: {uri}")
        return model