import os
from config import CONFIG
from Database.databasemanager import DatabaseManager, logging, logger
from core.datapreprocessor import DataPreprocessor
from core.modelevaluator import ModelEvaluator
from core.mlflowtracker import MLflowTracker
import pandas as pd


class DiabetesPredictionPipeline:
    def __init__(self):
        os.makedirs(CONFIG["output_dir"], exist_ok=True)

        self.db = DatabaseManager()
        self.proc = DataPreprocessor()
        self.eval = ModelEvaluator(CONFIG["output_dir"])
        self.tracker = MLflowTracker(
            CONFIG["experiment_name"],
            CONFIG["tracking_uri"]
        )
        logger.info("DiabetesPredictionPipeline initialized")


    def run_training(self):

        df = self.db.load_data()

        X_train, X_test, y_train, y_test = self.proc.prepare(df)
        feature_names = self.proc.feature_names
        all_results = {}

        for model_name, model_fn in CONFIG["models"].items():
            model = model_fn()
            logger.info(f"\nTraining: {model_name}")

            # fit
            model.fit(X_train, y_train)

            # Evaluate
            metrics = self.eval.compute_metrics(
                model, X_test, y_test, X_train, y_train
            )
            self.eval.print_report(model, X_test, y_test, model_name)

            # Charts
            cm_path = self.eval.plot_confusion_matrix(
                model, X_test, y_test, model_name
            )
            roc_path = self.eval.plot_roc_curve(
                model, X_test, y_test, model_name
            )
            feat_path = self.eval.plot_feature_importance(
                model, feature_names, model_name
            )
            artifacts = [cm_path, roc_path, feat_path]

            # get model hyperparameters to log
            params = model.get_params()

            # log to MLFlow
            run_id = self.tracker.log_run(
                model_name = model_name,
                model = model,
                params = params,
                metrics = metrics,
                feature_names = feature_names,
                X_train = X_train,
                artifact_paths = artifacts 
            )

            # log to SQL database
            self.db.log_experiment(
                run_id = run_id,
                model_name = model_name,
                accuracy = metrics["accuracy"],
                f1 = metrics["f1"],
                roc_auc = metrics["roc_auc"]
            )

            all_results[model_name] = metrics
        self.proc.save(CONFIG.get("preprocessor_path", "preprocessor.pkl"))

            # promote best model to production
        logger.info(f"\nPromoting best model to the production...")
        self.tracker.promote_best_model()

            # summary
        self._print_summary(all_results)
        return all_results


    def _print_summary(self, results):

        print("\n" + "="*65)
        print("  MODEL COMPARISON SUMMARY")
        print("="*65)
        print(f"  {'Model':<25} {'Accuracy':>9} {'F1':>8} {'ROC-AUC':>9}")
        print(f"  {'-'*25} {'-'*9} {'-'*8} {'-'*9}")
        best_auc = 0
        best_model = ""
        for name, m in results.items():
            marker = " <-- BEST" if m["roc_auc"] == max(
                r["roc_auc"] for r in results.values()
            ) else ""
            print(f"  {name:<25} {m['accuracy']:>9.4f} "
                  f"{m['f1']:>8.4f} {m['roc_auc']:>9.4f}{marker}")
            if m["roc_auc"] > best_auc:
                best_auc = m["roc_auc"]
                best_model = name
        print("="*65)
        print(f"  Best: {best_model} (ROC-AUC = {best_auc:.4f})")
        print(f"  MLflow UI: mlflow ui  -->  http://localhost:5000")
        print("="*65 + "\n")


    def predict(self, patient_data, patient_id, model):

        if not self.proc.is_fitted:
            preprocessor_path = CONFIG.get("preprocessor_path", "preprocessor.pkl")
            self.proc = DataPreprocessor.load(preprocessor_path)
            logger.info("Loaded preprocessor state from disk.")
 

        mapping = {
            "pregnancies": "pregnancies",
            "glucose": "glucose",
            "bloodpressure": "blood_pressure",
            "skinthickness": "skin_thickness",
            "insulin": "insulin",
            "bmi": "bmi",
            "diabetespedigreefunction": "diabetes_pedigree",
            "age": "age"
        }
        
        mapped_data = {}
        for k, v in patient_data.items():
            low_k = k.lower()
            if low_k in mapping:
                mapped_data[mapping[low_k]] = v
            else:
                mapped_data[low_k] = v

        df = pd.DataFrame([mapped_data])
        df = self.proc.clean_data(df)
        df = self.proc.feature_engineering(df)

        for col in self.proc.feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[self.proc.feature_names]

        df_imputed = pd.DataFrame(
            self.proc.imputer.transform(df),
            columns = self.proc.feature_names
        )
        df_scaled = pd.DataFrame(
            self.proc.scaler.transform(df_imputed),
            columns = self.proc.feature_names
        )

        # predict
        risk_score = model.predict_proba(df_scaled)[0][1]
        prediction = int(risk_score >= 0.5)
        risk_label = (
            "HIGH RISK" if risk_score >= 0.7 else
            "MEDIUM RISK" if risk_score >= 0.4 else
            "LOW RISK"
        )

        # save to SQL
        self.db.save_predictions(
            patient_id = patient_id,
            model_name = "Production",
            model_version = "latest",
            risk_score = float(risk_score),
            prediction = prediction
        )

        result = {
            "risk_score": round(float(risk_score), 4),
            "prediction": prediction,
            "risk_label": risk_label,
        }
        logger.info(
            f"Prediction: {risk_label} "
            f"(score={risk_score:.4f})"
        )
        return result
                   

