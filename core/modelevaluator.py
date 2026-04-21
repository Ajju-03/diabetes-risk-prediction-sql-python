from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
import seaborn as sns
import os
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from config import CONFIG

class ModelEvaluator:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def compute_metrics(self, model, X_test, y_test, X_train, y_train):
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        cv = StratifiedKFold(n_splits=CONFIG["cv_folds"], 
                            shuffle=True, 
                            random_state=CONFIG["random_state"])
        cv_scores = cross_val_score(model, X_train, y_train,
                                cv=cv, scoring="roc_auc" )
        return {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall": round(recall_score(y_test, y_pred), 4),
            "f1": round(f1_score(y_test, y_pred), 4),
            "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
            "cv_roc_auc_mean": round(cv_scores.mean(), 4),
            "cv_roc_auc_std": round(cv_scores.std(), 4),
        }


    def plot_confusion_matrix(self, model, X_test, y_test, model_name):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("Predicted labels")
        ax.set_ylabel("True labels")
        ax.set_title(f"Confusion Matrix for {model_name}", fontweight='bold')
        plt.tight_layout()
        path = os.path.join(self.output_dir, f"cm_{model_name}.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        return path
    
    def plot_roc_curve(self, model, X_test, y_test, model_name):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig, ax= plt.subplots(figsize=(5,4))
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, color="#2563EB", lw=2,
                label=f"ROC Curve (AUC = {auc:.3f})")
        ax.plot([0,1], [0,1], "k--", lw=1, alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve — {model_name}", fontweight="bold")
        ax.legend(loc="lower right")
        plt.tight_layout()
        path = os.path.join(self.output_dir, f"roc_{model_name}.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        return path
    

    def plot_feature_importance(self, model, feature_names, model_name):
        if not hasattr(model, "feature_importances_"):
            return None
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        fig, ax = plt.subplots(figsize=(8,5))
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(feature_names)))
        ax.bar(range(len(feature_names)),
               importances[indices], color=colors)
        ax.set_xticks(range(len(feature_names)))
        ax.set_xticklabels(
            [feature_names[i] for i in indices],
            rotation=45, ha="right", fontsize=9
        )
        ax.set_title(f"Feature Importance — {model_name}", fontweight="bold")
        ax.set_ylabel("Importance Score")
        plt.tight_layout()
        path = os.path.join(self.output_dir, f"feat_{model_name}.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        return path

    
    def print_report(self, model, X_test, y_test, model_name):
        y_pred = model.predict(X_test)
        print(f"Classification Report for {model_name}")
        print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))