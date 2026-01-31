from sklearn.ensemble import RandomForestClassifier
from preprocessing import preprocess_data
import matplotlib.pyplot as plt
import seaborn as sns
from xgb import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train_model():
    X_train_scaled, X_test_scaled, y_train_resampled, y_test = preprocess_data()

    xgb_model = XGBClassifier(n_estimators=100, random_state=42)
    xgb_model.fit(X_train_scaled, y_train_resampled)
    y_pred = xgb_model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return xgb_model, accuracy, cm, report

    # Heatmap
def plot_confusion_matrix(cm):    
    plt.figure()
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    xgb_model, accuracy, cm, report = train_model()
    plot_confusion_matrix(cm)

    print("Model Accuracy :", accuracy)
    print("\nConfusion matrix :\n", cm)
    print("\nClassification Report :\n", report)