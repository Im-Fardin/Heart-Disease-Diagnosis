import os
import matplotlib
matplotlib.use("Agg")  # Use headless backend for safe Docker execution
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
import joblib


def plot_confusion(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)

    os.makedirs("plots", exist_ok=True)
    path = f"plots/confusion_{title.replace(' ', '_')}.png"
    plt.savefig(path)
    plt.close()


def plot_roc(y_true, y_score, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")

    os.makedirs("plots", exist_ok=True)
    path = f"plots/roc_{title.replace(' ', '_')}.png"
    plt.savefig(path)
    plt.close()


def show_classification_report(y_true, y_pred):
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred))


def save_model(model, path):
    joblib.dump(model, path)
    print(f"Model saved to: {path}")
