import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Headless-safe for Docker
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from preprocess import load_data, clean_and_engineer
from evaluate import (
    plot_confusion,
    plot_roc,
    show_classification_report,
    save_model,
)

import os
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)


def build_preprocessor():
    numeric_features = ["HR_ExerciseImpact", "Age", "MaxHR", "Oldpeak", "Cholesterol"]
    categorical_features = ["ChestPainType", "ST_Slope"]
    ordinal_cat = ["Sex", "ExerciseAngina"]

    categorical_transformer = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"),
    )

    numeric_transformer = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
    )

    ordinal_transformer = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OrdinalEncoder(),
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
            ("ord", ordinal_transformer, ordinal_cat),
        ]
    )

    return make_pipeline(preprocessor, KNNImputer(n_neighbors=5))


def plot_learning_curve(model, X, y, label):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=10, scoring="f1", train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label=f"{label} Train")
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label=f"{label} CV")


def train_and_evaluate_model(name, model, param_grid, X_train, X_test, y_train, y_test):
    print(f"\nTraining and tuning: {name}")
    grid = GridSearchCV(model, param_grid, cv=10, scoring="f1", n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    print(f"Best Params: {grid.best_params_}")
    print(f"Best CV F1 Score: {np.round(grid.best_score_ * 100, 2)}")

    # Evaluate on test set
    y_pred = grid.predict(X_test)
    y_score = (
        grid.predict_proba(X_test)[:, 1]
        if hasattr(grid, "predict_proba")
        else grid.decision_function(X_test)
    )

    plot_confusion(y_test, y_pred, title=f"{name} - Confusion Matrix")
    plot_roc(y_test, y_score, title=f"{name} - ROC Curve")
    show_classification_report(y_test, y_pred)

    save_model(grid, f"models/{name.lower()}_model.joblib")
    plot_learning_curve(grid, X_train, y_train, name)

    return grid


def main():
    df = clean_and_engineer(load_data("data/heart.csv"))
    X, y = df.drop(columns="HeartDisease"), df["HeartDisease"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=df["ST_Slope"], random_state=42
    )

    pipe = build_preprocessor()

    plt.figure(figsize=(12, 8))
    for name, clf, grid in [
        ("SGD", SGDClassifier(random_state=42), {
            "sgdclassifier__alpha": [0.0001, 0.001],
            "sgdclassifier__penalty": ["l2", "elasticnet"]
        }),
        ("KNN", KNeighborsClassifier(), {
            "kneighborsclassifier__n_neighbors": [5, 9],
            "kneighborsclassifier__weights": ["uniform", "distance"]
        }),
        ("NB", GaussianNB(), {
            "gaussiannb__var_smoothing": [1e-9, 1e-8]
        }),
        ("SVM", SVC(probability=True, random_state=42), {
            "svc__C": [1, 10],
            "svc__kernel": ["rbf", "linear"]
        }),
        ("Tree", DecisionTreeClassifier(random_state=42), {
            "decisiontreeclassifier__max_depth": [None, 10],
            "decisiontreeclassifier__criterion": ["gini", "entropy"]
        }),
        ("RF", RandomForestClassifier(random_state=42), {
            "randomforestclassifier__n_estimators": [100],
            "randomforestclassifier__max_depth": [None, 20]
        }),
    ]:
        full_pipeline = make_pipeline(pipe, clf)
        train_and_evaluate_model(name, full_pipeline, grid, X_train, X_test, y_train, y_test)

    plt.title("Learning Curves of All Models")
    plt.xlabel("Training Set Size")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("plots/all_learning_curves.png")
    plt.close()


if __name__ == "__main__":
    main()
