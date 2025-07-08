# src/preprocess.py

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import KNNImputer

ORDINAL_COLS = ['Sex', 'ExerciseAngina', 'ChestPainType', 'RestingECG', 'ST_Slope']
CONTINUOUS = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
DISCRETE = ['FastingBS']
TARGET = 'HeartDisease'


def load_data(rel_path: str) -> pd.DataFrame:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    full_path = os.path.join(base_dir, rel_path)
    return pd.read_csv(full_path)


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Clean invalid zeroes
    df["RestingBP"].replace(0, np.nan, inplace=True)
    df.dropna(subset=["RestingBP"], inplace=True)
    df["Cholesterol"].replace(0, np.nan, inplace=True)

    # Encode categories
    encoder = OrdinalEncoder()
    encoded = encoder.fit_transform(df[ORDINAL_COLS])
    cat_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(), index=df.index)

    # Join back
    df_encoded = pd.concat([df.drop(columns=ORDINAL_COLS), cat_df], axis=1)

    # Impute
    knn = KNNImputer(n_neighbors=5)
    df_encoded = pd.DataFrame(knn.fit_transform(df_encoded), columns=df_encoded.columns, index=df_encoded.index)

    # Feature engineering
    df_encoded["HR_ExerciseImpact"] = df_encoded["MaxHR"] / (df_encoded["Oldpeak"] + 5)

    # Drop weak features
    df_encoded.drop(columns=["RestingECG", "FastingBS"], inplace=True)

    return df_encoded
