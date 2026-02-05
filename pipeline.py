import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def convert_types(df, numeric_cols=None, categorical_cols=None):
    df = df.copy()
    if numeric_cols:
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # avoid NaNs
    if categorical_cols:
        for col in categorical_cols:
            df[col] = df[col].astype('category')
    return df


def collapse_categories(df, collapse_map=None):
    df = df.copy()
    if collapse_map:  # dict of col names and collapse functions
        for col, func in collapse_map.items():
            df[col] = df[col].apply(func)
    return df


def encode_categoricals(df, categorical_cols=None, drop_first=True):
    df = df.copy()
    if categorical_cols:
        df = pd.get_dummies(
            df, columns=categorical_cols, drop_first=drop_first
        )
    return df


def normalize_numeric(df, numeric_cols=None):
    df = df.copy()
    if numeric_cols:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df


def drop_columns(df, drop_cols=None):
    df = df.copy()
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
    return df


def create_target(df, target_col):
    y = df[target_col]
    X = df.drop(columns=[target_col])
    target_prevalence = y.mean()  # need target to be binary
    return X, y, target_prevalence

def split_train_tune_test(X, y, test_size=0.2, tune_size=0.25, random_state=42):
    X_tune_test, X_test, y_tune_test, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train, X_tune, y_train, y_tune = train_test_split(
        X_tune_test, y_tune_test, test_size=tune_size,
        random_state=random_state
    )

    return X_train, X_tune, X_test, y_train, y_tune, y_test


