# Step 1: Setup and imports
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def full_pipeline(train_data, test_data):

    # Explicitly set the PassengerId column as the index column
    train_data = train_data.set_index("PassengerId")
    test_data = test_data.set_index("PassengerId")

    # Step 4: Separate labels from features
    X_train = train_data.drop(["Survived"], axis=1)
    y_train = train_data["Survived"]
    X_test = test_data.copy()

    # Step 6: Define numerical and categorical columns
    num_features = ["Age", "SibSp", "Parch", "Fare"]
    cat_features = ["Pclass", "Sex", "Embarked", "Title"]

    # Step 7: Build pipelines
    num_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder()),
        ]
    )

    full_pipeline = ColumnTransformer(
        [("num", num_pipeline, num_features), ("cat", cat_pipeline, cat_features)]
    )

    # Step 8: Apply transformations
    X_train_prepared = full_pipeline.fit_transform(X_train)
    X_test_prepared = full_pipeline.transform(X_test)

    # Step 9: Extract final feature names
    cat_encoder = full_pipeline.named_transformers_["cat"]["encoder"]
    cat_feature_names = cat_encoder.get_feature_names_out(cat_features)
    all_feature_names = num_features + list(cat_feature_names)

    return X_train_prepared, X_test_prepared, y_train, all_feature_names
