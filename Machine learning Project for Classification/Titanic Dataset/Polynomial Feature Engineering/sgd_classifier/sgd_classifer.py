# The aim for Titanic dataset classification is typically to predict whether a passenger survived or not. We want a model that is both accurate and reliable in its predictions. Therefore, we generally aim for high values across the board: precision, recall, and F1-score.


import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Load the datasets
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")


# Preprocessing function
def preprocess_data(df):
    df.drop(
        columns=["PassengerId", "Ticket", "Cabin", "Name"],
        inplace=True,
        errors="ignore",
    )  # errors='ignore' handles missing columns in test.csv
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
    return df


# Preprocess both train and test data
df_train = preprocess_data(df_train)
df_test = preprocess_data(df_test)


# Define features and target
X = df_train.drop("Survived", axis=1)
y = df_train["Survived"]
X_test = df_test.copy()  # Use the entire test.csv for testing (predictions)


# Split into train/val (75%/25%)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)


# Define numerical and categorical features
num_features = ["Age", "Fare", "Pclass", "SibSp", "Parch"]
cat_features = ["Sex", "Embarked"]

# Create preprocessing pipeline
num_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="mean")),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
    ]
)

preprocessor = ColumnTransformer(
    [
        ("num", num_pipeline, num_features),
        ("cat", SimpleImputer(strategy="most_frequent"), cat_features),
    ]
)


# Grid Search (optional, can be time-consuming)
pipe = Pipeline(
    [("preprocessor", preprocessor), ("clf", SGDClassifier(random_state=42))]
)

param_grid = {
    "clf__alpha": [0.0001, 0.001, 0.01],
    "clf__penalty": ["l2", "l1", "elasticnet"],
    "clf__max_iter": [100, 150, 200],
    "clf__loss": ["hinge", "log_loss", "modified_huber"],
    "clf__learning_rate": ["constant", "optimal", "invscaling"],
    "clf__eta0": [0.1, 0.01, 0.001],
    "clf__power_t": [0.5, 0.1, 0.01],
    "clf__tol": [1e-3],
    "clf__early_stopping": [True],
}


start = time.time()

grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy", verbose=True)
grid_search = grid_search.fit(X_train, y_train)

end = time.time()
print(f"Training time: {round(end - start, 4)} seconds")

best_model = grid_search.best_estimator_

print("\nGrid Search Results:")
print("Best params:", grid_search.best_params_)
# Calculate RMSE on the training data (as a proxy since we don't have labels for the test data)
y_val_pred = best_model.predict(X_val)

# Calculate Accuracy on Training data
accuracy = accuracy_score(y_val, y_val_pred)
precision = precision_score(y_val, y_val_pred)
recall = recall_score(y_val, y_val_pred)
f1 = 2 * (precision * recall) / (precision + recall)

print(f"Precision on Validation Data: {precision}")
print(f"Recall on Validation Data: {recall}")
print(f"Accuracy on Validation Data: {accuracy}")
print("Best F1 score:", grid_search.best_score_)


# Learning curve plotting function
def plot_learning_curve(estimator, X_train, y_train, X_val, y_val):
    """
    Generate a learning curve plot showing training and validation F1 Score.

    Args:
        estimator: The machine learning model (pipeline) to evaluate.
        title: Title of the learning curve plot.
        X_train: Training data features.
        y_train: Training data target.
        X_val: Validation data features.
        y_val: Validation data target.
    """

    plt.figure()
    plt.title("plot_learning_curve")
    plt.xlabel("Training Set Size")
    plt.ylabel("F1 Score")
    plt.grid()

    train_sizes = np.linspace(0.1, 1.0, 5)  # 5 training set sizes
    train_f1, val_f1 = [], []

    for train_size in train_sizes:
        size = int(train_size * len(X_train))
        X_subset, y_subset = X_train[:size], y_train[:size]

        estimator.fit(X_subset, y_subset)
        y_train_pred = estimator.predict(X_subset)
        y_val_pred = estimator.predict(X_val)

        train_f1.append(f1_score(y_subset, y_train_pred))
        val_f1.append(f1_score(y_val, y_val_pred))

    plt.plot(train_sizes * len(X_train), train_f1, "o-", color="r", label="Training F1")
    plt.plot(train_sizes * len(X_train), val_f1, "o-", color="g", label="Validation F1")

    plt.legend(loc="best")
    plt.show()


# We can no longer print a classification report since we don't have true labels for the test data.
# print(classification_report(y_test, y_pred))
plot_learning_curve(best_model, X_train, y_train, X_val, y_val)


# --- Feature Importance (SGDClassifier --- Co-efficients) ---
if (
    best_model.named_steps["clf"].penalty == "l1"
    or best_model.named_steps["clf"].penalty == "elasticnet"
):
    feature_names = (
        best_model.named_steps["preprocessor"]
        .transformers_[0][1]["poly"]
        .get_feature_names_out(num_features)
    )
    feature_names = np.append(feature_names, cat_features)

    importances = best_model.named_steps["clf"].coef_[
        0
    ]  # Access coefficients correctly
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    importance_df = importance_df.sort_values(by="Importance", key=abs, ascending=False)
    print("\nFeature Importances (SGDClassifier):")
    print(importance_df)
else:
    # Feature Importance for SGDClassifier with L2 penalty using Permutation Importance
    print("\nCalculating feature importances using permutation importance...")

    # Extract the preprocessor
    preprocessor = best_model.named_steps["preprocessor"]

    # Get feature names after preprocessing
    numerical_features = ["Age", "Fare", "Pclass", "SibSp", "Parch"]  # Adjust if needed
    categorical_features = ["Sex", "Embarked"]  # Adjust if needed

    numerical_features_transformed = preprocessor.transformers_[0][1][
        "poly"
    ].get_feature_names_out(numerical_features)
    all_feature_names = np.append(numerical_features_transformed, categorical_features)

    # Perform permutation importance
    r_multi = permutation_importance(
        best_model, X_val, y_val, n_repeats=30, random_state=42, n_jobs=-1
    )  # Adjust n_repeats for more stable results

    # Store results
    importances = r_multi.importances_mean
    std = r_multi.importances_std

    importance_df = pd.DataFrame(
        {"Feature": all_feature_names, "Importance": importances, "std": std}
    )
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    print("\nFeature Importances (Permutation Importance):")
    print(importance_df)


# Since we don't have y_test (actual survival labels for test.csv), we make predictions and can't evaluate accuracy/f1
y_pred = best_model.predict(X_test)
# --- Outputting predictions to a CSV (for submission or further analysis) ---
output = pd.DataFrame(
    {"PassengerId": df_test.index + 1, "Survived": y_pred}
)  # Assumes PassengerId is just an index
output.to_csv("submission.csv", index=False)
print("\nPredictions saved to submission.csv")
