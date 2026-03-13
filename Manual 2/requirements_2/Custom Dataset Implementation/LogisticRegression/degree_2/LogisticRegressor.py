# The aim for Titanic dataset classification is typically to predict whether a passenger survived or not. We want a model that is both accurate and reliable in its predictions. Therefore, we generally aim for high values across the board: precision, recall, and F1-score.

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

# 1. Data Loading and Exploration
# Load the datasets
try:
    df_train = pd.read_csv("train.csv")
    df_test = pd.read_csv("test.csv")
    print("Datasets loaded successfully.")
except FileNotFoundError:
    print(
        "Error: train.csv or test.csv not found. Please ensure the files are in the correct directory."
    )
    exit()  # Exit if files are not found


# 2. Data Preprocessing
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


# 3. Feature Engineering and Selection
# Define features and target
X = df_train.drop("Survived", axis=1)
y = df_train["Survived"]
X_test = df_test.copy()


# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Define numerical and categorical features
numerical_features = ["Age", "Fare", "SibSp", "Parch"]
categorical_features = ["Pclass", "Sex", "Embarked"]  # Pclass is treated as categorical

# Create a preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                    ("scaler", StandardScaler()),
                ]
            ),
            numerical_features,
        ),
        (
            "cat",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            categorical_features,
        ),
    ],
    remainder="passthrough",  # Pass through any other columns not explicitly transformed
)


# 4. Model Training (Logistic Regression with Regularization)
pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(solver="liblinear", random_state=42),
        ),  # Use liblinear for better performance
    ]
)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    "classifier__C": [0.01, 0.1, 1, 10, 100],  # Inverse of regularization strength
    "classifier__penalty": ["l1", "l2"],
    "classifier__max_iter": [100, 150, 200, 250, 300, 350, 400, 450, 500],
    "classifier__tol": [1e-3, 1e-4, 1e-5],
    "classifier__class_weight": ["balanced", None],
}


grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy", verbose=1)

start_time = time.time()
grid_search.fit(X_train, y_train)
end_time = time.time()

training_time = end_time - start_time
print(f"\nTraining time: {training_time:.2f} seconds")

print("Best parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_

# 5. Model Evaluation
y_pred = best_model.predict(X_val)
y_pred_proba = best_model.predict_proba(X_val)[:, 1]  # Probabilities for ROC AUC

print("\nClassification Report:")
print(classification_report(y_val, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_pred))

accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
roc_auc = roc_auc_score(y_val, y_pred_proba)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")


# ROC Curve
fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--")  # Random classifier line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()


# 6. Learning Curve
def plot_learning_curve(estimator, X, y):
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_scores, val_scores = [], []
    for train_size in train_sizes:
        size = int(len(X) * train_size)
        X_subset, y_subset = X[:size], y[:size]
        estimator.fit(X_subset, y_subset)
        train_scores.append(estimator.score(X_subset, y_subset))
        val_scores.append(
            estimator.score(X_val, y_val)
        )  # Re-using X_val, y_val for simplicity

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes * len(X), train_scores, "o-", label="Training Score")
    plt.plot(train_sizes * len(X), val_scores, "o-", label="Validation Score")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid()
    plt.show()


plot_learning_curve(best_model, X_train, y_train)


# 7. Feature Importance
# Get coefficients from the logistic regression model
try:
    coefficients = best_model.named_steps["classifier"].coef_[0]
except AttributeError:
    print("Error: Could not extract coefficients from the trained model.")
    exit()


# Get feature names after preprocessing
num_features = None
if "poly" in best_model.named_steps["preprocessor"].transformers_[0][1]:
    num_features = (
        best_model.named_steps["preprocessor"]
        .transformers_[0][1]["poly"]
        .get_feature_names_out(numerical_features)
    )
else:
    # Get feature names after preprocessing
    # Since PolynomialFeatures is removed, use the original numerical features
    num_features = best_model.named_steps["preprocessor"].transformers_[0][
        2
    ]  # Get numerical feature names from the pipeline

cat_features = (
    best_model.named_steps["preprocessor"]
    .transformers_[1][1]
    .get_feature_names_out(categorical_features)
)


# Combine feature names
all_features = np.concatenate([num_features, cat_features])

# Match coefficients with feature names
feature_importance = pd.DataFrame(
    {"Feature": all_features, "Coefficient": coefficients}
)
feature_importance["Abs_Coefficient"] = abs(feature_importance["Coefficient"])
feature_importance = feature_importance.sort_values("Abs_Coefficient", ascending=False)

# Plot feature importance
plt.figure(figsize=(12, 6))
sns.barplot(
    x="Coefficient", y="Feature", data=feature_importance.head(20), orient="h"
)  # Showing top 20
plt.title("Top 20 Feature Importances")
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.show()

print("\nTop 20 Feature Importances:")
print(
    feature_importance[["Feature", "Coefficient"]].head(20)
)  # Displaying top 20 features


# 8. Prediction and Submission
y_test_pred = best_model.predict(X_test)
submission = pd.DataFrame(
    {"PassengerId": df_test.index + 892, "Survived": y_test_pred}
)  # Corrected PassengerId
submission.to_csv("submission.csv", index=False)
print("\nPredictions saved to submission.csv")
