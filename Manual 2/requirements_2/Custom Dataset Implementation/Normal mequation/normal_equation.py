import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
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
    )
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
    return df


# Preprocess both train and test data
df_train = preprocess_data(df_train)
df_test = preprocess_data(df_test)

# Define features and target
X = df_train.drop("Survived", axis=1)
y = df_train["Survived"]
X_test_unprocessed = df_test.copy()  # Store original test data for later

# Split into train/val (75%/25%)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

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

# Preprocess training and validation data
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)


# --- Linear Regression with Normal Equation ---
class RidgeRegressionNormalEquation:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.theta = None

    def fit(self, X, y):
        X_b = np.c_[np.ones((len(X), 1)), X]
        n_features = X_b.shape[1]

        L2_penalty = self.alpha * np.eye(n_features)
        L2_penalty[0, 0] = 0  # Don’t regularize bias

        self.theta = np.linalg.inv(X_b.T @ X_b + L2_penalty) @ X_b.T @ y

    def predict(self, X):
        X_b = np.c_[np.ones((len(X), 1)), X]
        return X_b @ self.theta


start_time = time.time()

alphas = [0.001, 0.01, 0.1, 0.5, 1.0, 10.0, 100.0]
results = []

for alpha in alphas:
    model = RidgeRegressionNormalEquation(alpha=alpha)
    model.fit(X_train_processed, y_train)

    y_val_pred = np.round(model.predict(X_val_processed))
    f1 = f1_score(y_val, y_val_pred)
    acc = accuracy_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred)
    recall = recall_score(y_val, y_val_pred)

    results.append((alpha, acc, f1, precision, recall))

end_time = time.time()
print(f"Training time (Normal Equation): {end_time - start_time} seconds")

# Print the results as a table
print("Alpha | Accuracy | F1 Score | Precision | Recall")
print("-----------------------------------------")
for alpha, acc, f1, precision, recall in results:
    print(f"{alpha:5} | {acc:.4f}   | {f1:.4f}   | {precision:.4f}   | {recall:.4f}")


alphas_list, acc_list, f1_list, precision_list, recall_list = zip(*results)

plt.figure(figsize=(8, 5))
plt.plot(alphas_list, f1_list, marker="o", label="F1 Score")
plt.plot(alphas_list, acc_list, marker="s", label="Accuracy")
plt.plot(alphas_list, precision_list, marker="^", label="Precision")
plt.plot(alphas_list, recall_list, marker="D", label="Recall")
plt.xscale("log")
plt.xlabel("Alpha (log scale)")
plt.ylabel("Score")
plt.title("Grid Search over Alpha for Ridge Regression")
plt.grid(True)
plt.legend()
plt.show()


# --- Feature Importance ---

# Find the best alpha based on the highest F1 score
best_alpha = max(results, key=lambda item: item[2])[0]  # item[2] is the F1 score

print(f"Best alpha: {best_alpha}")

# Fit a new model with the best alpha
best_model = RidgeRegressionNormalEquation(alpha=best_alpha)
best_model.fit(X_train_processed, y_train)

# Access the theta values of the best model
best_theta = best_model.theta

print("\nTheta values for the best model:")
print(best_theta)


# Get feature names after preprocessing
num_feature_names = preprocessor.named_transformers_["num"][
    "poly"
].get_feature_names_out(num_features)
# Add bias + numeric + categorical features
all_feature_names = ["bias"] + list(num_feature_names) + cat_features

# Sanity check to avoid mismatch
assert len(all_feature_names) == len(
    best_theta
), "Mismatch between features and coefficients!"

# Print as a sorted table by absolute importance
print("\nFeature Importance (from Ridge Regression Normal Equation):")
feature_importances = sorted(
    zip(all_feature_names, best_theta), key=lambda x: abs(x[1]), reverse=True
)

for feature, coef in feature_importances:
    print(f"{feature:30}: {coef:.5f}")


# --- Learning Curve ---
def plot_learning_curve(model, X, y, train_sizes=np.linspace(0.1, 1.0, 10)):
    train_errors, val_errors = [], []
    for train_size in train_sizes:
        size = int(len(X) * train_size)
        X_subset, y_subset = X[:size], y[:size]
        model.fit(X_subset, y_subset)
        y_train_pred = np.round(model.predict(X_subset))  # Round for classification
        y_val_pred = np.round(
            model.predict(X_val_processed)
        )  # Use consistent validation set

        train_error = mean_squared_error(y_subset, y_train_pred)
        val_error = mean_squared_error(y_val, y_val_pred)
        train_errors.append(train_error)
        val_errors.append(val_error)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes * len(X), np.sqrt(train_errors), label="Training Error")
    plt.plot(train_sizes * len(X), np.sqrt(val_errors), label="Validation Error")
    plt.xlabel("Training Set Size")
    plt.ylabel("RMSE (Root Mean Squared Error)")  # More appropriate for regression
    plt.title("Learning Curve for Normal Equation (Classification)")
    plt.legend()
    plt.grid(True)
    plt.show()


# Plot the learning curve for the best model
plot_learning_curve(best_model, X_train_processed, y_train)


# --- Prediction on Test Data --- (unmodified)
X_test_processed = preprocessor.transform(X_test_unprocessed)
y_test_pred = best_model.predict(X_test_processed)

y_pred_binary = np.round(y_test_pred).astype(int)

# --- Outputting Predictions --- (unmodified)
output = pd.DataFrame({"PassengerId": df_test.index + 1, "Survived": y_pred_binary})
output.to_csv("submission_normal_equation.csv", index=False)
print("\nPredictions saved to submission_normal_equation.csv")
