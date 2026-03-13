import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

# Load data (assuming train.csv is available)
try:
    df_train = pd.read_csv("train.csv")
except FileNotFoundError:
    print("Error: train.csv not found.")
    exit()


# Preprocessing (same as before)
def preprocess_data(df):
    df.drop(
        columns=["PassengerId", "Ticket", "Cabin", "Name"],
        inplace=True,
        errors="ignore",
    )
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
    return df


df_train = preprocess_data(df_train)

# Features and target
X = df_train.drop("Survived", axis=1)
y = df_train["Survived"]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature Engineering (with variable polynomial degree)
numerical_features = ["Age", "Fare", "SibSp", "Parch"]
categorical_features = ["Pclass", "Sex", "Embarked"]


def create_pipeline(degree):
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        (
                            "poly",
                            PolynomialFeatures(degree=degree, include_bias=False),
                        ),  # Degree is now a parameter
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
        remainder="passthrough",
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", SGDClassifier(random_state=42)),
        ]
    )
    return pipeline


# Evaluation Loop
degrees = range(1, 6)  # Test degrees 1 through 5
train_scores, val_scores = [], []

for degree in degrees:
    print(f"Training model with polynomial degree: {degree}")
    pipeline = create_pipeline(degree)
    pipeline.fit(X_train, y_train)
    train_scores.append(accuracy_score(y_train, pipeline.predict(X_train)))
    val_scores.append(accuracy_score(y_val, pipeline.predict(X_val)))
    print(classification_report(y_val, pipeline.predict(X_val)))


# Plotting
plt.figure(figsize=(10, 6))
plt.plot(degrees, train_scores, marker="o", label="Training Accuracy")
plt.plot(degrees, val_scores, marker="o", label="Validation Accuracy")
plt.xlabel("Polynomial Degree")
plt.ylabel("Accuracy")
plt.title("Impact of Polynomial Degree on Model Accuracy")
plt.xticks(degrees)
plt.legend()
plt.grid(True)
plt.show()

print("Training and Validation Scores:")
for degree, train_score, val_score in zip(degrees, train_scores, val_scores):
    print(
        f"Degree: {degree}, Training Accuracy: {train_score:.4f}, Validation Accuracy: {val_score:.4f}"
    )
