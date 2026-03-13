import gradio as gr
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    hinge_loss,
    log_loss,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_data():
    """Load MNIST dataset and return flattened images with labels"""
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X, y = mnist["data"], mnist["target"].astype(np.uint8)
    return X, y


def create_pipeline():
    """Create sklearn pipeline with preprocessing and classifier"""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", SGDClassifier(random_state=42)),
        ]
    )


def train_and_evaluate(X_train, y_train, X_val, y_val):
    """Train model with hyperparameter tuning and return best model"""
    pipeline = create_pipeline()

    param_grid = {
        "classifier__loss": ["hinge", "log_loss"],
        "classifier__alpha": [1e-4, 1e-3],
        "classifier__penalty": ["l2"],
        "classifier__max_iter": [150, 250, 350, 450, 550],
        "classifier__tol": [1e-3],
        "classifier__early_stopping": [True],
    }

    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring="accuracy", verbose=1)
    grid.fit(X_train, y_train)

    print("Best hyperparameters:", grid.best_params_)
    return grid.best_estimator_


def plot_learning_curve(model, X_train, y_train, X_val, y_val):
    """Generate learning curve plot"""
    train_sizes = np.linspace(0.1, 1.0, 8)
    train_losses, val_losses = [], []

    final_loss = model.get_params()["classifier__loss"]

    for frac in train_sizes:
        size = int(frac * len(X_train))
        X_sub, y_sub = X_train[:size], y_train[:size]

        model_clone = clone(model)
        model_clone.fit(X_sub, y_sub)

        if final_loss == "log_loss":
            train_loss = log_loss(y_sub, model_clone.predict_proba(X_sub))
            val_loss = log_loss(y_val, model_clone.predict_proba(X_val))
            loss_name = "Log Loss"
        else:  # hinge loss
            train_loss = hinge_loss(y_sub, model_clone.decision_function(X_sub))
            val_loss = hinge_loss(y_val, model_clone.decision_function(X_val))
            loss_name = "Hinge Loss"

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes * len(X_train), train_losses, "o-", label=f"Train {loss_name}")
    plt.plot(
        train_sizes * len(X_train), val_losses, "s-", label=f"Validation {loss_name}"
    )
    plt.title(f"Learning Curve (SGDClassifier - {loss_name})")
    plt.xlabel("Training Set Size")
    plt.ylabel(loss_name)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"learning_curve_{final_loss}.png")
    plt.show()


def show_confusion_matrix(y_true, y_pred, title):
    """Display confusion matrix with given title"""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title(title)
    plt.show()


def main():
    """Main training pipeline"""
    X, y = load_data()

    # Split into train/val/test (60%/20%/20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, stratify=y_train, random_state=42
    )

    # Train and evaluate model
    model = train_and_evaluate(X_train, y_train, X_val, y_val)
    joblib.dump(model, "best_pipeline.joblib")

    # Generate learning curve
    plot_learning_curve(model, X_train, y_train, X_val, y_val)

    # Evaluate on all sets
    for x, y_true, name in [
        (X_val, y_val, "Validation"),
        (X_test, y_test, "Test"),
    ]:
        y_pred = model.predict(x)
        show_confusion_matrix(y_true, y_pred, f"{name} Confusion Matrix")
        print(f"{name} Report:\n{classification_report(y_true, y_pred)}")


if __name__ == "__main__":
    main()
