import os
import zipfile

import joblib
import matplotlib.pyplot as plt
import numpy as np
from google.colab import files
from scipy.ndimage import shift
from sklearn.base import clone
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
from sklearn.preprocessing import StandardScaler

# Load MNIST
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X, y = mnist["data"], mnist["target"].astype(np.uint8)
X = X.values if hasattr(X, "values") else X  # For some pandas versions

# Reshape to 28x28 for shifting
X_images = X.reshape(-1, 28, 28)

# Augment by shifting left and right
X_left = np.array([shift(img, shift=[0, -5], cval=0) for img in X_images])
X_right = np.array([shift(img, shift=[0, 5], cval=0) for img in X_images])

# Stack and flatten
X_aug = np.concatenate([X_images, X_left, X_right])
y_aug = np.concatenate([y, y, y])
X_aug_flat = X_aug.reshape(X_aug.shape[0], -1)

# Scale features
scaler = StandardScaler()
X_aug_flat = scaler.fit_transform(X_aug_flat)

# Stratified split: train 60%, val 20%, test 20%
X_temp, X_test, y_temp, y_test = train_test_split(
    X_aug_flat, y_aug, test_size=0.2, stratify=y_aug, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
)

# GridSearchCV for SGDClassifier
param_grid = {
    "loss": ["hinge", "log_loss"],
    "alpha": [1e-4, 1e-3],
    "penalty": ["l2"],
    "max_iter": [150, 250, 350, 450, 550],
    "tol": [1e-3],
    "early_stopping": [True],
}

sgd = SGDClassifier(random_state=42)
grid = GridSearchCV(sgd, param_grid, cv=3, scoring="accuracy", verbose=1)
grid = grid.fit(X_train, y_train)

# Print best hyperparameters
print("Best hyperparameters found by GridSearchCV:")
print(grid.best_params_)

# Get the best estimator
grid = grid.best_estimator_

# Save the best model to a file
joblib.dump(grid, f"best_fine_tune_sgd.joblib")
print(f"Best model saved to 'best_fine_tune_sgd.joblib'")

# **Save the scaler**
joblib.dump(scaler, "scaler.joblib")
print("Scaler saved to 'scaler.joblib'")


# Training sizes: 10% to 100%
train_sizes = np.linspace(0.1, 1.0, 8)
train_losses = []
val_losses = []

final_loss = grid.get_params().get("loss")

for frac in train_sizes:
    size = int(frac * len(X_train))
    X_sub = X_train[:size]
    y_sub = y_train[:size]

    model = clone(grid)
    model.fit(X_sub, y_sub)

    if final_loss == "log_loss":
        # Predict probabilities
        y_sub_proba = model.predict_proba(X_sub)
        y_val_proba = model.predict_proba(X_val)

        # Compute log loss
        train_loss = log_loss(y_sub, y_sub_proba)
        val_loss = log_loss(y_val, y_val_proba)
        loss_name = "Log Loss"

    elif final_loss == "hinge":
        # Predict margin scores
        y_sub_score = model.decision_function(X_sub)
        y_val_score = model.decision_function(X_val)

        # Compute hinge loss
        train_loss = hinge_loss(y_sub, y_sub_score)
        val_loss = hinge_loss(y_val, y_val_score)
        loss_name = "Hinge Loss"

    else:
        raise ValueError(f"Loss '{final_loss}' is not supported in this dynamic plot.")

    train_losses.append(train_loss)
    val_losses.append(val_loss)

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(
    train_sizes * len(X_train), train_losses, label=f"Train {loss_name}", marker="o"
)
plt.plot(
    train_sizes * len(X_train), val_losses, label=f"Validation {loss_name}", marker="s"
)
plt.title(f"Learning Curve (SGDClassifier - {loss_name})")
plt.xlabel("Training Set Size")
plt.ylabel(loss_name)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"sgd_final_model_learning_curve_{final_loss}.png")
plt.show()


# Confusion Matrices
def show_confusion(title, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title(title)
    plt.show()
    return cm


y_train_pred = grid.predict(X_train)
y_val_pred = grid.predict(X_val)
y_test_pred = grid.predict(X_test)

train_cm = show_confusion("Train Confusion Matrix", y_train, y_train_pred)
val_cm = show_confusion("Validation Confusion Matrix", y_val, y_val_pred)
test_cm = show_confusion("Test Confusion Matrix", y_test, y_test_pred)

cr = classification_report(y_train, y_train_pred)
print("Training report:\n", cr)

cr = classification_report(y_val, y_val_pred)
print("Validation report:\n", cr)

cr = classification_report(y_test, y_test_pred)
print("Test report:\n", cr)

# Print Matrices
print("Train CM:\n", train_cm)
print("Validation CM:\n", val_cm)
print("Test CM:\n", test_cm)


# Define the zip filename
zip_filename = "images_and_models.zip"

# Create a zip file
with zipfile.ZipFile(zip_filename, "w") as zipf:
    for root, _, files_list in os.walk("/content/"):
        for file in files_list:
            if file.endswith(".png") or file.endswith(".joblib"):
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, "/content/")
                zipf.write(file_path, arcname)

# Trigger download
files.download(zip_filename)
