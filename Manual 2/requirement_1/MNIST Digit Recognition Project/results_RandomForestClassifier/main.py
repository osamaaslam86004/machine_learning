import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import shift
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, learning_curve, train_test_split

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

# Stratified split: train 60%, val 20%, test 20%
X_temp, X_test, y_temp, y_test = train_test_split(
    X_aug_flat, y_aug, test_size=0.2, stratify=y_aug, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
)

# GridSearchCV for Random Forest
param_grid = {"n_estimators": [30, 40, 50], "max_depth": [100, 150, 200]}
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
grid = GridSearchCV(rf, param_grid, cv=3, scoring="accuracy", verbose=1)
grid = grid.fit(X_train, y_train)

# Print best hyperparameters
print("Best hyperparameters found by GridSearchCV:")
print(grid.best_params_)

# Get the best estimator from GridSearchCV
grid = grid.best_estimator_  # best model

# Save the best model to a file
joblib.dump(grid, f"best_fine_tune_random_forest.joblib")
print(f"Best model saved to 'best_fine_tune_random_forest.joblib'")
# Learning Curve
train_sizes, train_scores, val_scores = learning_curve(
    grid, X_train, y_train, cv=3, scoring="accuracy"
)
plt.plot(train_sizes, train_scores.mean(axis=1), label="Train")
plt.plot(train_sizes, val_scores.mean(axis=1), label="Validation")
plt.title("Learning Curve")
plt.xlabel("Training Size")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig(f"random_forest_learning_curve.png")


# Confusion Matrices
def show_confusion(title, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title(title)
    plt.show()
    return cm


y_val_pred = grid.predict(X_val)
y_test_pred = grid.predict(X_test)


val_cm = show_confusion("Validation Confusion Matrix", y_val, y_val_pred)
test_cm = show_confusion("Test Confusion Matrix", y_test, y_test_pred)

cr = classification_report(y_val, y_val_pred)
print("Validation report:\n", cr)

cr = classification_report(y_test, y_test_pred)
print("Test report:\n", cr)


# Print Matrices
print("Validation CM:\n", val_cm)
print("Test CM:\n", test_cm)

import os
import zipfile

from google.colab import files

# Define the zip filename
zip_filename = "images_and_models.zip"

# Create a zip file
with zipfile.ZipFile(zip_filename, "w") as zipf:
    for root, _, files_list in os.walk("/content/"):
        for file in files_list:
            if file.endswith(".png") or file.endswith(".joblib"):
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(
                    file_path, "/content/"
                )  # relative path inside zip
                zipf.write(file_path, arcname)

# Trigger download
files.download(zip_filename)
