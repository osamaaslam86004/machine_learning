import os
import zipfile

import joblib
import matplotlib.pyplot as plt
import numpy as np
from google.colab import files
from skimage.feature import hog
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


# ===============================
# 1. Custom Transformer
# ===============================
class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, zones=(4, 4), pixels_per_cell=(7, 7), cells_per_block=(2, 2)):
        self.zones = zones
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_images = X.reshape(-1, 28, 28)
        hog_features = []
        zoning_features = []

        for img in X_images:
            hog_feat = hog(
                img,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                feature_vector=True,
            )
            zone_feat = self._extract_zoning_features(img)
            hog_features.append(hog_feat)
            zoning_features.append(zone_feat)

        return np.hstack([hog_features, zoning_features])

    def _extract_zoning_features(self, img):
        h, w = img.shape
        zh, zw = self.zones
        zone_features = []
        for i in range(zh):
            for j in range(zw):
                block = img[
                    i * h // zh : (i + 1) * h // zh, j * w // zw : (j + 1) * w // zw
                ]
                zone_features.append(block.mean())
        return zone_features


# 3. Load MNIST data
mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X, y = mnist["data"], mnist["target"].astype(np.uint8)
X = X.values if hasattr(X, "values") else X
X_images = X.reshape(-1, 28, 28)


# 4. Split the data into training, validation, and test sets
X_temp, X_test, y_temp, y_test = train_test_split(
    X_images, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
)

# 5. Create the pipeline
pipeline = Pipeline(
    [
        ("features", FeatureExtractor()),  # Use the custom transformer
        ("scaler", StandardScaler()),  # Feature scaling
        ("sgd", SGDClassifier(random_state=42)),
    ]
)


# 6. Set up GridSearchCV
param_grid = {
    "sgd__loss": ["hinge", "log_loss"],
    "sgd__alpha": [1e-4, 1e-3],
    "sgd__penalty": ["l2"],
    "sgd__max_iter": [150, 250, 350, 450, 550],
    "sgd__tol": [1e-3],
    "sgd__early_stopping": [True],
}


grid = GridSearchCV(pipeline, param_grid, cv=3, scoring="accuracy", verbose=1)

# 7. Train the model using GridSearchCV
grid = grid.fit(X_train, y_train)

# 8. Print the best hyperparameters
print("Best hyperparameters found by GridSearchCV:")
print(grid.best_params_)

# 9. Get the best estimator from GridSearchCV
best_pipeline = grid.best_estimator_


# 8. Save the pipeline
model_path = "sgd_pipeline.joblib"  # Choose a filename
joblib.dump(best_pipeline, model_path)
print(f"Trained pipeline saved to {model_path}")


# Training sizes: 10% to 100%
train_sizes = np.linspace(0.1, 1.0, 8)
train_losses = []
val_losses = []

final_loss = best_pipeline.get_params()["sgd__loss"]


for frac in train_sizes:
    size = int(frac * len(X_train))
    X_sub = X_train[:size]
    y_sub = y_train[:size]

    model = clone(best_pipeline)
    # the pipeline internally: Applies FeatureExtractor().transform() to X_sub
    # Then StandardScaler().fit_transform(). Then trains SGDClassifier on that.
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

        # Predict margin scores (use the estimator within the pipeline)
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


y_train_pred = best_pipeline.predict(X_train)
y_val_pred = best_pipeline.predict(X_val)
y_test_pred = best_pipeline.predict(X_test)

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

# Extract features before scaling
feature_extractor = best_pipeline.named_steps["features"]
X_train_features = feature_extractor.transform(X_train)

# Print shape of zoning features
zoning_features_example = feature_extractor._extract_zoning_features(X_train[0])
print(f"Shape of zoning features: {np.array(zoning_features_example).shape}")

# Print shape of HOG features
hog_features_example = hog(
    X_train[0], pixels_per_cell=(7, 7), cells_per_block=(2, 2), feature_vector=True
)
print(f"Shape of HOG features: {hog_features_example.shape}")

# Print shape of combined features before scaling
print(f"Shape of X_train features before scaling: {X_train_features.shape}")


# Apply the entire pipeline to X_train and print the shape
X_train_transformed = best_pipeline.transform(X_train)  # Use transform
print(f"Shape of transformed X_train after pipeline: {X_train_transformed.shape}")


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
