import os
import zipfile

import gradio as gr
import joblib
import matplotlib.pyplot as plt
import numpy as np
from google.colab import files
from scipy.ndimage import shift
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

# ---------------------- Custom Transformers ----------------------


class ZoningFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, zones=(4, 4)):
        self.zones = zones

    def extract_zoning_features(self, img):
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

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self.extract_zoning_features(img) for img in X])


class HOGFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, pixels_per_cell=(7, 7), cells_per_block=(2, 2)):
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array(
            [
                hog(
                    img,
                    pixels_per_cell=self.pixels_per_cell,
                    cells_per_block=self.cells_per_block,
                    feature_vector=True,
                )
                for img in X
            ]
        )


class FeatureCombiner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.hog_extractor = HOGFeatureExtractor()
        self.zoning_extractor = ZoningFeatureExtractor()

    def fit(self, X, y=None):
        self.hog_extractor.fit(X)
        self.zoning_extractor.fit(X)
        return self

    def transform(self, X):
        hog_feats = self.hog_extractor.transform(X)
        zoning_feats = self.zoning_extractor.transform(X)
        return np.hstack([hog_feats, zoning_feats])


# ---------------------- Data Preparation ----------------------


def load_and_augment_mnist():
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X, y = mnist["data"], mnist["target"].astype(np.uint8)
    X = X.values if hasattr(X, "values") else X
    X_images = X.reshape(-1, 28, 28)

    # Augment: shift left and right
    X_left = np.array([shift(img, shift=[0, -5], cval=0) for img in X_images])
    X_right = np.array([shift(img, shift=[0, 5], cval=0) for img in X_images])

    # Combine
    X_aug = np.concatenate([X_images, X_left, X_right])
    y_aug = np.concatenate([y, y, y])

    return X_aug, y_aug


# ---------------------- Evaluation & Visualization ----------------------


def plot_learning_curve(model, X_train, y_train, X_val, y_val, loss_name):
    train_sizes = np.linspace(0.1, 1.0, 8)
    train_losses, val_losses = [], []

    for frac in train_sizes:
        size = int(frac * len(X_train))
        X_sub, y_sub = X_train[:size], y_train[:size]
        clf = clone(model)
        clf.fit(X_sub, y_sub)

        if loss_name == "log_loss":
            train_loss = log_loss(y_sub, clf.predict_proba(X_sub))
            val_loss = log_loss(y_val, clf.predict_proba(X_val))
        else:
            train_loss = hinge_loss(y_sub, clf.decision_function(X_sub))
            val_loss = hinge_loss(y_val, clf.decision_function(X_val))

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    plt.figure(figsize=(8, 5))
    plt.plot(
        train_sizes * len(X_train), train_losses, label=f"Train {loss_name}", marker="o"
    )
    plt.plot(
        train_sizes * len(X_train),
        val_losses,
        label=f"Validation {loss_name}",
        marker="s",
    )
    plt.title(f"Learning Curve (SGDClassifier - {loss_name})")
    plt.xlabel("Training Set Size")
    plt.ylabel(loss_name)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"learning_curve_{loss_name}.png")
    plt.show()


def show_confusion(title, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title(title)
    plt.grid(False)
    plt.show()
    return cm


# ---------------------- Pipeline Training ----------------------


def main():
    X_aug, y_aug = load_and_augment_mnist()

    # Split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_aug, y_aug, test_size=0.2, stratify=y_aug, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
    )
    print("input shape after train test split:", X_train.shape)  # during training

    # Define pipeline
    pipeline = Pipeline(
        [
            ("feature_extractor", FeatureCombiner()),
            ("scaler", StandardScaler()),
            ("sgd", SGDClassifier(random_state=42)),
        ]
    )

    # Grid Search
    param_grid = {
        "sgd__loss": ["hinge", "log_loss"],
        "sgd__alpha": [1e-4, 1e-3],
        "sgd__penalty": ["l2"],
        "sgd__max_iter": [150, 250, 350],
        "sgd__tol": [1e-3],
        "sgd__early_stopping": [True],
    }

    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring="accuracy", verbose=1)
    print("input shape in GridseachCV:", X_train.shape)  # during training

    grid.fit(X_train, y_train)
    print("Expecting input shape after grid fit:", X_train.shape)  # during training

    print("Best hyperparameters:", grid.best_params_)

    best_model = grid.best_estimator_
    joblib.dump(best_model, "mnist_sgd_pipeline.joblib")
    print("Pipeline saved to mnist_sgd_pipeline.joblib")

    # Evaluate
    final_loss = grid.best_params_["sgd__loss"]
    plot_learning_curve(best_model, X_train, y_train, X_val, y_val, final_loss)

    for split_name, X_split, y_split in [
        ("Validation", X_val, y_val),
        ("Test", X_test, y_test),
    ]:
        y_pred = best_model.predict(X_split)
        cm = show_confusion(f"{split_name} Confusion Matrix", y_split, y_pred)
        print(
            f"{split_name} classification report:\n",
            classification_report(y_split, y_pred),
        )
        print(f"{split_name} confusion matrix:\n", cm)

    # Extract features before scaling
    feature_extractor = best_model.named_steps["feature_extractor"]
    joblib.dump(feature_extractor, "feature_extractor.joblib")
    print("feature_extractor saved ..........")

    scaler = best_model.named_steps["scaler"]
    joblib.dump(scaler, "scaler.joblib")
    print("scaler saved ...........")

    X_train_features = feature_extractor.transform(X_train)

    # Print shape of zoning features
    zoning_features_example = (
        feature_extractor.zoning_extractor.extract_zoning_features(X_train[0])
    )
    print(f"Shape of zoning features: {np.array(zoning_features_example).shape}")

    # Print shape of HOG features
    hog_features_example = hog(
        X_train[0], pixels_per_cell=(7, 7), cells_per_block=(2, 2), feature_vector=True
    )
    print(f"Shape of HOG features: {hog_features_example.shape}")

    # Print shape of combined features before scaling
    print(f"Shape of X_train features before scaling: {X_train_features.shape}")

    # Apply the entire pipeline to X_train and print the shape
    X_train_transformed = best_model.transform(X_train)  # Use transform
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


if __name__ == "__main__":
    main()
