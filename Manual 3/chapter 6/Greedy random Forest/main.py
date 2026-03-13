# STEP 0: Setup
!pip install -q rgf_python

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import joblib
import zipfile
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, mean_squared_error, mean_absolute_error
)
from rgf.sklearn import RGFClassifier
from google.colab import files

# STEP 1: Load and Augment MNIST
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X, y = mnist["data"], mnist["target"].astype(np.uint8)
X = X.reshape(-1, 28, 28)

def augment_image(img):
    return [cv2.flip(img, 1), cv2.flip(img, 0), cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)]

X_aug, y_aug = [], []
for i in range(len(X)):
    img = X[i]
    label = y[i]
    X_aug.append(img)
    y_aug.append(label)
    for aug in augment_image(img):
        X_aug.append(aug)
        y_aug.append(label)
    if (i + 1) % 10000 == 0:
        print(f"Processed: {i+1} / {len(X)}")

X_aug = np.array(X_aug).reshape(-1, 784)
y_aug = np.array(y_aug)

# STEP 2: Train/Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_aug, y_aug, test_size=0.2, stratify=y_aug, random_state=42
)

# STEP 3: Train RGFClassifier
clf = RGFClassifier(max_leaf=500, n_iter=50, min_samples_leaf=10, algorithm="RGF", verbose=True)
clf.fit(X_train, y_train)

# STEP 4: Learning Curve
train_sizes, train_scores, val_scores = learning_curve(
    clf, X_train, y_train, cv=3, scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
)
train_acc = train_scores.mean(axis=1)
val_acc = val_scores.mean(axis=1)

plt.plot(train_sizes, train_acc, label="Train Accuracy")
plt.plot(train_sizes, val_acc, label="Validation Accuracy")
plt.xlabel("Training Size")
plt.ylabel("Accuracy")
plt.title("RGFClassifier Learning Curve")
plt.grid(True)
plt.legend()
plt.savefig("rgf_classifier_learning_curve.png")
plt.close()

# STEP 5: Evaluation
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

print(f"\nTrain MSE: {mean_squared_error(y_train, y_train_pred):.4f}")
print(f"Train MAE: {mean_absolute_error(y_train, y_train_pred):.4f}")
print(f"Test MSE: {mean_squared_error(y_test, y_test_pred):.4f}")
print(f"Test MAE: {mean_absolute_error(y_test, y_test_pred):.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot(cmap="Oranges")
plt.title("RGFClassifier Test Confusion Matrix")
plt.savefig("rgf_classifier_confusion_matrix.png")
plt.close()

# Classification Report
report = classification_report(y_test, y_test_pred, output_dict=False)
print("\nClassification Report (Test):\n")
print(report)

# STEP 6: Feature Importance
importances = clf.feature_importances_
top_idx = np.argsort(importances)[-20:][::-1]
top_vals = importances[top_idx]
feature_names = [f"pixel_{i}" for i in top_idx]

plt.figure(figsize=(10, 6))
plt.barh(feature_names, top_vals)
plt.gca().invert_yaxis()
plt.title("Top 20 RGF Feature Importances")
plt.xlabel("Importance")
plt.grid(True)
plt.tight_layout()
plt.savefig("rgf_classifier_feature_importance.png")
plt.close()

# Save model
joblib.dump(clf, "rgf_classifier_model.joblib")

# STEP 7: Zip Results
zip_filename = "rgf_outputs.zip"
with zipfile.ZipFile(zip_filename, "w") as zipf:
    for file in [
        "rgf_classifier_model.joblib",
        "rgf_classifier_learning_curve.png",
        "rgf_classifier_confusion_matrix.png",
        "rgf_classifier_feature_importance.png"
    ]:
        zipf.write(file)

# Trigger download
files.download(zip_filename)

# STEP 8: Shut down Colab runtime
import IPython
print("✅ Training complete. Runtime will shut down after file is downloaded.")
IPython.display.display(IPython.display.Javascript('''
  async function shutdown() {
    await google.colab.kernel.invokeFunction('shutdown');
  }
  shutdown();
'''))
