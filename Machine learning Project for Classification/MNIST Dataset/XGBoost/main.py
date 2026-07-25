"""
This script implements a machine learning pipeline for classifying handwritten digits from the MNIST dataset
using the Xtremely Greedy Boosting classifier, XGBClassifier, from the xgboost library. The pipeline includes
data loading, preprocessing, model training, evaluation, and result visualization.

The code performs the following steps:

1. **Load MNIST Dataset**: The MNIST dataset is fetched from OpenML, and the target labels are converted to integers.

2. **Train-Test Split**: The dataset is split into training and test sets, with a smaller test set size to reduce 
     memory usage.

3. **Data Scaling (Optional)**: The features are scaled using StandardScaler to standardize the dataset, which can 
     improve model performance.

4. **XGBoost Classifier Setup**: An instance of the XGBClassifier is created with specified hyperparameters, including
     the tree method, maximum depth, number of estimators, learning rate, evaluation metric, and verbosity level.

5. **Model Training**: The classifier is trained on the training dataset.

6. **Learning Curve Visualization**: The learning curve is plotted to visualize the training and validation accuracy
    as a function of the training set size.

7. **Model Predictions**: Predictions are made on the test set using the trained model.

8. **Confusion Matrix Visualization**: A confusion matrix is generated and displayed to visualize the performance of
     the model on the test set.

9. **Classification Report**: A classification report is printed to provide detailed metrics on the model's 
     performance, including precision, recall, and F1-score.

10. **Model and Scaler Saving**: The trained model and the scaler are saved to files for future use, such as in a Gradio app.

Dependencies:
- numpy
- matplotlib
- scikit-learn
- xgboost
- joblib

Usage:
Run the script in a Python environment with the required libraries installed. The output will include model
training results, visualizations, and saved files containing the trained model and scaler.
"""

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np
import joblib  # For saving model

# Load MNIST
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
y = y.astype(int)

# Train-test split (keep test small to reduce memory use)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=10000, random_state=42
)

# OPTIONAL: Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# XGBoost Classifier
xgb_clf = XGBClassifier(
    tree_method="hist",
    max_depth=6,
    n_estimators=100,
    learning_rate=0.1,
    eval_metric="mlogloss",
    verbosity=1,
)

# Train
xgb_clf.fit(X_train, y_train)

# Learning Curve
train_sizes, train_scores, val_scores = learning_curve(
    xgb_clf,
    X_train,
    y_train,
    cv=3,
    train_sizes=np.linspace(0.1, 1.0, 5),
    scoring="accuracy",
    n_jobs=-1,
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mean, "o-", label="Training Accuracy")
plt.plot(train_sizes, val_mean, "o-", label="Validation Accuracy")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.title("Learning Curve (XGBoost on MNIST)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Predictions
y_pred = xgb_clf.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=xgb_clf.classes_)
disp.plot(cmap="viridis", xticks_rotation="vertical", values_format="d")
plt.title("Confusion Matrix (XGBoost)")
plt.show()

# Classification Report
print(classification_report(y_test, y_pred))

# ✅ Save model and scaler for Gradio app
joblib.dump(xgb_clf, "xgb_mnist_model.joblib")
joblib.dump(scaler, "scaler_mnist.joblib")
