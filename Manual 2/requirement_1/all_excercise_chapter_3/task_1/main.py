import numpy as np
from evaluate_fine_tune_model import evaluate_fine_tune_model
from evaluate_model import evaluate_model
from fine_tune_model import fine_tune_model
from load_split_data import load_split_data
from sklearn.datasets import fetch_openml
from train_model import train_model

# ## Setup and Data Loading
#
# Loading the MNIST dataset using Scikit-Learn's `fetch_openml` as shown in the source.
mnist = fetch_openml("mnist_784", version=1)
X, y = mnist["data"], mnist["target"]
y = y.astype(np.uint8)
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]


if __name__ == "__main__":

    X_train, X_test, y_train, y_test = load_split_data()

    # Multiclass Classification (KNeighborsClassifier) with OvR-strategy
    print("\nTraining KNeighborsClassifier (Multiclass 0-9):")
    model = train_model(X_train, y_train)

    # Evaluate KNeighborsClassifier
    print("\nEvaluating KNeighborsClassifier (Multiclass 0-9):")
    evaluate_model(model, X_test, y_test)

    # Fine Tune KNeighborsClassifier
    print("\nFine-Tune KNeighborsClassifier (Multiclass 0-9):")
    best_fine_tune_model = fine_tune_model(model, X_train, y_train)

    # Evaluate Fine-Tune KNeighborsClassifier
    print("\nEvaluate Fine-Tune KNeighborsClassifier (Multiclass 0-9):")
    evaluate_fine_tune_model(best_fine_tune_model, X_test, y_test)
