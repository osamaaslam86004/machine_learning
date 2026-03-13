import numpy as np
from sklearn.datasets import fetch_openml

from load_split_data import load_split_data
from OvO_sgc_binary import evaluate_sg_classifier_one_vs_one_strategy
from OvR_sgc_binary import evaluate_sg_classifier_one_vs_rest_strategy
from randomforest_binary import train_evaluate_random_forest_binary_classifier
from randomforest_multi_class import train_evaluate_random_forest_multiclass_classifier
from sgc_binary import train_evaluate_sgd_binary_classifier

# ## Setup and Data Loading
#
# Loading the MNIST dataset using Scikit-Learn's `fetch_openml` as shown in the source.
mnist = fetch_openml("mnist_784", version=1)
X, y = mnist["data"], mnist["target"]
y = y.astype(np.uint8)
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]


# To confirm that the MNIST dataset has a relatively balanced distribution of digits
#
# Count the occurrences of each digit in the training set
digit_counts = np.bincount(y_train)

# Print the counts for each digit
for digit, count in enumerate(digit_counts):
    print(f"Digit {digit}: {count} occurrences")

# Calculate and print the percentage of each digit
total_count = len(y_train)
for digit, count in enumerate(digit_counts):
    percentage = (count / total_count) * 100
    print(f"Digit {digit}: {percentage:.2f}%")


if __name__ == "__main__":

    X_train, X_test, y_train, y_test = load_split_data()

    # Binary Classification (SGD)
    print("\nTraining and evaluating SGDClassifier (Binary 5-detector)...")
    train_evaluate_sgd_binary_classifier(X_train, X_test, y_train, y_test)

    # Binary Classification (RandomForest) with OvR-strategy
    print("\nTraining and evaluating RandomForest Classifier (Binary 5-detector)...")
    train_evaluate_random_forest_binary_classifier(X_train, X_test, y_train, y_test)

    print("\nSGDClassifier (Multiclass 0-9) with OvR-strategy:")
    evaluate_sg_classifier_one_vs_rest_strategy(X_train, y_train, X_test, y_test)

    # Multiclass Classification (RandomForest) with OvR-strategy
    print("\nSGDClassifier (Multiclass 0-9) with OvO-strategy:")
    evaluate_sg_classifier_one_vs_one_strategy(X_train, y_train, X_test, y_test)

    # Multiclass Classification (RandomForest) with OvR-strategy
    print("\nRandomForest Classifier (Multiclass 0-9) with OvR-strategy:")
    train_evaluate_random_forest_multiclass_classifier(X_train, X_test, y_train, y_test)
