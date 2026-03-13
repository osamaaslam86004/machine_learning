import os

import gdown
import numpy as np
from sklearn.datasets import fetch_openml


def load_split_data():
    """
    1. Load MNIST dataset. The source uses 'mnist_784', version=1
    2. Split data into train and test
    3. Cast labels to integers
    """
    print("Loading MNIST dataset...")

    mnist = fetch_openml(
        "mnist_784", version=1, as_frame=False
    )  # as_frame=False for numpy arrays as in source examples

    X, y = mnist["data"], mnist["target"]
    y = y.astype(np.uint8)  # Cast labels to integers as in source

    # Split data into training and test sets (MNIST is pre-split into first 60k and last 10k)
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    print(
        f"Dataset loaded. Training shape: {X_train.shape}, Test shape: {X_test.shape}"
    )

    return X_train, X_test, y_train, y_test


def load_augmented_data(file_id, output_path):
    """
    Download and load augmented MNIST dataset from Google Drive (.npz format).
    """
    print("🌐 Downloading augmented MNIST data from Google Drive...")

    # Download only if the file does not exist
    if not os.path.exists(output_path):
        gdown.download(id=file_id, output=output_path, quiet=False)

    print("📦 Loading PCA-reduced MNIST data from file...")

    # Load from compressed file
    data = np.load(output_path)

    # Extract arrays
    X_train_aug = data["X_train_aug"]
    X_test = data["X_test"]
    y_train_aug = data["y_train_aug"]
    y_test = data["y_test"]

    print(
        f"✅ Loaded Augmented data. Training shape: {X_train_aug.shape}, Test shape: {y_train_aug.shape}"
    )
    print(
        f"✅ Loaded Augmented data. Testing shape: {X_test.shape}, Test shape: {y_test.shape}"
    )
    return X_train_aug, X_test, y_train_aug, y_test


def load_pca_applied_augmented_data(file_id, output_path):
    """
    Download and load Augmented PCA-reduced MNIST dataset from Google Drive (.npz format).
    """
    print("🌐 Downloading Augmented PCA-reduced MNIST data from Google Drive...")

    # Download only if the file does not exist
    if not os.path.exists(output_path):
        gdown.download(id=file_id, output=output_path, quiet=False)

    print("📦 Loading PCA-reduced MNIST data from file...")

    # Load from compressed file
    data = np.load(output_path)

    # Extract arrays
    X_train_aug_pca = data["X_train_aug_pca"]
    X_test_pca = data["X_test_pca"]
    y_train_aug = data["y_train_aug"]
    y_test = data["y_test"]

    print(
        f"✅ Loaded Augmented PCA reduced data. Training shape: {X_train_aug_pca.shape}, Test shape: {y_train_aug.shape}"
    )
    print(
        f"✅ Loaded Augmented PCA reduced data. Testing shape: {X_test_pca.shape}, Test shape: {y_test.shape}"
    )
    return X_train_aug_pca, X_test_pca, y_train_aug, y_test


def load_pca_applied_data(file_id, output_path):
    """
    Download and load PCA-reduced MNIST dataset from Google Drive (.npz format).
    """
    print("🌐 Downloading PCA-reduced MNIST data from Google Drive...")

    # Download only if the file does not exist
    if not os.path.exists(output_path):
        gdown.download(id=file_id, output=output_path, quiet=False)

    print("📦 Loading PCA-reduced MNIST data from file...")

    # Load from compressed file
    data = np.load(output_path)

    # Extract arrays
    X_train = data["X_train_pca"]
    X_test = data["X_test_pca"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    print(
        f"✅ Loaded PCA data. Training shape: {X_train.shape}, Test shape: {y_train.shape}"
    )
    print(
        f"✅ Loaded PCA data. Testing shape: {X_test.shape}, Test shape: {y_test.shape}"
    )
    return X_train, X_test, y_train, y_test


def count_occurrences_of_each_digit_in_training_set(y_train):
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
