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

    # ## Setup and Data Loading
    #
    # Loading the MNIST dataset using Scikit-Learn's `fetch_openml`
    # X_train is already flattened by default.
    #
    # X_train.shape = (60000, 784) means 60k images each is (1,784) array
    mnist = fetch_openml(
        "mnist_784", version=1, as_frame=False
    )  # as_frame=False for numpy arrays

    X, y = mnist["data"], mnist["target"]
    y = y.astype(np.uint8)  # Cast labels to integers

    # Split data into training and test sets (MNIST is pre-split into first 60k and last 10k)
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    print(
        f"Dataset loaded. Training shape: {X_train.shape}, Test shape: {X_test.shape}"
    )

    return X_train, X_test, y_train, y_test


def load_augmented_pca_applied_data():
    """
    Download and load PCA-reduced MNIST dataset from Google Drive (.npz format).
    """
    print("🌐 Downloading PCA-reduced MNIST data from Google Drive...")

    # Google Drive file ID from the shareable link
    file_id = "1I31qhsyEcw7E9vx4JcjF6rJ6PKGVkokk"
    output_path = "/content/mnist_pca_95_data.npz"

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
        f"✅ Loaded PCA data. Training shape: {X_train.shape}, Test shape: {X_test.shape}"
    )
    return X_train, X_test, y_train, y_test
