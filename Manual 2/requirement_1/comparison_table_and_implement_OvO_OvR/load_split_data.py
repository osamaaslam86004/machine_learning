# ## Setup and Data Loading
#
# Loading the MNIST dataset using Scikit-Learn's `fetch_openml` as shown in the source.
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
