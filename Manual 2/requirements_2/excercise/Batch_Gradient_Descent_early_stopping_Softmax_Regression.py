import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# 1. Softmax function
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # for numerical stability
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# 2. One-hot encoding
def one_hot(y, num_classes):
    return np.eye(num_classes)[y]


# 3. Cross-entropy loss
def cross_entropy(y_true, y_pred):
    m = y_true.shape[0]
    epsilon = 1e-15  # avoid log(0)
    loss = -np.sum(y_true * np.log(y_pred + epsilon)) / m
    return loss


# 4. Predict function
def predict(X, W, b):
    logits = X @ W + b
    return softmax(logits)


# 5. Training function with batch gradient descent and early stopping
def train_softmax(
    X_train, y_train, X_val, y_val, num_classes, lr=0.1, epochs=500, patience=10
):

    m, n = X_train.shape
    W = np.zeros((n, num_classes))
    b = np.zeros((1, num_classes))

    y_train_oh = one_hot(y_train, num_classes)
    y_val_oh = one_hot(y_val, num_classes)

    best_val_loss = float("inf")
    best_W, best_b = None, None
    patience_counter = 0

    for epoch in range(epochs):
        # Forward pass
        logits = X_train @ W + b
        probs = softmax(logits)

        # Loss
        loss = cross_entropy(y_train_oh, probs)

        # Gradient computation
        dW = (X_train.T @ (probs - y_train_oh)) / m
        db = np.mean(probs - y_train_oh, axis=0, keepdims=True)

        # Gradient descent update
        W -= lr * dW
        b -= lr * db

        # Validation
        val_probs = predict(X_val, W, b)
        val_loss = cross_entropy(y_val_oh, val_probs)

        print(f"Epoch {epoch+1} - Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_W, best_b = W.copy(), b.copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

    return best_W, best_b


if __name__ == "__main__":

    # Load a dataset
    data = load_iris()
    X, y = data.data, data.target
    num_classes = len(np.unique(y))

    # Preprocess
    X = StandardScaler().fit_transform(X)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    W, b = train_softmax(X_train, y_train, X_val, y_val, num_classes)

    # Predict
    def predict_classes(X, W, b):
        probs = predict(X, W, b)
        return np.argmax(probs, axis=1)

    y_pred = predict_classes(X_val, W, b)
    print(f"Predictions: {y_pred} --- True Labels: {y_val}")

    # Compute classification accuracy by  Compares predictions to actual labels and computes
    print("Accuracy:", np.mean(y_pred == y_val))

    # Computes cross-entropy loss using:
    # One-hot-encoded true labels.
    # Predicted probabilities from predict(X_val, W, b)
    print(
        "Validation Loss:",
        cross_entropy(one_hot(y_val, num_classes), predict(X_val, W, b)),
    )
