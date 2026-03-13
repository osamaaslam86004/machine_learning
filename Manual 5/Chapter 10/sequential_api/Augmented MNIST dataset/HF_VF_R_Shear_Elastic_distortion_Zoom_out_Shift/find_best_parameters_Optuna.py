"""
1. Define a Keras model creation function that takes trial parameters.
2. Use Optuna to tune hyperparameters like:
   -- Number of layers/units
   -- Dropout rate
   -- Optimizer
   -- Learning rate
   -- Batch size
   -- Use the augmented + HOG feature set.
3. Run optuna.study.Study.optimize() to find the best hyperparameters.
"""

import os

import cv2
import numpy as np
import optuna
import pandas as pd
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.utils import to_categorical


# -------------------- HOG Feature Extractor --------------------
class HOGFeatureExtractor:
    def __init__(self, pixels_per_cell=(7, 7), cells_per_block=(2, 2)):
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

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


def build_model(trial, input_dim):
    model = Sequential()

    # Explicit input layer
    model.add(Input(shape=(input_dim,)))

    # Hidden layers
    n_layers = trial.suggest_int("n_layers", 1, 3)
    for i in range(n_layers):
        num_units = trial.suggest_int(f"n_units_l{i}", 64, 512)
        model.add(Dense(num_units, activation="relu"))
        dropout_rate = trial.suggest_float(f"dropout_l{i}", 0.1, 0.5)
        model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(10, activation="softmax"))

    # Optimizer
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

    if optimizer_name == "Adam":
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_name == "RMSprop":
        optimizer = RMSprop(learning_rate=learning_rate)
    else:
        optimizer = SGD(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def objective(trial):
    # ==== 1. Load CSV ====
    df = pd.read_csv("augmented_dataset/labels.csv")

    # ==== 2. Load Images + Labels ====
    X = []
    y = []

    for _, row in df.iterrows():
        path = os.path.join("augmented_dataset", row["filename"])
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # 28x28 grayscale
        if img is not None:
            X.append(img)
            y.append(int(row["label"]))
        else:
            print(f"⚠️ Skipped unreadable file: {row['filename']}")

    X = np.array(X)
    y = np.array(y)

    # ==== 3. Extract HOG Features ====
    hog_extractor = HOGFeatureExtractor()
    X_hog = hog_extractor.transform(X)
    y_cat = to_categorical(y, 10)

    # ==== 4. Split into train/val ====
    X_train, X_val, y_train, y_val = train_test_split(
        X_hog, y_cat, test_size=0.1, random_state=42
    )

    # ==== 5. Build Model ====
    model = build_model(trial, input_dim=X_train.shape[1])

    # ==== 6. Batch size ====
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    early_stop = EarlyStopping(
        patience=3, monitor="val_loss", restore_best_weights=True
    )

    # ==== 7. Train ====
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0,
    )

    # ==== 8. Return best val accuracy ====
    val_acc = max(history.history["val_accuracy"])
    return val_acc


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)  # Run 30 trials (you can adjust)

# Best trial
print("✅ Best hyperparameters:")
for key, value in study.best_params.items():
    print(f"{key}: {value}")

print(f"\n✅ Best validation accuracy: {study.best_value:.4f}")
