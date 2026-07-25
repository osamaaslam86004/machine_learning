import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import cv2
from sklearn.utils import shuffle



def augment_image(img):
    """Return horizontal flip, vertical flip, and 90° rotation of the input image"""
    return [
        cv2.flip(img, 1),  # horizontal flip
        cv2.flip(img, 0),  # vertical flip
        cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)  # rotate 90°
    ]

# Set seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


# Augment training set
X_aug, y_aug = [], []
for i in range(len(x_train)):
    img = x_train[i]
    label = y_train[i]

    X_aug.append(img)        # Original
    y_aug.append(label)

    for aug in augment_image((img * 255).astype(np.uint8)):  # Convert to uint8 for OpenCV
        aug = aug.astype("float32") / 255.0                  # Normalize again
        X_aug.append(aug)
        y_aug.append(label)

    if (i + 1) % 10000 == 0:
        print(f"Processed {i+1} / {len(x_train)} images...")

# Convert to NumPy arrays
X_aug = np.array(X_aug)
y_aug = np.array(y_aug)

# Shuffle the augmented data
X_aug, y_aug = shuffle(X_aug, y_aug, random_state=42)

# One-hot encode labels for training
# y_train_cat = to_categorical(y_train, 10)
# One-hot encode labels for augmented data
y_aug_cat = to_categorical(y_aug, 10)
y_test_cat = to_categorical(y_test, 10)

# Save to .npz (compressed format)
np.savez_compressed("augmented_mnist.npz", X=X_aug, y=y_aug)
print("✅ Augmented MNIST saved as 'augmented_mnist.npz'")


# Build the model
model = Sequential([
    tf.keras.Input(shape=(28, 28)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')  # 10-class output
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping callback
early_stop = EarlyStopping(
    monitor='val_loss',     # Watch validation loss
    patience=3,             # Wait for 3 epochs without improvement
    restore_best_weights=True,  # Restore weights from best epoch
    verbose=1
)

# Model checkpoint: save best model based on val_loss
checkpoint = ModelCheckpoint(
    filepath='best_mnist_model.keras',  # .keras format
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Reduce LR when a plateau is detected
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)


# Train the model and save training history
history = model.fit(
    # x_train, y_train_cat,
    X_aug, y_aug_cat,
    epochs=100,                  # Set higher limit; early stop will terminate early
    batch_size=128,
    validation_split=0.1,
    callbacks=[early_stop, checkpoint, reduce_lr],  # 👈 Add all three
    verbose=2
)


# Evaluate on test data
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"\n✅ Test Accuracy: {test_acc:.4f}, Loss: {test_loss:.4f}")

# 🔐 Save the model for Gradio App
# model.save("mnist_gradio_model.keras")
# print("Model saved ............")

# Plot training and validation accuracy/loss
def plot_history(history):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axs[0].plot(history.history['accuracy'], label='Train')
    axs[0].plot(history.history['val_accuracy'], label='Val')
    axs[0].set_title('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()

    # Loss
    axs[1].plot(history.history['loss'], label='Train')
    axs[1].plot(history.history['val_loss'], label='Val')
    axs[1].set_title('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

plot_history(history)

# Confusion matrix
y_pred = np.argmax(model.predict(x_test), axis=1)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Classification Report
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

