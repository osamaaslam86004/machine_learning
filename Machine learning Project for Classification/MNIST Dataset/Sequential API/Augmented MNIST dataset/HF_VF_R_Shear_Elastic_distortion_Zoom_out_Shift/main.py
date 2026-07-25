# -------------------- 1. Imports --------------------
import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import confusion_matrix, classification_report
from skimage.feature import hog

from torchvision import datasets, transforms

# -------------------- 2. Best params from Optuna --------------------
best_params = {
    'n_layers': 2,
    'n_units_l0': 438,
    'dropout_l0': 0.1625,
    'n_units_l1': 453,
    'dropout_l1': 0.2005,
    'optimizer': 'Adam',
    'learning_rate': 0.0002279,
    'batch_size': 64
}

# -------------------- 3. HOG Feature Extractor --------------------
class HOGFeatureExtractor:
    def __init__(self, pixels_per_cell=(7, 7), cells_per_block=(2, 2)):
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

    def transform(self, X):
        return np.array([
            hog(img,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                feature_vector=True)
            for img in X
        ])

# -------------------- 4. Load Augmented Training Data --------------------
df = pd.read_csv("augmented_dataset/labels.csv")
X, y = [], []

for _, row in df.iterrows():
    path = os.path.join("augmented_dataset", row["filename"])
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        X.append(img)
        y.append(int(row["label"]))
X = np.array(X)
y = np.array(y)

# Extract HOG features
hog_extractor = HOGFeatureExtractor()
X_hog = hog_extractor.transform(X)
y_cat = to_categorical(y, 10)

# -------------------- 5. Load Test Set from Torchvision --------------------
transform = transforms.Compose([
    transforms.ToTensor(),
])
torch_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

x_test = np.array([img.squeeze(0).numpy() for img, _ in torch_test])
y_test = np.array([label for _, label in torch_test])
y_test_cat = to_categorical(y_test, 10)

X_test_hog = hog_extractor.transform(x_test)

# -------------------- 6. Build Model --------------------
model = Sequential()
model.add(Input(shape=(X_hog.shape[1],)))
model.add(Dense(best_params['n_units_l0'], activation='relu'))
model.add(Dropout(best_params['dropout_l0']))
model.add(Dense(best_params['n_units_l1'], activation='relu'))
model.add(Dropout(best_params['dropout_l1']))
model.add(Dense(10, activation='softmax'))

model.compile(
    optimizer=Adam(learning_rate=best_params['learning_rate']),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------- 7. Callbacks --------------------
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
checkpoint = ModelCheckpoint("best_hog_mlp_model.keras", monitor='val_loss', save_best_only=True, verbose=1)

# -------------------- 8. Train --------------------
history = model.fit(
    X_hog, y_cat,
    batch_size=best_params['batch_size'],
    epochs=30,
    validation_split=0.1,
    callbacks=[early_stop, checkpoint],
    verbose=2
)

# Optionally Save Final Model (already saved best via checkpoint)
# model.save("best_hog_mlp_model.keras")

# -------------------- 9. Evaluate on Test Set --------------------
test_loss, test_acc = model.evaluate(X_test_hog, y_test_cat, verbose=0)
print(f"\n✅ Test Accuracy: {test_acc:.4f}, Loss: {test_loss:.4f}")

# -------------------- 10. Confusion Matrix & Classification Report --------------------
y_pred = np.argmax(model.predict(X_test_hog), axis=1)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

# -------------------- 11. Training History --------------------
def plot_history(history):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    axs[0].plot(history.history['accuracy'], label='Train')
    axs[0].plot(history.history['val_accuracy'], label='Val')
    axs[0].set_title('Accuracy')
    axs[0].legend()

    axs[1].plot(history.history['loss'], label='Train')
    axs[1].plot(history.history['val_loss'], label='Val')
    axs[1].set_title('Loss')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

plot_history(history)
