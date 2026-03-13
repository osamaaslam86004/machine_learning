import cv2
from skimage.feature import hog
import gradio as gr
from tensorflow.keras.models import load_model
from PIL import Image, ImageEnhance, ImageOps
import pandas as pd
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import io

# Load trained MLP model
model = load_model("best_hog_mlp_model.keras")

# HOG feature extractor
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

hog_extractor = HOGFeatureExtractor()

def predict_digit(img):
    # Convert from PIL to NumPy
    img = np.array(img)

    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize to 28x28
    img = cv2.resize(img, (28, 28))

    # Invert colors (white digit on black background)
    img = cv2.bitwise_not(img)

    # Normalize to [0, 1]
    img = img.astype("float32") / 255.0

    # Extract HOG features
    features = hog_extractor.transform([img])

    # Predict
    preds = model.predict(features)[0]
    top_class = np.argmax(preds)

    # Return class-wise probabilities
    return {str(i): float(preds[i]) for i in range(10)}


# Gradio interface
interface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="MNIST Digit Classifier (HOG + MLP)",
    description="Draw a digit (0-9) and the model will classify it using HOG features + MLP (92.85% accuracy)."
)

interface.launch(share=True, debug=True)
