import gradio as gr
import numpy as np
import joblib
from PIL import Image
import io

# Load model and scaler
model = joblib.load("/content/xgb_mnist_model.joblib")
scaler = joblib.load("/content/scaler_mnist.joblib")

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Convert user-drawn image to 28x28 grayscale, flatten, and scale
    """
    # Convert to grayscale and resize to 28x28
    image = image.convert("L").resize((28, 28))

        # Convert to numpy array and normalize to [0, 1]
    image_array = np.array(image).astype(np.float32) / 255.0

    # Invert: MNIST digits are white on black background
    image = 1.0 - image_array

    # Flatten and scale
    image = image.reshape(1, -1)  # shape: (1, 784)
    image = scaler.transform(image)
    return image

def predict_digit(image):
    processed = preprocess_image(image)
    prediction = model.predict(processed)[0]
    return f"Predicted digit: {prediction}"

# Create Gradio interface
interface = gr.Interface(
    fn=predict_digit,
    inputs=[gr.Image(type="pil", label="Draw or Upload Digit")],
    outputs="text",
    title="MNIST Digit Classifier with XGBoost",
    description="Draw a digit (0-9) in the box and the model will predict it!"
)

interface.launch(debug=True)
