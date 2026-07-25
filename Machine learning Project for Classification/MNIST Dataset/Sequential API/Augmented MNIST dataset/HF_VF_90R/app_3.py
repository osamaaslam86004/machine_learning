import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model("best_mnist_model (1).keras")

# Define prediction function
def predict_digit(img: Image.Image):
    # Convert to grayscale and resize to 28x28
    img = img.convert("L").resize((28, 28))

    # Convert to numpy array and normalize
    img_array = np.array(img).astype("float32") / 255.0

    # Invert colors: MNIST digits are white on black
    img_array = 1.0 - img_array

    # Reshape to match model input
    img_array = img_array.reshape(1, 28, 28)

    # Predict
    prediction = model.predict(img_array)[0]
    return {str(i): float(prediction[i]) for i in range(10)}


# Gradio interface
interface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="MNIST Digit Classifier",
    description="Draw a digit (0-9) and get the model's prediction."
)

# Launch the app
interface.launch()
