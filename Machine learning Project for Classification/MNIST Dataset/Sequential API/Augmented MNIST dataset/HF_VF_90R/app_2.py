import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import pandas as pd


# model = load_model("best_mnist_model.keras")

model = load_model("best_mnist_model.keras")

# Define prediction function
def predict_digit(image: Image.Image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))

    # Step 2: Convert uploaded image to grayscale and resize to 28x28
    if image.mode != "L":
        image = image.convert("L")
    image = image.resize((28, 28))  # Resize to MNIST size

    # Invert the image here
    image = ImageOps.invert(image)

    # Step 3: Flatten the image
    flattened = np.array(image).reshape(1, -1)  # shape = (1, 784)
    data = pd.DataFrame(flattened)

    # Step 4: Reshape, Normalize, Show, and Save
    for i in range(1):  # Only one image
        sample = np.reshape(
            data.iloc[i].values / 255.0, (28, 28)
        )  # Normalize like MNIST

        # Convert back to image format (0–255) for saving
        img_to_save = Image.fromarray((sample * 255).astype(np.uint8))
        img_to_save.save("mnist_like.png")  # Save as PNG

        print("Image saved as mnist_like.png")

    image_np = sample.reshape(
        1, 28, 28
    )  # Proper input shape: (n_samples, height, width)

    # Predict using the model
    prediction = model.predict(image_np)[0]
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
