import os

import gradio as gr
import joblib
import numpy as np
from PIL import Image
from skimage.transform import resize


def predict_digit(image):
    # Load the model from the specified path
    model_path = os.path.join(
        r"D:\Machine learning\Arch technologhy\Manual 2\requirement_1\MNIST Digit Recognition Project\results_SGClassifier\sgd_pipeline.joblib"
    )

    try:
        # Update the pipeline's SGDClassifier step with the loaded model
        pipeline = joblib.load(model_path)
        print(f"Pre-trained model loaded and pipeline updated from {model_path}")
    except FileNotFoundError:
        print(
            f"No pre-trained model found at {model_path}. Using a default initialized SGDClassifier."
        )
        # You might want to train the pipeline here if no model is found:
        # pipeline.fit(X_train, y_train) # replace with your training data

    """Predicts the digit from a 28x28 grayscale image."""

    # Ensure image is a PIL Image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))

    # Convert to grayscale if not already
    if image.mode != "L":
        image = image.convert("L")

    # Convert to NumPy array
    image = np.array(image)

    # Resize to 28x28 if needed
    if image.shape != (28, 28):
        image = resize(image, (28, 28), anti_aliasing=True)

    # Reshape to (1, 28, 28) for the pipeline
    image = image.reshape(1, 28, 28)

    # Predict using the pipeline
    prediction = pipeline.predict(image)

    return str(prediction[0])


# Create a Gradio interface
demo = gr.Interface(
    fn=predict_digit,
    inputs=[gr.Image(type="pil", label="Draw or Upload Digit")],
    outputs="text",
    title="MNIST Digit Recognition",
    description="Upload a handwritten digit image to recognize the digit.",
)


demo.launch(share=True, pwa=True)
