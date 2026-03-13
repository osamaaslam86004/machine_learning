import os

import gradio as gr
import joblib
import numpy as np
from PIL import Image
from skimage.transform import resize

# Load the model from the specified path
pipeline_path = os.path.join(
    r"D:\Machine learning\Arch technologhy\Manual 2\requirement_1\MNIST Digit Recognition Project\results_SGClassifier\step_1_no_aug_no_hog\best_sgd_model.joblib"
)


try:
    # Update the pipeline's SGDClassifier step with the loaded model
    pipeline = joblib.load(pipeline_path)
    print(f"Pre-trained model loaded and pipeline updated from {pipeline_path}")

except FileNotFoundError:
    print(
        f"No pre-trained model found at {pipeline_path}. Using a default initialized SGDClassifier."
    )
    # You might want to train the pipeline here if no model is found:
    # pipeline.fit(X_train, y_train) # replace with your training data


def predict_digit(image):
    """Predicts the digit from a 28x28 grayscale image and returns decision scores."""

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
        image = resize(image, (28, 28))

    # Reshape and flatten for the pipeline
    image = image.reshape(1, 28 * 28)

    # prediction
    prediction = pipeline.predict(image)[0]

    return str(prediction)


# Create a Gradio interface
demo = gr.Interface(
    fn=predict_digit,
    inputs=[gr.Image(type="pil", label="Draw or Upload Digit")],
    outputs=[gr.Label(num_top_classes=1, label="Prediction")],
    title="MNIST Digit Recognition with Decision Scores",
    description="Image to recognize the digit and see decision scores for each class.",
)


demo.launch(share=True, pwa=True)
