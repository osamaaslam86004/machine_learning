import gradio as gr
import joblib
import numpy as np
import pandas as pd
from PIL import Image

# Load the model from the specified path
model_path = "/content/best_fine_tune_random_forest.joblib"


try:
    # Update loaded model
    pipeline = joblib.load(model_path)
    print(f"model successfully loaded from {model_path}")


except FileNotFoundError:
    print(
        f"No model found at {model_path}. Using a default initialized Randomforest Classifier."
    )
    # You might want to train the pipeline here if no model is found:
    # pipeline.fit(X_train, y_train) # replace with your training data


def predict_digit(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))

    # Step 1: Convert to grayscale if not already
    if image.mode != "L":
        image = image.convert("L")

    # Step 2: Resize to 28x28
    image = image.resize((28, 28))

    # Step 3: Convert to NumPy array and normalize
    img_array = np.array(image) / 255.0  # shape = (28, 28), float64

    # Save for visualization (optional)
    Image.fromarray((img_array * 255).astype(np.uint8)).save("mnist_like.png")
    print("Image saved as mnist_like.png")

    # Step 4: Flatten to match RandomForest input (1, 784)
    image_flat = img_array.reshape(1, -1)

    # Step 5: Predict
    prediction = pipeline.predict(image_flat)[0]
    confidence_scores = pipeline.predict_proba(image_flat)[0]

    return str(prediction), confidence_scores.tolist()


# Create a Gradio interface
demo = gr.Interface(
    fn=predict_digit,
    inputs=[gr.Image(type="pil", label="Draw or Upload Digit")],
    outputs=[
        gr.Label(num_top_classes=1, label="Prediction"),
        gr.Textbox(label="Confidence Scores for All Classes"),
    ],
    title="MNIST Digit Recognition",
    description="Image to recognize the digit.",
)


demo.launch(share=True, pwa=True, debug=True)
