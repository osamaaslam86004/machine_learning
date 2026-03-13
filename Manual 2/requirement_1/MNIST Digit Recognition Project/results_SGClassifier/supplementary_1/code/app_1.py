import os

import joblib
import numpy as np
from PIL import Image, ImageEnhance

import gradio as gr

machine_type = True

# Initlizing variables
sgd_fine_tune_model = None
sgd_fine_tune_pca = None
sgd_fine_tune_scaler = None

random_forest_fine_tune_model = None
random_forest_fine_tune_pca = None


if machine_type:

    def load_pca_model_scaler_for_local_machine():
        global sgd_fine_tune_pca, sgd_fine_tune_model, sgd_fine_tune_scaler
        global random_forest_fine_tune_pca, random_forest_fine_tune_model

        # Define the path to the directory containing the saved files
        sgd_classifi_rresults_dir = r"D:\Machine learning\Arch technologhy\Manual 2\requirement_1\MNIST Digit Recognition Project\results_SGClassifier"

        random_forest_results_dir = r"D:\Machine learning\Arch technologhy\Manual 2\requirement_1\MNIST Digit Recognition Project\results_RandomForestClassifier"

        # Loading fine tune model
        try:
            sgd_fine_tune_model = joblib.load(
                os.path.join(
                    sgd_classifi_rresults_dir, "best_sgd_ovr_fine-tune_model.joblib"
                )
            )
            sgd_fine_tune_pca = joblib.load(
                os.path.join(sgd_classifi_rresults_dir, "pca_fine-tune.joblib")
            )
            sgd_fine_tune_scaler = joblib.load(
                os.path.join(sgd_classifi_rresults_dir, "scaler_fine_tune.joblib")
            )
            print("Loaded the fine-tuned model, PCA, and Scaler successfully.")

        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Required file not found: {e}. Please ensure the model, PCA, and Scaler are trained and saved correctly in the specified directory."
            )

        try:
            random_forest_fine_tune_model = joblib.load(
                os.path.join(
                    random_forest_results_dir, "best_rf_ovr_fine-tune_model.joblib"
                )
            )
            random_forest_fine_tune_pca = joblib.load(
                os.path.join(random_forest_results_dir, "pca_fine_tune_0.99.joblib")
            )  # produces 154 features
            print("✅ Loaded the trained (non-augmented) model, PCA, and Scaler.")

        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Required file not found: {e}. Please ensure the model, PCA, and Scaler are trained and saved correctly in the specified directory."
            )

        return (
            sgd_fine_tune_pca,
            sgd_fine_tune_model,
            sgd_fine_tune_scaler,
            random_forest_fine_tune_pca,
            random_forest_fine_tune_model,
        )

else:

    def load_pca_model_scaler_for_colab_notebook():
        global sgd_fine_tune_pca, sgd_fine_tune_model, sgd_fine_tune_scaler
        global random_forest_fine_tune_pca, random_forest_fine_tune_model

        # Define the path to the directory containing the saved files
        #  Using Google Drive for persistent storage
        results_dir = "/content/"  # Or your preferred path in Google Drive
        sgd_classifi_rresults_dir = os.path.join(results_dir, "results_SGClassifier")
        random_forest_results_dir = os.path.join(
            results_dir, "results_RandomForestClassifier"
        )

        # Loading fine tune model
        try:
            sgd_fine_tune_model = joblib.load(
                os.path.join(
                    sgd_classifi_rresults_dir, "best_sgd_ovr_fine-tune_model.joblib"
                )
            )
            sgd_fine_tune_pca = joblib.load(
                os.path.join(sgd_classifi_rresults_dir, "pca_fine-tune.joblib")
            )
            sgd_fine_tune_scaler = joblib.load(
                os.path.join(sgd_classifi_rresults_dir, "scaler_fine_tune.joblib")
            )
            print("Loaded the fine-tuned model, PCA, and Scaler successfully.")

        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Required file not found: {e}. Please ensure the model, PCA, and Scaler are trained and saved correctly in the specified directory."
            )

        try:
            random_forest_fine_tune_model = joblib.load(
                os.path.join(
                    random_forest_results_dir, "best_rf_ovr_fine-tune_model.joblib"
                )
            )
            random_forest_fine_tune_pca = joblib.load(
                os.path.join(random_forest_results_dir, "pca_fine_tune_0.99.joblib")
            )  # produces 154 features
            print("✅ Loaded the trained (non-augmented) model, PCA, and Scaler.")

        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Required file not found: {e}. Please ensure the model, PCA, and Scaler are trained and saved correctly in the specified directory."
            )

        return (
            sgd_fine_tune_pca,
            sgd_fine_tune_model,
            sgd_fine_tune_scaler,
            random_forest_fine_tune_pca,
            random_forest_fine_tune_model,
        )


def convert_to_black_and_white(image):
    """Converts a PIL Image to black and white (white digit on black background)
       and reshapes it to 28x28 pixels.

    Args:
        image: A PIL Image object.

    Returns:
        A PIL Image object that is black and white (white digit on black background)
        and has dimensions 28x28.
    """
    # # Convert to grayscale
    if image.mode != "L":
        image = image.convert("L")

    # Enhance the contrast of the grayscale image
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2000)  # Increase contrast (you can adjust the factor)

    brightness = ImageEnhance.Brightness(image)
    image = brightness.enhance(3.5)  # Increase brightness (you can adjust the factor)

    # Resize to 28x28
    image = image.resize((28, 28), Image.LANCZOS)

    # return bw_image
    return image


# Function to preprocess the image and make a prediction
def predict_digit(image, model_choice):
    """
    Predicts the digit in a user-uploaded image using the fine-tuned model.

    Args:
        image: A PIL Image object of a handwritten digit (0-9).

    Returns:
        A dictionary with the predicted digit as key and the confidence score as value.
    """

    if image is None:
        return "No image uploaded"

        # Ensure image is a PIL Image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))

    # 1. Convert to grayscale and resize to 28x28 (MNIST standard)
    image = convert_to_black_and_white(image)

    # 2. Convert to numpy array
    image_np = np.array(image).astype("float64")

    # 3. Flatten the image
    image_flattened = image_np.flatten().reshape(1, -1)

    if machine_type:
        (
            sgd_fine_tune_pca,
            sgd_fine_tune_model,
            sgd_fine_tune_scaler,
            random_forest_fine_tune_pca,
            random_forest_fine_tune_model,
        ) = load_pca_model_scaler_for_local_machine()
    else:
        (
            sgd_fine_tune_pca,
            sgd_fine_tune_model,
            sgd_fine_tune_scaler,
            random_forest_fine_tune_pca,
            random_forest_fine_tune_model,
        ) = load_pca_model_scaler_for_colab_notebook()

    # Select appropriate model pipeline
    if model_choice == "SGDClassifier":
        model = sgd_fine_tune_model
        pca = sgd_fine_tune_pca
        scaler = sgd_fine_tune_scaler
    else:
        model = random_forest_fine_tune_model
        pca = random_forest_fine_tune_pca

    # 4. Apply PCA
    image_pca = pca.transform(image_flattened)

    # 5. Apply StandardScaler
    if model_choice == "SGDClassifier":
        image_scaled = scaler.transform(image_pca)
    else:
        image_scaled = image_pca  # No scaler for RandomForest in this setup

    # Get decision scores or probabilities
    if model_choice == "SGDClassifier":

        decision_scores = model.decision_function(image_scaled)
        top_class = str(np.argmax(decision_scores))

    else:
        prediction_prob = model.predict_proba(image_scaled)
        print(f"model.predict_proba(image_scaled): {prediction_prob}")

    # Create label dictionary
    if model_choice == "SGDClassifier":
        return top_class

    result = {str(i): float(prediction_prob[0][i]) for i in range(10)}
    print(f"result RandomForest: {result}")

    return result


# Create a Gradio interface
demo = gr.Interface(
    fn=lambda image, model_choice: predict_digit(image, model_choice),
    inputs=[
        gr.Image(type="pil", label="Draw or Upload Digit"),
        gr.Radio(
            ["SGDClassifier", "RandomForestClassifier"],
            label="Select Model",
            # value sets the default selection for the radio button
            value="SGDClassifier",
        ),
    ],
    outputs=gr.Label(num_top_classes=10),
    title="MNIST Digit Recognizer Demo",
    description="Choose between 'SGDClassifier' and 'RandomForestClassifier' to classify digits from 0–9.",
)

if not machine_type:
    raise EnvironmentError("This function should only be run in a Colab environment.")

demo.launch(share=True, pwa=True)
