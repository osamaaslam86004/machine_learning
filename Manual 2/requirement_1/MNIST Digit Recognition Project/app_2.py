import argparse
import os
from builtins import EnvironmentError

import cv2
import joblib
import numpy as np
from PIL import Image, ImageEnhance

import gradio as gr


def add_bool_arg(parser, name, default, help):
    parser.add_argument(f"--{name}", dest=name, action="store_true", help=help)
    parser.add_argument(
        f"--no-{name}", dest=name, action="store_false", help=f"Do not {help}"
    )
    parser.set_defaults(**{name: default})


#     You can run the script with different combinations of arguments. Here are some examples:
# python app.py: This will run the script with default values: local_machine=True and select_MNIST_like=False.
# python app.py --local_machine: This will set local_machine=True (as it defaults to True) and select_MNIST_like=False (default).
# python app.py --no-local_machine: This sets local_machine=False and select_MNIST_like=False (default).
# python app.py --select_MNIST_like: This will run the script with local_machine=True (default) and select_MNIST_like=True.
# python app.py --no-select_MNIST_like: This will run the script with local_machine=True (default) and select_MNIST_like=False. This is the same as running python app.py with no arguments related to select_MNIST_like.
# python app.py --local_machine --select_MNIST_like: This will run the script with both flags set to True.
# python app.py --no-local_machine --no-select_MNIST_like: This will run the script with both flags set to False.
# Remember that the order of arguments does not matter. For instance, python app.py --select_MNIST_like --no-local_machine is equivalent to python app.py --no-local_machine --select_MNIST_like.
parser = argparse.ArgumentParser(description="Run the digit recognition app.")
add_bool_arg(
    parser,
    "local_machine",
    True,
    "Set to True if running on a local machine, False for Colab.",
)
# add_bool_arg(
#     parser,
#     "select_MNIST_like",
#     False,
#     "Set to True to convert images to MNIST-like format, False otherwise.",
# )

args = parser.parse_args()

machine_type = True


if machine_type:

    def load_pca_model_scaler_for_local_machine():

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


def convert_like_mnist_dataset(image):
    """
    Convert any digit image into MNIST-style format.

    This script reads a grayscale digit image (e.g., scanned handwritten digit),
    and transforms it into a 28x28 pixel image with the following characteristics:

        - Digit is white (pixel value 255) on a black background (pixel value 0)
        - Digit is centered both vertically and horizontally
        - Image size is exactly 28x28 pixels
        - Digit is resized to fit in a 20x20 box while preserving aspect ratio
        - Padding is added to maintain MNIST-style layout

    Steps:
        1. Load the input image in grayscale
        2. Apply thresholding to extract the digit
        3. Find the bounding box around the digit
        4. Resize the digit to 20x20 while preserving its aspect ratio
        5. Center the digit in a 28x28 black image
        6. Invert colors so the digit becomes white on black background
        7. Save and display the result

    Args:
        image: A PIL Image object.

    Returns:
        A PIL Image object that is black and white (white digit on black background)
        and has dimensions 28x28.
    """

    # if machine_type == "colab_notebook":
    #     cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    # Ensure pillow image is in grayscale
    if image.mode != "L":
        image = image.convert("L")

    # Convert PIL Image to NumPy array
    img = np.array(image)

    if img is None:
        print("Image not found or couldn't be read.")
        exit()

    # Step 1: Threshold to get white digit on black background
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 2: Find bounding box of the digit
    coords = cv2.findNonZero(binary)
    x, y, w, h = cv2.boundingRect(coords)
    digit = binary[y : y + h, x : x + w]

    # Step 3: Resize digit to fit in 20x20 box while preserving aspect ratio
    h, w = digit.shape
    if h > w:
        new_h = 20
        new_w = int(w * (20 / h))
    else:
        new_w = 20
        new_h = int(h * (20 / w))

    resized_digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Step 4: Paste digit into a 28x28 black image, centered
    final_image = np.zeros((28, 28), dtype=np.uint8)

    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    final_image[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = (
        resized_digit
    )

    # Step 5: Invert the image so the digit is white on black background (like MNIST)
    mnist_style = 255 - final_image

    # Save and show
    cv2.imwrite("mnist_style_digit.png", mnist_style)

    return mnist_style


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
    image = convert_like_mnist_dataset(image)

    # 2. Convert to numpy array
    image_np = np.array(image).astype("float64")

    # 3. Flatten the image
    image_flattened = image_np.flatten().reshape(1, -1)

    global machine_type

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
    else:
        decision_scores = model.predict_proba(image_scaled)

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

    # # Softmax-like normalization (optional but helps for display)
    # def softmax(x):
    #     e_x = np.exp(x - np.max(x))
    #     return e_x / e_x.sum()

    # probs = softmax(decision_scores[0])

    # # Create label dictionary
    # result = {str(i): float(probs[i]) for i in range(10)}
    # return result


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
