import os

import cv2
import joblib
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

import gradio as gr

# Load the model from the specified path
model_path = os.path.join(
    # r"D:\Machine learning\Arch technologhy\Manual 2\requirement_1\MNIST Digit Recognition Project",
    # "new results",
    # "best_fine_tune_random_forest.joblib",
    r"D:\Machine learning\Arch technologhy\Manual 2\requirement_1\MNIST Digit Recognition Project\results_SGClassifier\best_fine_tune_sgd.joblib"
)


try:
    model = joblib.load(model_path)
    print(f"Model loaded successfully from {model_path}")  # Added a success message
except FileNotFoundError:
    print(
        f"Error: Model file not found at {model_path}. Please check the path."
    )  # Improved error message
    exit()  # Exit if the model can't be loaded


def pixelate(image, block_size=2):
    """Pixelates an image by averaging pixel values in blocks.

    Args:
      image: A NumPy array representing the image.
      block_size: The size of the pixelation blocks.  Larger values result in more pixelation.

    Returns:
      A NumPy array representing the pixelated image.
    """
    height, width = image.shape
    pixelated = np.zeros_like(image)

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            # Extract the block
            y_end = min(y + block_size, height)
            x_end = min(x + block_size, width)
            block = image[y:y_end, x:x_end]

            # Calculate the average value of the block
            average = np.mean(block)

            # Fill the block in the pixelated image with the average value
            pixelated[y:y_end, x:x_end] = average

    return pixelated


def preprocess_to_mnist_style(pil_image):
    # Step 1: Convert to grayscale
    gray = pil_image.convert("L")

    # Step 2: Invert if needed (MNIST digits are white on black)
    gray = ImageOps.invert(gray)

    # Enhance the contrast of the grayscale image
    enhancer = ImageEnhance.Contrast(gray)
    image = enhancer.enhance(2000)  # Increase contrast (you can adjust the factor)

    brightness = ImageEnhance.Brightness(image)
    image = brightness.enhance(300.5)  # Increase brightness (you can adjust the factor)

    # Step 3: Apply binary thresholding
    bw = image.point(lambda x: 255 if x > 50 else 0, "1")

    # Convert to NumPy for further processing
    np_img = np.array(bw).astype(np.uint8)

    # Step 4: Force horizontal and vertical pixelation (remove curves)
    # Optional step to pixelate anti-aliased borders
    # You can use morphological operations like dilation or blocky filters if needed
    # For now we skip that but you can manually add it

    pixelated_img = pixelate(np_img, block_size=4)  # Apply pixelation

    # Step 5: Crop to bounding box of the digit
    coords = np.argwhere(pixelated_img)
    if coords.size == 0:
        return Image.fromarray(np.zeros((28, 28), dtype="float64"))  # return blank

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1  # slices are exclusive at the top
    cropped = np_img[y0:y1, x0:x1]

    # Step 6: Resize to fit in 20x20 box (MNIST standard)
    cropped_img = Image.fromarray(cropped * 255).resize((20, 20), Image.NEAREST)

    # Step 7: Paste onto 28x28 black canvas and center
    new_img = Image.new("L", (28, 28), 0)
    upper_left = ((28 - 20) // 2, (28 - 20) // 2)
    new_img.paste(cropped_img, upper_left)

    # 2. Convert to numpy array
    new_img = np.array(new_img).astype("float64")

    # Calculate the new dimensions after 800% zoom (8x increase)
    height, width = new_img.shape
    new_height = int(height * 8)
    new_width = int(width * 8)

    # Resize the image using INTER_LINEAR for good quality upscaling
    zoomed_image = cv2.resize(
        new_img, (new_width, new_height), interpolation=cv2.INTER_LINEAR
    )

    cv2.imwrite("mnist_style_digit.png", zoomed_image)

    return new_img


def predict_digit(image):
    try:
        # Ensure image is a PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))

        # 1. Convert to grayscale and resize to 28x28 (MNIST standard)
        # image = convert_like_mnist_dataset(image)
        image = preprocess_to_mnist_style(image)

        # 3. Flatten the image
        image_flattened = image.flatten().reshape(1, -1)

        # prediction_prob = model.predict_proba(image_flattened)
        decision_scores = model.decision_function(image_flattened)
        print(f"model.predict_proba(image_scaled): {decision_scores}")

        result = {str(i): float(decision_scores[0][i]) for i in range(10)}
        print(f"result RandomForest: {result}")

        return result

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return "Error"  # Return an error message to the user


# Create a Gradio interface
demo = gr.Interface(
    fn=predict_digit,
    inputs=[gr.Image(type="pil", label="Draw or Upload Digit")],
    outputs=gr.Label(num_top_classes=10),
    title="MNIST Digit Recognition",
    description="Upload a handwritten digit image to recognize the digit.",
)


demo.launch(share=True, pwa=True)
