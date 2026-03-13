import gradio as gr
from tensorflow.keras.models import load_model
from PIL import Image, ImageEnhance, ImageOps
import pandas as pd
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import io


model = load_model("best_mnist_model.keras")

def convert_numpy_image_to_pillow(image):
  """
  To save the preprocessed MNIST-style image (the one passed to the model) for visualization, you can:
  1. Convert the mnist_like_image (which is a NumPy array) to a PIL image.
  2. Scale it from [0, 1] to [0, 255].
  3. Convert it to uint8 and save.
  """
  # ✅ Save the processed image for visualization
  image_to_save = (image * 255).astype(np.uint8)
  image_pil = Image.fromarray(image_to_save)
  image_pil.save("mnist_like_input.png")  # You can change the filename/path


def enhance_images(image):
  # Create an enhancer object for contrast
  factor = 32.5
  enhancer = ImageEnhance.Contrast(image)
  enhanced_image = enhancer.enhance(factor)  # Enhance the contrast

  return enhanced_image


def pad_image_to_square(image):
    """
    Pad the image equally on the shorter side to make it square.
    """
    height, width = image.shape[:2]
    if height == width:
        return image

    diff = abs(height - width)
    pad_before = diff // 2
    pad_after = diff - pad_before

    if height < width:
        padding = ((pad_before, pad_after), (0, 0), (0, 0))
    else:
        padding = ((0, 0), (pad_before, pad_after), (0, 0))

    return np.pad(image, padding, mode='constant', constant_values=0)


def threshold_image(grayscale_image, threshold=0.5):
    """
    Convert grayscale image to binary (0 or 1) using a threshold.
    """
    return (grayscale_image > threshold).astype(float)


def convert_to_mnist_style(image):
    # Pad to square
    image = pad_image_to_square(image)

    # Convert to grayscale using weighted average method
    grayscale_image = image.dot([0.07, 0.72, 0.21])

    # Invert grayscale so digit is white
    inverted = 1.0 - grayscale_image

    # Resize to 28x28
    final_image = resize(inverted, (28, 28), anti_aliasing=True)

    return final_image



# Define prediction function
def predict_digit(image: Image.Image):
    # Read the image using PIL and convert to numpy array
    pil_image = image.convert("RGB")

    pil_image = enhance_images(pil_image)

    image = np.array(pil_image) / 255.0  # Normalize

    mnist_like_image = convert_to_mnist_style(image)

    convert_numpy_image_to_pillow(mnist_like_image)

    # Step 3: Flatten the image
    reshaped = mnist_like_image.reshape(1, 28, 28)  # ✅ no channel dimension

    # Predict using the model
    prediction = model.predict(reshaped)[0]
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
interface.launch(debug=True, share=True)
