import gradio as gr
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import hog
from sklearn.base import BaseEstimator, TransformerMixin

# Step 1: Upload an RGB image


# ---------------------- Custom Transformers ----------------------


class ZoningFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, zones=(4, 4)):
        self.zones = zones

    def extract_zoning_features(self, img):
        h, w = img.shape
        zh, zw = self.zones
        zone_features = []
        for i in range(zh):
            for j in range(zw):
                block = img[
                    i * h // zh : (i + 1) * h // zh, j * w // zw : (j + 1) * w // zw
                ]
                zone_features.append(block.mean())
        return zone_features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self.extract_zoning_features(img) for img in X])


class HOGFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, pixels_per_cell=(7, 7), cells_per_block=(2, 2)):
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array(
            [
                hog(
                    img,
                    pixels_per_cell=self.pixels_per_cell,
                    cells_per_block=self.cells_per_block,
                    feature_vector=True,
                )
                for img in X
            ]
        )


class FeatureCombiner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.hog_extractor = HOGFeatureExtractor()
        self.zoning_extractor = ZoningFeatureExtractor()

    def fit(self, X, y=None):
        self.hog_extractor.fit(X)
        self.zoning_extractor.fit(X)
        return self

    def transform(self, X):
        hog_feats = self.hog_extractor.transform(X)
        zoning_feats = self.zoning_extractor.transform(X)
        return np.hstack([hog_feats, zoning_feats])


# Load the model from the specified path
model_path = "/content/mnist_sgd_pipeline.joblib"


try:
    # Update loaded model
    pipeline = joblib.load(model_path)
    print(f"model successfully loaded from {model_path}")


except FileNotFoundError:
    print(f"No model found at {model_path}. Using a default initialized SGDClassifier.")
    # You might want to train the pipeline here if no model is found:
    # pipeline.fit(X_train, y_train) # replace with your training data


def predict_digit(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))

    # Step 2: Convert uploaded image to grayscale and resize to 28x28
    if image.mode != "L":
        image = image.convert("L")
    image = image.resize((28, 28))  # Resize to MNIST size

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

    prediction = pipeline.predict(image_np)[0]
    confidence_scores = pipeline.decision_function(image_np)[0]
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


# import cv2
# import gradio as gr
# import joblib
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from PIL import Image, ImageEnhance, ImageOps
# from skimage.feature import hog
# from sklearn.base import BaseEstimator, TransformerMixin

# # Load the model from the specified path
# model_path = "/content/mnist_sgd_pipeline.joblib"


# try:
#     # Update loaded model
#     pipeline = joblib.load(model_path)
#     print(f"model successfully loaded from {model_path}")


# except FileNotFoundError:
#     print(f"No model found at {model_path}. Using a default initialized SGDClassifier.")
#     # You might want to train the pipeline here if no model is found:
#     # pipeline.fit(X_train, y_train) # replace with your training data


# def pixelate(image, block_size=2):
#     """Pixelates an image by averaging pixel values in blocks.

#     Args:
#       image: A NumPy array representing the image.
#       block_size: The size of the pixelation blocks.  Larger values result in more pixelation.

#     Returns:
#       A NumPy array representing the pixelated image.
#     """
#     height, width = image.shape
#     pixelated = np.zeros_like(image)

#     for y in range(0, height, block_size):
#         for x in range(0, width, block_size):
#             # Extract the block
#             y_end = min(y + block_size, height)
#             x_end = min(x + block_size, width)
#             block = image[y:y_end, x:x_end]

#             # Calculate the average value of the block
#             average = np.mean(block)

#             # Fill the block in the pixelated image with the average value
#             pixelated[y:y_end, x:x_end] = average

#     return pixelated


# def preprocess_to_mnist_style(pil_image):
#     # Step 1: Convert to grayscale
#     gray = pil_image.convert("L")

#     # Step 2: Invert if needed (MNIST digits are white on black)
#     gray = ImageOps.invert(gray)

#     # Enhance the contrast of the grayscale image
#     enhancer = ImageEnhance.Contrast(gray)
#     image = enhancer.enhance(2000)  # Increase contrast (you can adjust the factor)

#     brightness = ImageEnhance.Brightness(image)
#     image = brightness.enhance(300.5)  # Increase brightness (you can adjust the factor)

#     # Step 3: Apply binary thresholding
#     bw = image.point(lambda x: 255 if x > 50 else 0, "1")

#     # Convert to NumPy for further processing
#     np_img = np.array(bw).astype(np.uint8)

#     # Step 4: Force horizontal and vertical pixelation (remove curves)
#     # Optional step to pixelate anti-aliased borders
#     # You can use morphological operations like dilation or blocky filters if needed
#     # For now we skip that but you can manually add it

#     pixelated_img = pixelate(np_img, block_size=4)  # Apply pixelation

#     # Step 5: Crop to bounding box of the digit
#     coords = np.argwhere(pixelated_img)
#     if coords.size == 0:
#         return Image.fromarray(np.zeros((28, 28), dtype="float64"))  # return blank

#     y0, x0 = coords.min(axis=0)
#     y1, x1 = coords.max(axis=0) + 1  # slices are exclusive at the top
#     cropped = np_img[y0:y1, x0:x1]

#     # Step 6: Resize to fit in 20x20 box (MNIST standard)
#     cropped_img = Image.fromarray(cropped * 255).resize((20, 20), Image.NEAREST)

#     # Step 7: Paste onto 28x28 black canvas and center
#     new_img = Image.new("L", (28, 28), 0)
#     upper_left = ((28 - 20) // 2, (28 - 20) // 2)
#     new_img.paste(cropped_img, upper_left)

#     # 2. Convert to numpy array
#     new_img = np.array(new_img).astype("float64")

#     # Calculate the new dimensions after 800% zoom (8x increase)
#     height, width = new_img.shape
#     new_height = int(height * 8)
#     new_width = int(width * 8)

#     # Resize the image using INTER_LINEAR for good quality upscaling
#     zoomed_image = cv2.resize(
#         new_img, (new_width, new_height), interpolation=cv2.INTER_LINEAR
#     )

#     cv2.imwrite("mnist_style_digit.png", zoomed_image)

#     return new_img


# def predict_digit(image):
#     # Convert to grayscale if not already
#     if isinstance(image, np.ndarray):
#         image = Image.fromarray(image.astype(np.uint8))

#     # Step 2: Convert uploaded image to resize to 28x28
#     image = image.resize((28, 28))  # Resize to MNIST size

#     mnist_img = preprocess_to_mnist_style(image)  # returns 28x28 np array

#     # Shape must be (1, 28, 28) for the pipeline to extract features
#     image_np = mnist_img.reshape(1, 28, 28)

#     # Predict using pipeline (which includes feature extraction)
#     prediction = pipeline.predict(image_np)[0]
#     confidence_scores = pipeline.decision_function(image_np)[0]

#     return str(prediction), confidence_scores.tolist()


# # Create a Gradio interface
# demo = gr.Interface(
#     fn=predict_digit,
#     inputs=[gr.Image(type="pil", label="Draw or Upload Digit")],
#     outputs=[
#         gr.Label(num_top_classes=1, label="Prediction"),
#         gr.Textbox(label="Confidence Scores for All Classes"),
#     ],
#     title="MNIST Digit Recognition",
#     description="Image to recognize the digit.",
# )


# demo.launch(share=True, pwa=True, debug=True)
