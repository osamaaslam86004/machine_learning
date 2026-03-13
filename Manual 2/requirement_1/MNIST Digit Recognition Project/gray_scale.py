# import cv2
# import matplotlib.pyplot as plt
# import numpy as np

# # Step 1: Load your image
# img = cv2.imread("download_(2).jpg", cv2.IMREAD_GRAYSCALE)

# # Step 2: Binarize (white digit, black background)
# _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


# # Step 3: Find bounding box of digit
# coords = cv2.findNonZero(binary)
# x, y, w, h = cv2.boundingRect(coords)
# digit = binary[y : y + h, x : x + w]

# # Step 4: Resize digit to fit in 20x20 box (preserving aspect ratio)
# h, w = digit.shape
# if h > w:
#     new_h = 20
#     new_w = int(w * (20 / h))
# else:
#     new_w = 20
#     new_h = int(h * (20 / w))
# resized_digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

# # Step 5: Pad to make it 28x28 with digit centered
# # First pad to 20x20
# padded_digit = np.zeros((20, 20), dtype=np.uint8)
# start_x = (20 - resized_digit.shape[1]) // 2
# start_y = (20 - resized_digit.shape[0]) // 2
# padded_digit[
#     start_y : start_y + resized_digit.shape[0],
#     start_x : start_x + resized_digit.shape[1],
# ] = resized_digit

# # Now center it into 28x28
# final_image = np.zeros((28, 28), dtype=np.uint8)
# final_image[4:24, 4:24] = padded_digit

# # Optional: Invert back if needed (white digit on black background)
# mnist_like = cv2.bitwise_not(final_image)

# cv2.imwrite("mnist_style_digit.png", final_image)

# # Save/Show the result
# cv2.imwrite("mnist_style_digit.png", final_image)
# plt.imshow(final_image, cmap="gray")
# plt.axis("off")
# plt.title("MNIST-like Digit")
# plt.show()


import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load input image in grayscale
img = cv2.imread("download_(2).jpg", cv2.IMREAD_GRAYSCALE)

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

final_image[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized_digit

# Step 5: Invert the image so the digit is white on black background (like MNIST)
mnist_style = 255 - final_image

# Save and show
cv2.imwrite("mnist_style_digit.png", mnist_style)
plt.imshow(mnist_style, cmap="gray")
plt.title("True MNIST-style Digit")
plt.axis("off")
plt.show()
