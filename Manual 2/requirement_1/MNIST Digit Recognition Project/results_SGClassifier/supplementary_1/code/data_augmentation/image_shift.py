import numpy as np


def shift_image(image, direction):
    """Shifts a flattened 28x28 image in a specified direction.

    Args:
        image (np.ndarray): A flattened NumPy array representing a 28x28 image.
        direction (str): The direction to shift the image.
                            Must be one of 'left', 'right', 'up', or 'down'.

    Returns:
        np.ndarray: A flattened NumPy array representing the shifted image.

    Raises:
        ValueError: If an invalid direction is provided.
    """

    image = image.reshape(28, 28)
    shifted = np.zeros_like(image)

    if direction == "left":
        shifted[:, :-1] = image[:, 1:]
    elif direction == "right":
        shifted[:, 1:] = image[:, :-1]
    elif direction == "up":
        shifted[:-1, :] = image[1:, :]
    elif direction == "down":
        shifted[1:, :] = image[:-1, :]
    else:
        raise ValueError("Invalid direction. Use 'left', 'right', 'up', or 'down'.")

    # That line of code flattens the shifted = np.zeros_like(image), which is currently a 28x28 NumPy array, back into a 1D (Vertical) array of 784 elements. This is done to maintain consistency with the original input format where each image is represented as a flattened array. The flatten() method reshapes the array into a single dimension, preserving all the elements in their original order.
    return shifted.flatten()  # So is 1D with shape (784,)


def augment_dataset(X_train, y_train):
    """Augments a dataset by shifting each image in four directions (left, right, up, down).

    Args:
        X_train (np.ndarray): Training data features, where each row represents a flattened image.
        y_train (np.ndarray): Training data labels.

    Returns:
        tuple: A tuple containing the augmented training data features (X_combined) and labels (y_combined).
                X_combined and y_combined are numpy arrays that include both the original and augmented data.
    """

    augmented_images = []
    augmented_labels = []

    # each X_train is 28×28 pixels
    for image, label in zip(X_train, y_train):
        for direction in ["left", "right", "up", "down"]:
            shifted_img = shift_image(image, direction)
            augmented_images.append(shifted_img)
            augmented_labels.append(label)

    # An increase in training data size from 60,000 to 300,000 (60k original + 4 × 60k augmented)
    X_augmented = np.array(augmented_images)  # X_augmented is 300,000x784 array
    y_augmented = np.array(augmented_labels)

    # Combine original + augmented
    X_combined = np.vstack((X_train, X_augmented))
    y_combined = np.hstack((y_train, y_augmented))

    return X_combined, y_combined
