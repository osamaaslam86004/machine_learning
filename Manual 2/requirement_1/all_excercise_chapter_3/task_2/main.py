import numpy as np
from evaluate_fine_tune_model import evaluate_fine_tune_model
from evaluate_model import evaluate_model
from fine_tune_model import fine_tune_model, fine_tune_model_with_pca
from image_shift import augment_dataset
from load_split_data import load_augmented_pca_applied_data, load_split_data
from sklearn.datasets import fetch_openml
from train_model import train_model, train_model_with_pca


def train_fine_tune_model(X_train, y_train):

    # Multiclass Classification (KNeighborsClassifier) with OvR-strategy
    print("\nTraining KNeighborsClassifier (Multiclass 0-9):")
    model = train_model(X_train, y_train)

    # Evaluate KNeighborsClassifier
    print("\nEvaluating KNeighborsClassifier (Multiclass 0-9):")
    evaluate_model(model, X_test, y_test)

    # Fine Tune KNeighborsClassifier
    print("\nFine-Tune KNeighborsClassifier (Multiclass 0-9):")
    best_fine_tune_model = fine_tune_model(model, X_train, y_train)

    # Evaluate Fine-Tune KNeighborsClassifier
    print("\nEvaluate Fine-Tune KNeighborsClassifier (Multiclass 0-9):")
    evaluate_fine_tune_model(best_fine_tune_model, X_test, y_test)


def train_fine_tune_model_with_pca(X_train, y_train):

    # Multiclass Classification (KNeighborsClassifier) with OvR-strategy
    print("\nTraining KNeighborsClassifier (Multiclass 0-9):")
    model = train_model_with_pca(X_train, y_train)

    # Evaluate KNeighborsClassifier
    print("\nEvaluating KNeighborsClassifier (Multiclass 0-9):")
    evaluate_model(model, X_test, y_test)

    # Fine Tune KNeighborsClassifier
    print("\nFine-Tune KNeighborsClassifier (Multiclass 0-9):")
    best_fine_tune_model = fine_tune_model_with_pca(model, X_train, y_train)

    # Evaluate Fine-Tune KNeighborsClassifier
    print("\nEvaluate Fine-Tune KNeighborsClassifier (Multiclass 0-9):")
    evaluate_fine_tune_model(best_fine_tune_model, X_test, y_test)


if __name__ == "__main__":

    pca_pipeline = False

    if pca_pipeline:

        X_train, X_test, y_train, y_test = load_split_data()

        # Apply image shift data augmentation
        X_train, y_train = augment_dataset(X_train, y_train)

        train_fine_tune_model_with_pca(X_train, y_train)

    else:
        X_train, X_test, y_train, y_test = load_augmented_pca_applied_data()

        train_fine_tune_model(X_train, y_train)
