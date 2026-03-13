from load_split_data import (
    count_occurrences_of_each_digit_in_training_set,
    load_split_data,
)
from model_predictions import make_predictions_analyze_fine_tuned_model_on_test_data
from utils.random_forest import (
    random_forest_fine_tune_evaluate_visualize_errors,
    random_forest_train_evaluate_visulaize_errors,
)
from utils.sgc import (
    fine_tune_evaluate_visualize_errors,
    train_evaluate_visulaize_errors,
)

if __name__ == "__main__":

    # Load Original MNIST Dataset
    print("\n Loading Original Dtataset...")
    X_train_orignal, X_test_orignal, y_train_orignal, y_test_orignal = load_split_data()

    # Count the occurrences of each digit in the training set
    count_occurrences_of_each_digit_in_training_set(y_train_orignal)

    select_model = "random_forest"

    if select_model == "sg_classifier":

        # Train, evaluate, and Visualize errors
        train_evaluate_visulaize_errors(X_train_orignal, y_train_orignal)

        # Fine-tune, evaluate, and Visualize errors
        fine_tune_model, X_test_aug_pca, y_test_aug_pca = (
            fine_tune_evaluate_visualize_errors()
        )
    else:
        # Train, evaluate, and Visualize errors
        random_forest_train_evaluate_visulaize_errors(X_train_orignal, y_train_orignal)

        # Fine-tune, evaluate, and Visualize errors
        fine_tune_model, X_test_aug_pca, y_test_aug_pca = (
            random_forest_fine_tune_evaluate_visualize_errors()
        )

    # Count the occurrences of each digit in the training set
    print("count occurrences of each digit in test set")
    count_occurrences_of_each_digit_in_training_set(y_test_orignal)

    # Make Predictions and Evaluate
    print("Making Predictions and Evaluate")
    make_predictions_analyze_fine_tuned_model_on_test_data(
        fine_tune_model,
        X_test_aug_pca,
        y_test_aug_pca,
        model=select_model,
        type="predict",
    )
