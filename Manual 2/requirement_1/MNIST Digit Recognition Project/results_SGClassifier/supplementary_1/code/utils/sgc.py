from load_split_data import (
    load_augmented_data,
    load_pca_applied_augmented_data,
    load_pca_applied_data,
)
from plot_performance_visualize_errors.sgd_visualize_errors import plot_worst_errors
from train_evaluate.OvR_sgc_multiclass import (
    evaluate_sg_classifier_one_vs_one_strategy,
    train_sg_classifier_OvR_strategy,
)


def train_evaluate_visulaize_errors(X_train_orignal, y_train_orignal):
    """Trains and evaluates an SGDClassifier with OvR strategy on PCA-reduced MNIST data,
    visualizing the worst misclassifications.

    This function performs the following steps:
    1. Loads PCA-reduced (95% variance retained) MNIST data (60K, 154) from a specified Google Drive file.
    2. Trains an SGDClassifier with a One-vs-Rest (OvR) strategy using GridSearchCV for
       hyperparameter tuning.
    3. Evaluates the trained classifier on the training data using cross-validation,
       printing accuracy, precision, recall, F1 score, and the confusion matrix.
    4. Visualizes the worst misclassifications by plotting the original images corresponding
       to the instances with the highest decision function scores among the incorrect predictions.

    Args:
        X_train_orignal (np.ndarray): The original (unprocessed) training data.
        y_train_orignal (np.ndarray): The original training labels.

    Returns:
        None: The function does not return a value but prints evaluation metrics and saves
              plots of the worst misclassifications.
    """

    # Google Drive file ID from the shareable link
    file_id = "1jLwwYnD6CTPTGdu1Bub_04Co5ISO3dQH"
    output_path = "/content/mnist_pca_0.95_no_augmented_data.npz"

    X_train, X_test, y_train, y_test = load_pca_applied_data(file_id, output_path)

    print("\n Training SGDClassifier (Multiclass 0-9) with OvR-strategy....:")
    sdf_clf, X_train_scaled, decision_scores = train_sg_classifier_OvR_strategy(
        X_train, y_train, type="train"
    )

    print("\n Evaluating SGDClassifier......:")
    y_train_pred = evaluate_sg_classifier_one_vs_one_strategy(
        sdf_clf, X_train_scaled, y_train, type="train"
    )

    print("\n Creating plots for worst misclassifications after Training-Tunning...:")
    plot_worst_errors(
        X_train_orignal,
        y_train_orignal,
        y_train_pred,
        decision_scores,
        n_images=10,
        model="sgc",
        type="training",
    )


# def fine_tune_evaluate_visualize_errors(X_train_orignal, y_train_orignal):
def fine_tune_evaluate_visualize_errors():
    """Fine-tune and evaluates an SGDClassifier with OvR strategy on PCA-reduced Augmented MNIST data,
    visualizing the worst misclassifications.

    This function performs the following steps:
    1. Loads PCA-reduced ((95% variance retained) Augmented MNIST data (300K, 154) from a specified Google Drive file.
    2. Fine-tune an SGDClassifier with a One-vs-Rest (OvR) strategy using GridSearchCV for
       hyperparameter tuning.
    3. Evaluates the fine-tune classifier on the training data using cross-validation,
       printing accuracy, precision, recall, F1 score, and the confusion matrix.
    4. Visualizes the worst misclassifications by plotting the original images corresponding
       to the instances with the highest decision function scores among the incorrect predictions.

    Args:
        X_train_orignal (np.ndarray): The original (unprocessed) training data.
        y_train_orignal (np.ndarray): The original training labels.

    Returns:
        fine_tune_model: The fine-tuned SGDClassifier model
    """

    # Google Drive file ID from the shareable link
    file_id = "1-NC0Z9YwgHX90edHl0JNmm-Id7swba1E"
    output_path = "/content/augmented_pca_mnist_data.npz"

    # Loading PCA-reduced Augmented MNIST data
    X_train_aug_pca, X_test_aug_pca, y_train_aug, y_test_aug_pca = (
        load_pca_applied_augmented_data(file_id, output_path)
    )

    # Fine-tune the model
    print("\n Fine-tune SGDClassifier:")
    fine_tune_model, X_train_pca_scaled, decision_scores = (
        train_sg_classifier_OvR_strategy(X_train_aug_pca, y_train_aug, type="fine-tune")
    )

    # Evaluate the fine-tuned model
    print("\n Evaluating Fine-tuned SGDClassifier:")
    y_train_pca_pred = evaluate_sg_classifier_one_vs_one_strategy(
        fine_tune_model, X_train_pca_scaled, y_train_aug, type="fine-tune"
    )

    # Google Drive file ID from the shareable link
    file_id = "1-JtzYPd8l-kPeELqUgKEU33WPUQK_tg6"
    output_path = "/content/mnist_augmented_data.npz"

    # Loading Augmented MNIST data to get X_train_aug, and y_train_aug to
    # plot worst classifications
    X_train_aug, X_test_orignal, y_train_aug, y_test_orignal = load_augmented_data(
        file_id, output_path
    )

    # plot top 10 worst classifications
    print("\n Creating plots for worst misclassifications after Fine-Tunning...:")
    plot_worst_errors(
        X_train_aug,
        y_train_aug,
        y_train_pca_pred,
        decision_scores,
        n_images=10,
        model="sgc",
        type="fine-tune",
    )

    return fine_tune_model, X_test_aug_pca, y_test_aug_pca
