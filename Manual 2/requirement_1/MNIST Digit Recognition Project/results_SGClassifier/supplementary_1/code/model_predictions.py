import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from plot_performance_visualize_errors.plot_model_performance import (
    plot_confusion_matrix,
)


def make_predictions_analyze_fine_tuned_model_on_test_data(
    fine_tune_model, X_test_orignal, y_test_orignal, model="sgd", type="predict"
):
    """
    Makes predictions on the test set, evaluates the model on test set, and prints relevant metrics.

    Args:
        fine_tune_model: The fine-tune model.
        X_test_orignal: The test data.
        y_test_orignal: The true labels for the test data.
        type: string

    Returns:
        None: The function does not return a value but plots a confusion matrix.
    """

    # Make predictions on the scaled test set
    y_test_pred = fine_tune_model.predict(X_test_orignal)

    # Evaluate the model
    accuracy = accuracy_score(y_test_orignal, y_test_pred)
    conf_matrix = confusion_matrix(y_test_orignal, y_test_pred)
    class_report = classification_report(y_test_orignal, y_test_pred)

    print("Performance Metrics on Test Set.......:")
    print(f"  Accuracy: {accuracy:.4f}")
    print("  Confusion Matrix:\n", conf_matrix)
    print("  Classification Report:\n", class_report)

    # Plot Confusion matrix
    classes = np.arange(10)  # Define classes here
    plot_confusion_matrix(y_test_orignal, y_test_pred, classes, model, type)
