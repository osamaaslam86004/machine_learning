from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict, cross_val_score


def evaluate_model(model, X_train_prepared, y_train):
    """Evaluate the model using cross-validation."""

    # cross_val_score() does not use the .fit()ted model that you pass. It:
    # Clones the model internally (clone(model)).
    # Trains it from scratch on each fold of the cross-validation split.

    # Step 10: Evaluate using cross-validation
    scores = cross_val_score(model, X_train_prepared, y_train, cv=5)
    print("Cross-validation scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

    print(f"Scores: {scores}")


def custom_cross_validation_predict(model, X_train_prepared, y_train):
    """
    Evaluate the model using cross-validation and return predictions.
    """

    # cross_val_score() does not use the .fit()ted model that you pass. It:
    # Clones the model internally (clone(model)).
    # Trains it from scratch on each fold of the cross-validation split.

    # 2. Cross-validated predictions (for other metrics)
    y_train_pred = cross_val_predict(model, X_train_prepared, y_train, cv=5)

    # 3. Confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_train, y_train_pred))

    return y_train_pred
