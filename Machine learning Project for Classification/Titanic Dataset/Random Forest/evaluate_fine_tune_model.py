from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict, cross_val_score


def evaluate_fine_tune_model(best_fine_tune_model, X_train_selected, y_train):
    """Evaluate the model using cross-validation."""

    # cross_val_score() does not use the .fit()ted model that you pass. It:
    # Clones the model internally (clone(model)).
    # Trains it from scratch on each fold of the cross-validation split.

    # Step 10: Evaluate using cross-validation
    scores = cross_val_score(best_fine_tune_model, X_train_selected, y_train, cv=5)
    print("Cross-validation scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

    print(f"Scores: {scores}")

    # Generate cross-validated predictions to get classification report
    y_train_fine_tune_pred = cross_val_predict(
        best_fine_tune_model, X_train_selected, y_train, cv=5
    )

    return y_train_fine_tune_pred
