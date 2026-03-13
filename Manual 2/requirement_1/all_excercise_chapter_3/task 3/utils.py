import pandas as pd
from sklearn.metrics import classification_report


def create_submission_file_kaggle(predictions, filename_prefix, passenger_ids):
    submission = pd.DataFrame({"PassengerId": passenger_ids, "Survived": predictions})

    submission.to_csv(f"{filename_prefix}_submission.csv", index=False)
    print(f"{filename_prefix}_submission.csv file created.")


def print_classification_report(y_train, y_train_pred):
    # 4. Print classification report
    print("\nClassification Report:")
    print(classification_report(y_train, y_train_pred))


def print_selected_features_name_index(X_train_prepared, selector):

    selected_features_index = selector.get_support(indices=True)

    selected_features_name = X_train_prepared.columns[selector.get_support()]
    selected_features_name = selected_features_name.tolist()

    for index, name in zip(selected_features_index, selected_features_name):
        print(f"Index: {index}, Name: {name}")


def make_predictions_test_set(best_model, X_test_selected):
    """
    Make predictions on test set (determine survival).
    """

    # Final predictions on Kaggle test set (no y_test available)
    y_test_pred = best_model.predict(X_test_selected)

    return y_test_pred


def print_classification_report(y_train, y_train_pred):
    # 4. Print classification report
    print("\nClassification Report:")
    print(classification_report(y_train, y_train_pred))


def print_selected_features_name_index(all_feature_names, selector):
    selected_indices = selector.get_support(indices=True)
    print("\n📊 Selected Important Features:\n")
    for idx in selected_indices:
        print(f"Index: {idx:2d}, Name: {all_feature_names[idx]}")


def make_predictions_test_set(best_model, X_test_selected):
    """
    Make predictions on test set (determine survival).
    """

    # Final predictions on Kaggle test set (no y_test available)
    y_test_pred = best_model.predict(X_test_selected)

    return y_test_pred
