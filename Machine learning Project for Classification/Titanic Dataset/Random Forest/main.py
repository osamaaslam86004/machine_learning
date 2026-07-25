from custom_pipeleine import full_pipeline
from data_augmentation import extract_Mr_Miss_Mrs_from_name
from evaluate_fine_tune_model import evaluate_fine_tune_model
from evaluate_model import custom_cross_validation_predict, evaluate_model
from fine_tune_model import fine_tune_model, select_important_features
from load_split_data import fetch_titanic_data, load_titanic_data
from train_model import train_model
from utils import (
    create_submission_file_kaggle,
    make_predictions_test_set,
    print_classification_report,
    print_selected_features_name_index,
)

if __name__ == "__main__":

    # Download the data
    fetch_titanic_data()

    # Load training data
    train_data = load_titanic_data("train.csv")

    # Load test data
    test_data = load_titanic_data("test.csv")

    #  Extract the Title (e.g., "Mr", "Mrs", "Miss") and then Normalize rare Titles
    extract_Mr_Miss_Mrs_from_name(train_data, test_data)

    # Transform data using pipeline
    X_train_prepared, X_test_prepared, y_train, all_feature_names = full_pipeline(
        train_data, test_data
    )

    # train the model
    best_model = train_model(X_train_prepared, y_train)

    # Cross validation
    evaluate_model(best_model, X_train_prepared, y_train)

    # Make predictions on the test set
    y_train_pred = custom_cross_validation_predict(
        best_model, X_train_prepared, y_train
    )
    # Print classification report
    print_classification_report(y_train, y_train_pred)

    # predictions on Test set after Training model
    y_test_pred = make_predictions_test_set(best_model, X_test_prepared)

    # Create submission file
    #
    # Use original test_data to get PassengerId
    create_submission_file_kaggle(y_test_pred, "train", test_data["PassengerId"])

    # Feature Selection with model own's feature importance
    X_train_selected, selector = select_important_features(X_train_prepared, y_train)

    # print selected Features
    print_selected_features_name_index(all_feature_names, selector)

    # fine tune the model
    best_fine_tune_model = fine_tune_model(X_train_selected, y_train)

    # evaluate fine tune the model
    y_train_fine_tune_pred = evaluate_fine_tune_model(
        best_fine_tune_model, X_train_selected, y_train
    )

    # y_train and y_train_fine_tune_pred are your true and predicted labels, respectively
    print_classification_report(y_train, y_train_fine_tune_pred)

    # predictions on Test set
    X_test_selected = selector.transform(X_test_prepared)
    y_test_fine_tune_pred = make_predictions_test_set(
        best_fine_tune_model, X_test_selected
    )

    # Create submission file
    create_submission_file_kaggle(
        y_test_fine_tune_pred, "test", test_data["PassengerId"]
    )
