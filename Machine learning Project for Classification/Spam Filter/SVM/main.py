from evaluate_model import evaluate_model
from fetch_load_dataset import download_and_extract
from pre_processing_pipeline import pre_processing_pipeline
from split_data import test_train_split_data
from train_model import train_model

# Run the training pipeline
if __name__ == "__main__":

    # Step 1: Download and extract datasets
    print("Downloading and extracting datasets...")
    download_and_extract()

    # # Step 2: Train-test split
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = test_train_split_data()

    # Step 3: Text vectorization
    print("pre-processing pipeline and Vectorizing text data...")
    X_train_transformed, X_test_transformed = pre_processing_pipeline(X_train, X_test)

    # Step 4: Train classifier
    print("Training classifier...")
    clf = train_model(X_train_transformed, y_train)

    # Step 5: Evaluate
    print("Evaluating classifier...")
    y_test_pred = evaluate_model(clf, X_test_transformed, y_test)
