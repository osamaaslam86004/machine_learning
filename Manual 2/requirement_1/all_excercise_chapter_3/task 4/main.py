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






Splitting data into train and test sets...
Total emails: 3002 — Spam ratio: 16.69%
pre-processing pipeline and Vectorizing text data...

Training classifier...
Best parameters: {'alpha': 0.0001, 'loss': 'log_loss', 'penalty': 'elasticnet'}

Best cross-validation score: 0.9575121275121274

Evaluating classifier...
Accuracy: 0.9484193011647255

Classification Report:
              precision    recall  f1-score   support

         ham       0.98      0.96      0.97       501
        spam       0.81      0.90      0.85       100

    accuracy                           0.95       601
   macro avg       0.90      0.93      0.91       601
weighted avg       0.95      0.95      0.95       601


AUC: 0.973193612774451



============================================================
without stop_words = in Countvector is now comment out
============================================================


Splitting data into train and test sets...
Total emails: 3002 — Spam ratio: 16.69%
pre-processing pipeline and Vectorizing text data...

Training classifier...
Best parameters: {'alpha': 0.0001, 'loss': 'hinge', 'penalty': 'elasticnet'}

Best cross-validation score: 0.9729261954261954

Evaluating classifier...
Accuracy: 0.9667221297836939

Classification Report:
              precision    recall  f1-score   support

         ham       0.99      0.97      0.98       501
        spam       0.86      0.95      0.90       100

    accuracy                           0.97       601
   macro avg       0.93      0.96      0.94       601
weighted avg       0.97      0.97      0.97       601


AUC: 0.9855988023952097
