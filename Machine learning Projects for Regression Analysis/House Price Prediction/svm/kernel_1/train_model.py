# Required for training models
import joblib
from sklearn.svm import SVR


def print_trained_model_coefficients(full_pipeline, final_model):
    """
    Analyzes the best model's errors, printing feature importances
    or coefficients for linear SVR.
    """
    if hasattr(final_model, "coef_"):
        print("\nCoefficients of the train SVR model:")
        #  Note:  SVR with kernel='linear' has coef_, but not other kernels.
        #   Adapt for non-linear kernels if needed.
        feature_names = full_pipeline.get_feature_names_out()  # Get feature names
        for feature, coef in zip(
            feature_names, final_model.coef_[0]
        ):  # Assuming single output
            print(f"{feature}: {coef:.4f}")
    else:
        print(f"Coefficients not available for SVR with kernel: {final_model.kernel}")


def save_model(model):
    # Save the model to a file
    joblib.dump(model, "SVR_model.pkl")


def train_model(housing_prepared, housing_labels, full_pipeline):
    """
    Train a Model (Random Forest Regressor or SVR)
    """

    model = SVR(kernel="linear", C=180000)  # Default, can be tuned

    # Fit the model to the training data
    model.fit(housing_prepared, housing_labels)

    # save the model
    save_model(model)

    # print trained model coefficients
    print_trained_model_coefficients(full_pipeline, model)

    return model
