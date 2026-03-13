import os

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt

from data_preparation import automatic_feature_selection_pipeline, make_full_pipeline
from evaluate_model import display_scores, evaluate_fine_tuned_model, evaluate_model
from fine_tune_model import fine_tune_model
from get_data import fetch_housing_data, load_housing_data
from strated_dataset import create_stratified_test_set

# Set Matplotlib defaults
mpl.rc("axes", labelsize=14)
mpl.rc("xtick", labelsize=12)
mpl.rc("ytick", labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)


# Function to save figures
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def main():
    # Fetch the data
    fetch_housing_data(HOUSING_URL, HOUSING_PATH)

    # Load the data
    housing = load_housing_data(HOUSING_PATH)

    # Create a stratified test set
    strat_train_set, strat_test_set = create_stratified_test_set(housing)

    # Create a pipeline
    X_train = strat_train_set.drop("median_house_value", axis=1)
    y_train = strat_train_set["median_house_value"]

    full_pipeline = make_full_pipeline(X_train)

    train_model, grid_search, X_train, y_train = automatic_feature_selection_pipeline(
        full_pipeline, X_train, y_train
    )

    # Evaluate model
    forest_rmse_scores = evaluate_model(train_model, X_train, y_train)
    print(f"Model rmese scores {forest_rmse_scores}")

    # Display the cross-validation scores
    print("Random Forest Cross-Validation Scores:")
    display_scores(forest_rmse_scores)

    # Fine-Tune Model Using Randomized Search**
    fine_tuned_model, grid_search = fine_tune_model(full_pipeline, X_train, y_train)
    # Print the best hyperparameters found
    print("\nBest hyperparameters found by GridSearch Search:")
    print(grid_search.best_params_)

    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"]

    # Evaluate on the test set
    t_interval = evaluate_fine_tuned_model(fine_tuned_model, X_test, y_test)
    print("95% Confidence Interval for Test RMSE (t-dist):", t_interval)


if __name__ == "__main__":
    main()
