import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Custom transformer to add engineered features
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        add_bedrooms_per_room=True,
        rooms_ix=None,
        bedrooms_ix=None,
        population_ix=None,
        households_ix=None,
    ):
        self.add_bedrooms_per_room = add_bedrooms_per_room
        self.rooms_ix = rooms_ix
        self.bedrooms_ix = bedrooms_ix
        self.population_ix = population_ix
        self.households_ix = households_ix

    def fit(self, X, y=None):
        # Just check indices are set
        if None in [
            self.rooms_ix,
            self.bedrooms_ix,
            self.population_ix,
            self.households_ix,
        ]:
            raise ValueError(
                "Column indices must be provided to CombinedAttributesAdder"
            )
        return self

    def transform(self, X):
        rooms_per_household = X[:, self.rooms_ix] / X[:, self.households_ix]
        population_per_household = X[:, self.population_ix] / X[:, self.households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, self.bedrooms_ix] / X[:, self.rooms_ix]
            return np.c_[
                X,
                rooms_per_household,
                population_per_household,
                bedrooms_per_room,
            ]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

    def get_feature_names_out(self, input_features=None):
        # Fallback to generic names if input_features is None
        if input_features is None:
            input_features = [
                # or whatever size your input is
                f"x{i}"
                for i in range(8)
            ]
        new_feature_names = [
            "rooms_per_household",
            "population_per_household",
        ]
        if self.add_bedrooms_per_room:
            new_feature_names.append("bedrooms_per_room")
        return np.concatenate([input_features, new_feature_names])


# Custom transformer to select the 3 features by index
class SpecificFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, island_idx=None):  # Changed inland_idx to island_idx
        self.island_idx = island_idx  # Updated attribute name

    def fit(self, X, y=None):
        if self.island_idx is None:  # Updated attribute name
            raise ValueError(
                "island_idx must be provided to SpecificFeatureSelector"
            )  # Updated message
        return self

    def transform(self, X):
        # Select by provided indices: median_income, population_per_household, ISLAND
        return np.c_[
            X[:, [7, 9]], X[:, [self.island_idx]]
        ]  # Changed to self.island_idx and index 7 for median_income and 9 for population_per_household.


def make_full_pipeline(df):

    rooms_ix = df.columns.get_loc("total_rooms")
    bedrooms_ix = df.columns.get_loc("total_bedrooms")
    population_ix = df.columns.get_loc("population")
    households_ix = df.columns.get_loc("households")

    num_attribs = list(df.drop("ocean_proximity", axis=1))
    cat_attribs = ["ocean_proximity"]

    attribs_adder = CombinedAttributesAdder(
        rooms_ix=rooms_ix,
        bedrooms_ix=bedrooms_ix,
        population_ix=population_ix,
        households_ix=households_ix,
    )
    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("attribs_adder", attribs_adder),
            ("std_scaler", StandardScaler()),
        ]
    )

    # Temporary preparation pipeline to get 'ISLAND' index
    preparation_temp = ColumnTransformer(
        [
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_attribs),
        ]
    )
    preparation_temp.fit(df)  # Only for feature name inspection

    feature_names = preparation_temp.get_feature_names_out()
    island_idx = list(feature_names).index("cat__ocean_proximity_ISLAND")

    # Full pipeline with feature selection and estimator
    full_pipeline = Pipeline(
        [
            (
                "preparation",
                ColumnTransformer(
                    [
                        ("num", num_pipeline, num_attribs),
                        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_attribs),
                    ]
                ),
            ),
            ("feature_selector", SpecificFeatureSelector(island_idx=island_idx)),
            ("regressor", RandomForestRegressor(n_estimators=100, random_state=42)),
        ]
    )

    return full_pipeline


def get_automatic_selected_features(best_model, X_train):
    # --- Feature Importance BEFORE Fine-Tuning ---
    prep_step = best_model.named_steps["preparation"]
    selector_step = best_model.named_steps["feature_selector"]
    regressor = best_model.named_steps["regressor"]

    prep_step.fit(X_train)  # Ensure fitting before getting names
    prep_features = prep_step.get_feature_names_out()

    # Get the indices of selected features after the selector step
    selected_feature_indices = selector_step.transform(
        np.arange(len(prep_features)).reshape(1, -1)
    )[0].astype(int)

    final_feature_names = [prep_features[i] for i in selected_feature_indices]
    feature_importances = regressor.feature_importances_

    return final_feature_names, feature_importances


def automatic_feature_selection_pipeline(pipeline, X_train, y_train):

    param_grid = {
        "preparation__num__imputer__strategy": ["median", "mean"],
        "preparation__num__attribs_adder__add_bedrooms_per_room": [True, False],
        "regressor__n_estimators": [50, 100, 150],
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring="neg_mean_squared_error",
        return_train_score=True,
        verbose=2,
    )

    grid_search = grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print("Best Parameters for Training Set:", grid_search.best_params_)

    final_feature_names, feature_importances = get_automatic_selected_features(
        best_model, X_train
    )

    print("\nTop Feature Importances BEFORE Fine-Tuning:")
    for name, score in zip(final_feature_names, feature_importances):
        print(f"{name}: {score:.4f}")

    return best_model, grid_search, X_train, y_train
