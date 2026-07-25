import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
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
        if input_features is None:
            return None  # Or raise an error, depending on desired behavior
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
            ("feature_selector", SpecificFeatureSelector()),
            ("regressor", RandomForestRegressor(n_estimators=100, random_state=42)),
        ]
    )

    # Find index of 'ocean_proximity_ISLAND' after one-hot encoding
    preparation_step = full_pipeline.named_steps["preparation"]

    # Fit the preparation step separately to determine feature names
    preparation_step = preparation_step.fit(df)

    # Get feature names after one-hot encoding
    feature_names = preparation_step.get_feature_names_out()
    # # print feature names after one-hot encoding
    print(f"Feature names after OneHotEncoding: {feature_names}")
    island_idx = list(feature_names).index("cat__ocean_proximity_ISLAND")

    # Update the feature selector with the correct index
    full_pipeline.named_steps["feature_selector"].island_idx = island_idx

    return full_pipeline


def selected_features_pipeline_that_predict(strat_train_set, strat_test_set):

    X_train = strat_train_set.drop("median_house_value", axis=1)
    y_train = strat_train_set["median_house_value"]

    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"]

    pipeline = make_full_pipeline(X_train)
    train_model = pipeline.fit(X_train, y_train)

    # Now you can predict directly on raw dataframes:
    y_pred = pipeline.predict(X_test)
    print(f"prediction: {y_pred}")

    return pipeline, train_model, X_train, y_train, X_test, y_test, y_pred
