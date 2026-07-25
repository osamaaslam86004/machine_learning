common_titles_to_normalize = [
    "Lady",
    "Capt",
    "Col",
    "Don",
    "Dr",
    "Major",
    "Rev",
    "Sir",
    "Jonkheer",
    "Ms",
    "Mme",
]


def extract_Mr_Miss_Mrs_from_name(train_data, test_data):
    # Add 'Dona' to test set normalization (if it only exists in test set)
    rare_titles_test_set = common_titles_to_normalize + ["Dona"]

    # Add 'Dona' to test set normalization (if it only exists in test set)
    rare_titles_train_set = common_titles_to_normalize + ["Mlle", "the Countess"]

    for index, dataset in enumerate([train_data, test_data]):
        dataset["Title"] = dataset["Name"].str.extract(r",\s*([^\.]+)\.", expand=False)

        if index == 0:
            # Normalize Rare Titles for train data
            dataset["Title"] = dataset["Title"].replace(rare_titles_train_set, "Rare")

        else:
            # Normalize Rare Titles for test data
            #
            # Test data conatins 1 additonal rare tiltle 'Dona'
            dataset["Title"] = dataset["Title"].replace(rare_titles_test_set, "Rare")

    print("Train data title counts:")
    print(train_data["Title"].value_counts())

    print("\nTest data title counts:")
    print(test_data["Title"].value_counts())
