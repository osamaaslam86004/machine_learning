import numpy as np
from sklearn.model_selection import train_test_split

from fetch_load_dataset import load_emails


def test_train_split_data():
    """
    Loads the email dataset, splits it into training and testing sets, and prints the dataset size and spam ratio.

    Returns:
        tuple: A tuple containing the training and testing sets for features (X) and labels (y).
               Specifically: (X_train, X_test, y_train, y_test)
    """
    # Load ham and spam emails
    ham_texts, ham_labels = load_emails("easy_ham", "ham")
    spam_texts, spam_labels = load_emails("spam", "spam")

    # Combine ham and spam emails
    X = ham_texts + spam_texts
    y = ham_labels + spam_labels

    # Print dataset information
    print(f"Total emails: {len(X)} — Spam ratio: {np.mean(y):.2%}")

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test
