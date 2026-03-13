# The stop_words=stopwords.words("english") in CountVectorizer and stemmer = PorterStemmer() in EmailPreprocessor serve distinct purposes in your text processing pipeline.

# Stop Word Removal: stop_words=stopwords.words("english") in CountVectorizer removes common English words (like "the," "a," "is") that usually don't carry much meaning for text classification. This helps reduce the dimensionality of your data and potentially improves model performance by focusing on more informative words.

# Stemming: stemmer = PorterStemmer() in EmailPreprocessor, when enabled, reduces words to their root or base form (stem). For example, "running," "runs," and "ran" would all be stemmed to "run." This helps group related words together, potentially reducing feature space and improving generalization.

# You are not stemming twice. Stemming is performed within the EmailPreprocessor's transform method before the text is passed to the CountVectorizer. The CountVectorizer then operates on the stemmed text to create the binary vectors. The order of operations is pre-processing (including stemming if activated) followed by vectorization.


import nltk

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction.text import CountVectorizer
nltk.download("stopwords")
import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from urlextract import URLExtract


class EmailPreprocessor(BaseEstimator, TransformerMixin):
    """
    Custom transformer for preprocessing email text.

    This transformer applies a series of text cleaning steps, including
    header removal, lowercasing, URL and number replacement, punctuation
    removal, and stemming (optional).  It uses the urlextract library
    to identify URLs of any form.

    Args:
        strip_headers (bool): Whether to remove email headers (default: True).
        lowercase (bool): Whether to convert text to lowercase (default: True).
        remove_punctuation (bool): Whether to remove punctuation (default: True).
        replace_urls (str or None): String to replace URLs with, or None to keep (default: "URL").
        replace_numbers (str or None): String to replace numbers with, or None to keep (default: "NUMBER").
        stemming (bool): Whether to perform stemming using Porter stemmer (default: False).
    """

    def __init__(
        self,
        strip_headers=True,
        lowercase=True,
        remove_punctuation=True,
        replace_urls="URL",
        replace_numbers="NUMBER",
        stemming=False,
    ):
        self.strip_headers = strip_headers
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.replace_urls = replace_urls
        self.replace_numbers = replace_numbers
        self.stemming = stemming
        self.url_extractor = URLExtract()

    def fit(self, X, y=None):
        """
        This transformer doesn't need to learn anything from the training data,
        so it simply returns self.
        """
        return self

    def transform(self, X):
        """
        Applies the preprocessing steps to a list of emails.

        Args:
            X (list): A list of email strings.

        Returns:
            list: A list of preprocessed email strings.
        """
        X_transformed = []
        for email in X:
            text = email
            if self.strip_headers:
                text = text.split("\n\n", 1)[1] if "\n\n" in text else text
            if self.lowercase:
                text = text.lower()
            if self.replace_urls:
                urls = list(self.url_extractor.find_urls(text))
                urls.sort(key=lambda url: len(url), reverse=True)
                for url in urls:
                    text = text.replace(url, self.replace_urls)
            if self.replace_numbers:
                text = re.sub(r"\d+", self.replace_numbers, text)
            if self.remove_punctuation:
                text = re.sub(r"[^\w\s]", "", text)
            X_transformed.append(text)

        if self.stemming:
            stemmer = PorterStemmer()
            X_transformed = [
                " ".join([stemmer.stem(word) for word in text.split()])
                for text in X_transformed
            ]

        return X_transformed


def pre_processing_pipeline(X_train, X_test):
    """
    Creates a scikit-learn pipeline for preprocessing email text and
    vectorizing it into binary word presence/absence vectors.

    Returns:
        Pipeline: A scikit-learn Pipeline object containing an EmailPreprocessor
                  and a CountVectorizer.
    """

    # Create a preprocessing pipeline
    preprocess_pipeline = Pipeline(
        [
            ("email_preprocessor", EmailPreprocessor(stemming=True)),
            (
                "vectorizer",
                CountVectorizer(
                    # stop_words=stopwords.words("english"),
                    lowercase=False,  # Already handled
                    max_df=0.95,
                    min_df=5,
                    binary=True,
                ),
            ),
        ]
    )

    X_train_transformed = preprocess_pipeline.fit_transform(X_train)
    X_test_transformed = preprocess_pipeline.transform(X_test)

    return X_train_transformed, X_test_transformed


# def stemming_and_vectorize_text(X_train, X_test):
#     """
#     Vectorizes text data using binary word presence/absence indicators,
#     applying stemming and removing stop words.

#     Args:
#         X_train (list): Training text data.
#         X_test (list): Testing text data.

#     Returns:
#         tuple: A tuple containing the transformed training and testing data.
#                Specifically: (X_train_vec, X_test_vec)
#     """
#     # Initialize CountVectorizer with binary=True
#     vectorizer = CountVectorizer(  # Changed this
#         stop_words=stopwords.words("english"),
#         lowercase=True,
#         max_df=0.95,
#         min_df=5,
#         binary=True,  # Added this
#     )
#     # Fit and transform the training data
#     X_train_vec = vectorizer.fit_transform(X_train)
#     # Transform the testing data
#     X_test_vec = vectorizer.transform(X_test)

#     return X_train_vec, X_test_vec


# def stemming_and_vectorize_text(X_train, X_test):
#     """
#     Vectorizes text data using TF-IDF, applying stemming and removing stop words.

#     Args:
#         X_train (list): Training text data.
#         X_test (list): Testing text data.

#     Returns:
#         tuple: A tuple containing the TF-IDF transformed training and testing data.
#                Specifically: (X_train_tfidf, X_test_tfidf)
#     """
#     # Initialize TF-IDF vectorizer
#     tfidf = TfidfVectorizer(
#         stop_words=stopwords.words("english"), lowercase=True, max_df=0.95, min_df=5
#     )
#     # Fit and transform the training data
#     X_train_tfidf = tfidf.fit_transform(X_train)
#     # Transform the testing data
#     X_test_tfidf = tfidf.transform(X_test)

#     return X_train_tfidf, X_test_tfidf
