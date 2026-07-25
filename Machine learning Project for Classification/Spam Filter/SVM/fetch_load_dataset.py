import email
import os
import tarfile
import urllib.request
from glob import glob

# Constants
DOWNLOAD_ROOT = "https://spamassassin.apache.org/old/publiccorpus/"
FILES = {"20030228_easy_ham.tar.bz2": "easy_ham", "20030228_spam.tar.bz2": "spam"}
DATA_PATH = "datasets/spamassassin"


# Step 1: Download and extract datasets
def download_and_extract():
    os.makedirs(DATA_PATH, exist_ok=True)
    for filename, subdir in FILES.items():
        filepath = os.path.join(DATA_PATH, filename)
        if not os.path.isfile(filepath):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(DOWNLOAD_ROOT + filename, filepath)
            with tarfile.open(filepath) as tar:
                tar.extractall(path=DATA_PATH)
    print("Download and extraction complete.\n")


# Step 2: Load and parse emails
def load_emails(subdir, label):
    texts, labels = [], []
    full_path = os.path.join(DATA_PATH, subdir)
    for path in glob(full_path + "/**", recursive=True):
        if os.path.isfile(path):
            try:
                with open(path, "r", errors="ignore") as f:
                    msg = email.message_from_file(f)
                    body = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            if part.get_content_type() == "text/plain":
                                payload = part.get_payload(decode=True)
                                if payload:
                                    body += payload.decode(errors="ignore")
                    else:
                        payload = msg.get_payload(decode=True)
                        if payload:
                            body = payload.decode(errors="ignore")
                    texts.append(body)
                    labels.append(1 if label == "spam" else 0)
            except Exception as e:
                pass
    return texts, labels
