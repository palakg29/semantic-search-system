import os
import tarfile
import requests

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/20newsgroups-mld/20_newsgroups.tar.gz"

DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "20_newsgroups.tar.gz")
EXTRACTED = os.path.join(DATA_DIR, "20_newsgroups")


def download_dataset():

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    if not os.path.exists(DATA_FILE):

        print("Downloading dataset...")

        r = requests.get(DATA_URL, stream=True)

        with open(DATA_FILE, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

    if not os.path.exists(EXTRACTED):

        print("Extracting dataset...")

        with tarfile.open(DATA_FILE, "r:gz") as tar:
            tar.extractall(DATA_DIR)


def load_texts():

    if not os.path.exists(EXTRACTED):
        download_dataset()

    texts = []

    for root, dirs, files in os.walk(EXTRACTED):
        for file in files:
            path = os.path.join(root, file)

            try:
                with open(path, "r", encoding="latin1") as f:
                    txt = f.read()
                    texts.append(txt)
            except:
                continue

    return texts