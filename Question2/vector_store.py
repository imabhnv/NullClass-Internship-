import os
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_FILE = r"data.txt"

def load_data():
    if not os.path.exists(DATA_FILE):
        return ""
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return f.read()

def load_vector_store():
    if not os.path.exists(DATA_FILE):
        return None, None, None

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        corpus = [line.strip() for line in f if line.strip()]
    
    if not corpus:
        return None, None, None

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(corpus)
    return vectorizer, vectors, corpus

def update_vector_store():
    return load_vector_store()
