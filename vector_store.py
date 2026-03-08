import faiss
import numpy as np

class VectorStore:

    def __init__(self, dim):

        self.index = faiss.IndexFlatL2(dim)

        self.texts = []

    def add(self, vectors, texts):

        vec = np.array(vectors).astype("float32")

        self.index.add(vec)

        self.texts.extend(texts)

    def search(self, query_vector, k=5):

        q = np.array([query_vector]).astype("float32")

        dist, idx = self.index.search(q, k)

        results = []

        for i in idx[0]:
            if i < len(self.texts):
                results.append(self.texts[i])

        return results