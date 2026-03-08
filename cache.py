from sklearn.metrics.pairwise import cosine_similarity

class SemanticCache:

    def __init__(self, threshold=0.85):

        self.threshold = threshold

        self.entries = []

        self.hit = 0

        self.miss = 0

    def lookup(self, emb):

        for e in self.entries:

            sim = cosine_similarity([emb],[e["embedding"]])[0][0]

            if sim >= self.threshold:

                self.hit += 1

                return e, sim

        self.miss += 1

        return None, None

    def add(self, query, emb, result, cluster):

        self.entries.append({
            "query": query,
            "embedding": emb,
            "result": result,
            "cluster": cluster
        })

    def stats(self):

        total = self.hit + self.miss

        rate = self.hit/total if total else 0

        return {
            "total_entries": len(self.entries),
            "hit_count": self.hit,
            "miss_count": self.miss,
            "hit_rate": rate
        }

    def clear(self):

        self.entries = []

        self.hit = 0

        self.miss = 0