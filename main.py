from fastapi import FastAPI
from pydantic import BaseModel

import numpy as np

from app.dataset_loader import load_texts
from app.embeddings import get_embedding
from app.vector_store import VectorStore
from app.clustering import FuzzyCluster
from app.cache import SemanticCache


app = FastAPI()

texts = load_texts()[:2000]

embeddings = [get_embedding(t) for t in texts]

dim = len(embeddings[0])

vector_db = VectorStore(dim)

vector_db.add(embeddings, texts)

cluster_model = FuzzyCluster(10)

cluster_model.fit(embeddings)

cache = SemanticCache(0.85)


class Query(BaseModel):
    query: str


@app.post("/query")
def query(q: Query):

    emb = get_embedding(q.query)

    cached, sim = cache.lookup(emb)

    if cached:

        return {
            "query": q.query,
            "cache_hit": True,
            "matched_query": cached["query"],
            "similarity_score": float(sim),
            "result": cached["result"],
            "dominant_cluster": int(cached["cluster"])
        }

    results = vector_db.search(emb,1)

    result = results[0] if results else ""

    cluster_dist = cluster_model.predict(emb)

    dominant_cluster = int(np.argmax(cluster_dist))

    cache.add(q.query, emb, result, dominant_cluster)

    return {
        "query": q.query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": None,
        "result": result,
        "dominant_cluster": dominant_cluster
    }


@app.get("/cache/stats")
def stats():

    return cache.stats()


@app.delete("/cache")
def clear():

    cache.clear()

    return {"message":"cache cleared"}