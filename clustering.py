import numpy as np
import skfuzzy as fuzz

class FuzzyCluster:

    def __init__(self, clusters=10):

        self.clusters = clusters

        self.centers = None

    def fit(self, embeddings):

        data = np.array(embeddings).T

        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            data,
            self.clusters,
            2,
            error=0.005,
            maxiter=1000
        )

        self.centers = cntr

        self.membership = u

    def predict(self, vector):

        data = np.array([vector]).T

        u, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
            data,
            self.centers,
            2,
            error=0.005,
            maxiter=1000
        )

        return u[:,0]