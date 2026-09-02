import numpy as np


class CandidateElimination:

    def __init__(self, n_features, domains):
        self.n_features = n_features
        self.domains = domains
        self.S = ["Ø"] * n_features
        self.G = [["?"] * n_features]

    def covers(self, hypothesis, example):
        for h, x in zip(hypothesis, example):
            if h != "?" and h != x:
                return False
        return True

    def more_general(self, h1, h2):
        for a, b in zip(h1, h2):
            if a != "?" and a != b:
                return False
        return True

    def fit(self, X, y):

        X = np.asarray(X, dtype=str)
        y = np.asarray(y, dtype=str)

        for x, label in zip(X, y):

            if label == "High":

                # First positive example
                if self.S[0] == "Ø":
                    self.S = list(x)
                else:
                    # Generalize S
                    for i in range(self.n_features):
                        if self.S[i] != x[i]:
                            self.S[i] = "?"

                # Remove G hypotheses that do not cover positive example
                self.G = [
                    g for g in self.G
                    if self.covers(g, x)
                ]

            elif label == "Low":

                new_G = []

                for g in self.G:

                    if self.covers(g, x):

                        # Specialize G
                        for i in range(self.n_features):

                            if g[i] == "?":

                                for value in self.domains[i]:

                                    if value != x[i]:

                                        new_g = list(g)
                                        new_g[i] = value

                                        # Must remain at least as general as S
                                        valid = True

                                        if self.S[i] != "Ø":
                                            if self.S[i] != "?" and value != self.S[i]:
                                                valid = False

                                        if valid and new_g not in new_G:
                                            new_G.append(new_g)

                    else:
                        new_G.append(g)

                self.G = new_G

        return self

    def get_boundaries(self):
        return self.S, self.G


def create_risk_band(y):

    low = np.percentile(y, 33)
    high = np.percentile(y, 67)

    return np.where(
        y <= low,
        "Low",
        np.where(y >= high, "High", "Medium")
    )