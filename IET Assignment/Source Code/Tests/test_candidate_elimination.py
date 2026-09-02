import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from candidate_elimination import CandidateElimination


def test_candidate_elimination():
    X = np.array([
        ["Sunny", "Warm"],
        ["Sunny", "Cold"],
        ["Rainy", "Warm"]
    ])

    y = np.array([
        "High",
        "High",
        "Low"
    ])

    domains = [
        ["Sunny", "Rainy"],
        ["Warm", "Cold"]
    ]

    model = CandidateElimination(
        n_features=2,
        domains=domains
    )

    model.fit(X, y)

    S, G = model.get_boundaries()

    assert isinstance(S, list)
    assert isinstance(G, list)