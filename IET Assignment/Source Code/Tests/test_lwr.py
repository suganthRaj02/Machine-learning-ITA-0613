import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from locally_weighted_regression import LocallyWeightedRegression


def test_lwr_prediction():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])

    model = LocallyWeightedRegression(tau=1.0)
    model.fit(X, y)

    prediction = model.predict([[3]])

    assert np.isclose(prediction[0], 6, atol=0.5)