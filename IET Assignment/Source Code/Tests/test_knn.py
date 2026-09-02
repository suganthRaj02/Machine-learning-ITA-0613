import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from knn_regressor import KNNRegressor


def test_knn_euclidean():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 10, 20, 30])

    model = KNNRegressor(k=1, metric="euclidean")
    model.fit(X, y)

    prediction = model.predict([[2]])

    assert prediction[0] == 20


def test_knn_mahalanobis():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 10, 20, 30])

    model = KNNRegressor(k=1, metric="mahalanobis")
    model.fit(X, y)

    prediction = model.predict([[3]])

    assert prediction[0] == 30