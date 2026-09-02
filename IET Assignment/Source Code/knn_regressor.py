import numpy as np


class KNNRegressor:

    def __init__(self, k=5, metric="euclidean"):
        self.k = k
        self.metric = metric

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y, dtype=float)

        if self.metric == "mahalanobis":
            cov = np.atleast_2d(
                np.cov(self.X_train, rowvar=False)
            )

            cov += np.eye(cov.shape[0]) * 1e-6

            self.inv_cov = np.linalg.pinv(cov)

        return self

    def _distance(self, x):

        diff = self.X_train - x

        if self.metric == "euclidean":
            return np.sqrt(
                np.sum(diff ** 2, axis=1)
            )

        if self.metric == "mahalanobis":
            return np.sqrt(
                np.einsum(
                    "ij,jk,ik->i",
                    diff,
                    self.inv_cov,
                    diff
                )
            )

        raise ValueError("Unknown distance metric")

    def predict(self, X):

        X = np.asarray(X, dtype=float)

        predictions = []

        for x in X:

            distances = self._distance(x)

            indices = np.argsort(distances)[:self.k]

            predictions.append(
                np.mean(self.y_train[indices])
            )

        return np.array(predictions)


def mean_squared_error(y_true, y_pred):
    return np.mean(
        (y_true - y_pred) ** 2
    )


def mean_absolute_error(y_true, y_pred):
    return np.mean(
        np.abs(y_true - y_pred)
    )