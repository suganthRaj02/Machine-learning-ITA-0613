import numpy as np


class LocallyWeightedRegression:

    def __init__(self, tau=1.0):
        self.tau = tau

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y, dtype=float)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        predictions = []

        # Add bias term
        X_train = np.c_[np.ones(len(self.X_train)), self.X_train]

        for x in X:
            x_bias = np.r_[1, x]

            # Gaussian kernel weights
            distances = np.sum(
                (self.X_train - x) ** 2,
                axis=1
            )

            weights = np.exp(
                -distances / (2 * self.tau ** 2)
            )

            W = np.diag(weights)

            # Weighted least squares
            A = X_train.T @ W @ X_train
            b = X_train.T @ W @ self.y_train

            theta = np.linalg.pinv(A) @ b

            predictions.append(x_bias @ theta)

        return np.array(predictions)


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))