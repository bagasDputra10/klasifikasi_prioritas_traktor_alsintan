from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class MinMaxComposite(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        X = np.asarray(X, dtype=float)

        self.min_ = X.min(axis=0)

        rng = X.max(axis=0) - self.min_

        self.range_ = np.where(
            rng == 0,
            1.0,
            rng
        )

        return self

    def transform(self, X):

        X = np.asarray(X, dtype=float)

        Z = np.clip(
            (X - self.min_) / self.range_,
            0,
            1
        )

        return (
            (
                Z[:, 0]
                + Z[:, 1]
                + (1 - Z[:, 2])
            ) / 3.0
        ).reshape(-1, 1)