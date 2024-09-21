import numpy as np
from scipy.linalg import inv, det, LinAlgError
from sklearn.preprocessing import StandardScaler


def _class_means(X, y):
    """ Calculate the class means."""
    classes = np.unique(y)
    means = np.zeros((len(classes), X.shape[1]))

    for idx, cls in enumerate(classes):
        means[idx] = np.mean(X[y == cls], axis=0)
    return means


def _class_var(X, y):
    """ Calculate the class variances."""
    classes = np.unique(y)
    n_features = X.shape[1]
    variances = np.zeros((len(classes), n_features))
    for idx, cls in enumerate(classes):
        variances[idx] = np.var(X[y == cls], axis=0)
    return variances


def check_fit(model):
    """ Check if the model is fit."""
    if model.means_ is None:
        raise ValueError("Model is not fit yet.")


class LDAModel:
    def __init__(self, priors_=1.0):
        self.priors_ = priors_
        self.classes_ = np.array([])
        self.n_classes = None
        self.means_ = None
        self.covariance_ = None
        self.weights = None
        self.bias = None

    def _class_cov(self, X, y):
        """ Calculate the class covariance."""

        cov = np.zeros((X.shape[1], X.shape[1]))
        for i in range(self.n_classes):
            cov += np.cov(X[y == i].T) * self.priors_[i]

        return cov

    def fit(self, X, y):
        # TODO: Implement the fit method
        if X.shape[0] != len(y):
            raise ValueError("Number of samples in X and y does not match.")
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes = len(self.classes_)
        self.priors_ = np.bincount(y) / len(y)
        self.means_ = _class_means(X, y)
        self.covariance_ = self._class_cov(X, y)
        reg_cov = self.covariance_ + np.eye(X.shape[1]) * 1e-10
        self.weights = np.linalg.solve(reg_cov, self.means_.T).T
        self.bias = -0.5 * np.diag(np.dot(self.means_, self.weights.T)) + np.log(self.priors_)
        return self

    def predict(self, X):
        # TODO: Implement the predict method
        check_fit(self)
        scores = np.dot(X, self.weights.T) + self.bias
        indicts = np.argmax(scores, axis=1)
        return self.classes_[indicts]


class QDAModel:
    def __init__(self, priors_=1.0,eps=1e-10):
        self.eps = eps
        self.scaler = None
        self.priors_ = priors_
        self.classes_ = np.array([])
        self.n_classes = None
        self.means_ = None
        self.covariance_ = None
        self.log_det_covariances_ = np.array([])
        self.covariances_inv_ = np.array([])

    def _class_cov(self, X, y):
        """ Calculate the class covariance."""

        cov = np.zeros((self.n_classes, X.shape[1], X.shape[1]))
        cov_inv = np.zeros((self.n_classes, X.shape[1], X.shape[1]))
        log_det_cov = np.zeros(self.n_classes)
        for i in range(self.n_classes):

            cov[i] = np.cov(X[y == i], rowvar=False) + np.eye(cov[i].shape[0]) * self.eps
            try:
                cov_inv[i] = inv(cov[i])
                log_det_cov[i] = np.log(det(cov_inv[i]) + self.eps)
            except LinAlgError:
                cov_inv[i] = np.linalg.pinv(cov[i])
                log_det_cov[i] = np.log(det(cov[i]) + self.eps)
                print("Warning: LinAlgError encountered. Using pseudo-inverse.")

        return cov, cov_inv, log_det_cov

    def fit(self, X, y):
        # TODO: Implement the fit method
        if X.shape[0] != len(y):
            raise ValueError("Number of samples in X and y does not match.")
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes = len(self.classes_)
        self.priors_ = np.bincount(y) / len(y)
        self.means_ = _class_means(X, y)
        self.covariance_, self.covariances_inv_, self.log_det_covariances_ = self._class_cov(X, y)

        return self

    def predict(self, X):
        # TODO: Implement the predict method
        check_fit(self)

        scores = np.zeros((len(X), self.n_classes))

        for i in range(self.n_classes):
            diff = X - self.means_[i]
            inv_cov = self.covariances_inv_[i]
            log_det_cov = self.log_det_covariances_[i]

            prior = np.log(self.priors_[i])
            sum_diff_cov = np.sum(np.dot(diff, inv_cov) * diff, axis=1)
            pro_k = prior - 0.5 * sum_diff_cov - 0.5 * log_det_cov
            scores[:, i] = pro_k

        indicts = np.argmax(scores, axis=1)
        return indicts

    # def predict1(self, X):
    #     # TODO: Implement the predict method
    #     check_fit(self)
    #
    #     scores = np.zeros((len(X), self.n_classes))
    #     for n in range(len(X)):
    #         x = X[n]
    #         for i in range(self.n_classes):
    #             diff = x - self.means_[i]
    #             inv_cov = self.covariances_inv_[i]
    #             log_det_cov = self.log_det_covariances_[i]
    #             prior = np.log(self.priors_[i])
    #             sum_diff_cov = np.dot(np.dot(diff, inv_cov), diff.T)
    #             pro_k = prior - 0.5 * sum_diff_cov + 0.5 * log_det_cov
    #             scores[n, i] = pro_k
    #
    #     indicts = np.argmax(scores, axis=1)
    #     return indicts

class GaussianNBModel:
    def __init__(self, eps=1e-10):
        self.variances_ = None
        self.priors_ = None
        self.n_classes = None
        self.classes_ = None
        self.means_ = None
        self.n_features = None
        self.eps = eps

    def init_count(self, X, y):
        self.n_features = X.shape[1]
        self.priors_ = np.bincount(y) / len(y)
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes = len(self.classes_)
        self.means_ = np.zeros((self.n_classes, self.n_features))
        self.variances_ = np.zeros((self.n_classes, self.n_features))

    def fit(self, X, y):
        # TODO: Implement the fit method
        if X.shape[0] != len(y):
            raise ValueError("Number of samples in X and y does not match.")
        self.init_count(X, y)
        self.means_ = _class_means(X, y)
        self.variances_ = _class_var(X, y) + self.eps * np.var(X, axis=0).max()

    def pdf(self, class_idx, x):
        mean = self.means_[class_idx]
        var = self.variances_[class_idx]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def predict1(self, X):
        # TODO: Implement the predict method
        check_fit(self)
        scores = np.zeros((len(X), self.n_classes))
        for n in range(len(X)):
            x = X[n]
            for i in range(self.n_classes):
                prior = np.log(self.priors_[i])
                conditional = np.sum(np.log(self.pdf(i, x)))
                poster_k = prior + conditional
                scores[n, i] = poster_k
        indicts = np.argmax(scores, axis=1)
        return indicts

    def predict(self, X):
        # TODO: Implement the predict method
        check_fit(self)
        scores = np.zeros((len(X), self.n_classes))
        for n in range(len(X)):
            x = X[n]
            for i in range(self.n_classes):
                prior = np.log(self.priors_[i])
                diff = x-self.means_[i]
                var = self.variances_[i]
                log_var = -np.sum(np.log(var))/2
                conditional = -np.sum((diff**2)/(2*var))
                poster_k = prior + conditional+log_var
                scores[n, i] = poster_k
        indicts = np.argmax(scores, axis=1)
        return indicts
