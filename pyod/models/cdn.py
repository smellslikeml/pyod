# -*- coding: utf-8 -*-

"""Cauchy Displacement Network (CDN) for tabular anomaly detection.
Part of the codes are adapted from the reference implementation at
https://github.com/charles-cao/CDN
"""
# Author: Yang Cao and Zihao Guo <https://github.com/charles-cao/CDN>
# License: BSD 2 clause


import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from .base import BaseDetector


class CauchyActivation(nn.Module):
    """Cauchy activation with trainable parameters for every hidden unit.

    Each unit applies ``phi_j(z) = (alpha_j * z + beta_j) /
    (z^2 + gamma_j^2 + eps)``, where ``alpha_j``, ``beta_j``, and
    ``gamma_j`` are learned jointly with the rest of the network.
    """

    def __init__(self, num_features, alpha_init=0.1, beta_init=0.2,
                 gamma_init=0.6, eps=1e-6):
        super(CauchyActivation, self).__init__()
        self.alpha = nn.Parameter(torch.full((num_features,), alpha_init))
        self.beta = nn.Parameter(torch.full((num_features,), beta_init))
        self.gamma = nn.Parameter(torch.full((num_features,), gamma_init))
        self.eps = eps

    def forward(self, z):
        return (self.alpha * z + self.beta) / (
                z.square() + self.gamma.square() + self.eps)


class _CDNNet(nn.Module):
    """Displacement network ``f(x) = W2 phi(W1 x + b1) + b2``.

    ``phi`` is a row of trainable Cauchy units; see :class:`CDN`
    for the detection method built on top of this network.
    """

    def __init__(self, d, hidden, alpha_init, beta_init, gamma_init):
        super(_CDNNet, self).__init__()
        self.fc1 = nn.Linear(d, hidden)
        self.act = CauchyActivation(hidden, alpha_init, beta_init,
                                    gamma_init)
        self.fc2 = nn.Linear(hidden, d)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class CDN(BaseDetector):
    """Cauchy Displacement Network (CDN) for tabular anomaly detection.

    CDN compacts neighbourhood geometry into a small single-hidden-layer
    network with trainable Cauchy activation units. Each training point is
    regressed onto the mean of its ``k`` nearest training neighbours, so
    the network learns the local displacement field of the normal data. A
    query is scored by the displacement length between the query and the
    network output; inference takes a single forward pass and needs
    neither the training data nor a neighbour search.

    Although the paper evaluates CDN in the semi-supervised setting
    (trained on normal data only), the detector follows the standard
    unsupervised PyOD API and ignores ``y``.

    See Cao and Guo, "Anomalies as Residuals: Cauchy Displacement
    Networks", arXiv:2610.41387, 2026 for details.

    Parameters
    ----------
    hidden : int, optional (default=128)
        Width of the single hidden layer.

    k : int, optional (default=5)
        Number of nearest neighbours used to build the regression
        targets (the mean of the ``k`` nearest other training points).

    lr : float, optional (default=1e-3)
        Learning rate of the Adam optimizer.

    batch_size : int, optional (default=32)
        Mini-batch size; ``min(batch_size, n_samples)`` is used.

    steps : int, optional (default=2000)
        Number of training steps. The default is the configuration used
        for all datasets in the paper; smaller values trade accuracy for
        a faster fit.

    wd : float, optional (default=1e-5)
        Weight decay of the Adam optimizer.

    alpha_init, beta_init, gamma_init : float, optional
        (default=0.1, 0.2, 0.6)
        Initial values of the trainable Cauchy activation parameters
        ``alpha_j``, ``beta_j``, and ``gamma_j`` of every hidden unit.

    contamination : float in (0., 0.5], optional (default=0.1)
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. Used when fitting to define the
        threshold on the decision function.

    random_state : int, optional (default=42)
        Seed for network initialization and mini-batch sampling, so
        training is reproducible across calls to :meth:`fit`. Pass
        ``None`` to skip seeding.

    Attributes
    ----------
    net_ : torch.nn.Module
        Fitted displacement network.

    scaler_ : StandardScaler
        Standardizer fit on the training data.

    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data. The higher, the more
        abnormal. Outliers tend to have higher scores. This value is
        available once the detector is fitted.

    threshold_ : float
        The threshold is based on ``contamination``. It is the
        ``n_samples * contamination`` most abnormal samples in
        ``decision_scores_``. The threshold is calculated for generating
        binary outlier labels.

    labels_ : numpy array of shape (n_samples,)
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers/anomalies. It is generated by applying
        ``threshold_`` on ``decision_scores_``.
    """

    def __init__(self, hidden=128, k=5, lr=1e-3, batch_size=32, steps=2000,
                 wd=1e-5, alpha_init=0.1, beta_init=0.2, gamma_init=0.6,
                 contamination=0.1, random_state=42):
        super(CDN, self).__init__(contamination=contamination)
        self._classes = 2
        self.hidden = hidden
        self.k = k
        self.lr = lr
        self.batch_size = batch_size
        self.steps = steps
        self.wd = wd
        self.alpha_init = alpha_init
        self.beta_init = beta_init
        self.gamma_init = gamma_init
        self.random_state = random_state

    def _train(self, Xs, target):
        """Optimize the displacement network by mini-batch Adam.

        Parameters
        ----------
        Xs : numpy array of shape (n_samples, n_features)
            Standardized training data.

        target : numpy array of shape (n_samples, n_features)
            Regression target of each training point, i.e. the mean of
            its ``k`` nearest other training points.
        """
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            sample_generator = torch.Generator().manual_seed(
                self.random_state)
        else:
            sample_generator = torch.Generator()

        n_samples, n_features = Xs.shape
        self.net_ = _CDNNet(n_features, self.hidden, self.alpha_init,
                            self.beta_init, self.gamma_init)
        optimizer = torch.optim.Adam(self.net_.parameters(), lr=self.lr,
                                     weight_decay=self.wd)
        X_tensor = torch.from_numpy(Xs)
        target_tensor = torch.from_numpy(target)
        batch_size = min(self.batch_size, n_samples)

        self.net_.train()
        for _ in range(self.steps):
            sample_idx = torch.randint(n_samples, (batch_size,),
                                       generator=sample_generator)
            optimizer.zero_grad()
            loss = ((self.net_(X_tensor[sample_idx])
                     - target_tensor[sample_idx]) ** 2).mean()
            loss.backward()
            optimizer.step()
        self.net_.eval()

    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = check_array(X)
        self._set_n_classes(y)

        # standardize, then regress each point onto the mean of its k
        # nearest other training points
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X).astype(np.float32)
        n_neighbors = min(self.k + 1, Xs.shape[0])
        neighbors = NearestNeighbors(n_neighbors=n_neighbors).fit(Xs)
        neighbor_idx = neighbors.kneighbors(Xs, return_distance=False)[:, 1:]
        target = Xs[neighbor_idx].mean(axis=1).astype(np.float32)

        self._train(Xs, target)

        self.decision_scores_ = self.decision_function(X)
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        """Predict raw anomaly scores of X using the fitted detector.

        The anomaly score of a sample is the displacement length between
        the sample and the network output. For consistency, outliers are
        assigned with higher anomaly scores.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        check_is_fitted(self, ['net_', 'scaler_'])
        X = check_array(X)

        Xs = torch.from_numpy(self.scaler_.transform(X).astype(np.float32))
        self.net_.eval()
        with torch.no_grad():
            # this is the outlier score
            return (self.net_(Xs) - Xs).norm(dim=1).numpy()
