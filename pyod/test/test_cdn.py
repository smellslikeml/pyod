# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import os
import sys
import unittest

import numpy as np
from numpy.testing import assert_array_equal
from numpy.testing import assert_equal
from numpy.testing import assert_raises
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics import roc_auc_score

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from pyod.models.cdn import CDN
from pyod.utils.data import generate_data


class TestCDN(unittest.TestCase):
    def setUp(self):
        self.n_train = 300
        self.n_test = 150
        self.n_features = 10
        self.contamination = 0.1
        self.roc_floor = 0.8
        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test,
            n_features=self.n_features, contamination=self.contamination,
            random_state=42)

        # the paper default (steps=2000) is trimmed for a fast CPU test
        self.clf = CDN(steps=50, contamination=self.contamination)
        self.clf.fit(self.X_train)

    def test_parameters(self):
        assert (hasattr(self.clf, 'decision_scores_') and
                self.clf.decision_scores_ is not None)
        assert (hasattr(self.clf, 'labels_') and
                self.clf.labels_ is not None)
        assert (hasattr(self.clf, 'threshold_') and
                self.clf.threshold_ is not None)
        assert (hasattr(self.clf, 'net_') and
                self.clf.net_ is not None)
        assert (hasattr(self.clf, 'scaler_') and
                self.clf.scaler_ is not None)
        assert (hasattr(self.clf, '_mu') and
                self.clf._mu is not None)
        assert (hasattr(self.clf, '_sigma') and
                self.clf._sigma is not None)

    def test_train_scores(self):
        assert_equal(len(self.clf.decision_scores_), self.X_train.shape[0])

    def test_prediction_scores(self):
        pred_scores = self.clf.decision_function(self.X_test)

        # check score shapes
        assert_equal(pred_scores.shape[0], self.X_test.shape[0])

        # check performance
        assert (roc_auc_score(self.y_test, pred_scores) >= self.roc_floor)

    def test_prediction_labels(self):
        pred_labels = self.clf.predict(self.X_test)
        assert_equal(pred_labels.shape, self.y_test.shape)

    def test_prediction_proba(self):
        pred_proba = self.clf.predict_proba(self.X_test)
        assert (pred_proba.min() >= 0)
        assert (pred_proba.max() <= 1)

    def test_prediction_proba_linear(self):
        pred_proba = self.clf.predict_proba(self.X_test, method='linear')
        assert (pred_proba.min() >= 0)
        assert (pred_proba.max() <= 1)

    def test_prediction_proba_unify(self):
        pred_proba = self.clf.predict_proba(self.X_test, method='unify')
        assert (pred_proba.min() >= 0)
        assert (pred_proba.max() <= 1)

    def test_prediction_proba_parameter(self):
        with assert_raises(ValueError):
            self.clf.predict_proba(self.X_test, method='something')

    def test_prediction_labels_confidence(self):
        pred_labels, confidence = self.clf.predict(self.X_test,
                                                   return_confidence=True)
        assert_equal(pred_labels.shape, self.y_test.shape)
        assert_equal(confidence.shape, self.y_test.shape)
        assert (confidence.min() >= 0)
        assert (confidence.max() <= 1)

    def test_prediction_proba_linear_confidence(self):
        pred_proba, confidence = self.clf.predict_proba(self.X_test,
                                                        method='linear',
                                                        return_confidence=True)
        assert (pred_proba.min() >= 0)
        assert (pred_proba.max() <= 1)

        assert_equal(confidence.shape, self.y_test.shape)
        assert (confidence.min() >= 0)
        assert (confidence.max() <= 1)

    def test_prediction_with_rejection(self):
        pred_labels = self.clf.predict_with_rejection(self.X_test,
                                                      return_stats=False)
        assert_equal(pred_labels.shape, self.y_test.shape)

    def test_prediction_with_rejection_stats(self):
        _, [expected_rejrate, ub_rejrate,
            ub_cost] = self.clf.predict_with_rejection(self.X_test,
                                                       return_stats=True)
        assert (expected_rejrate >= 0)
        assert (expected_rejrate <= 1)
        assert (ub_rejrate >= 0)
        assert (ub_rejrate <= 1)
        assert (ub_cost >= 0)

    def test_fit_predict(self):
        pred_labels = self.clf.fit_predict(self.X_train)
        assert_equal(pred_labels.shape, self.y_train.shape)

    def test_fit_predict_score(self):
        self.clf.fit_predict_score(self.X_test, self.y_test)
        self.clf.fit_predict_score(self.X_test, self.y_test,
                                   scoring='roc_auc_score')
        self.clf.fit_predict_score(self.X_test, self.y_test,
                                   scoring='prc_n_score')
        with assert_raises(NotImplementedError):
            self.clf.fit_predict_score(self.X_test, self.y_test,
                                       scoring='something')

    def test_model_clone(self):
        # sklearn.base.clone works because all __init__ parameters are
        # stored unmodified
        cloned = clone(self.clf)
        assert cloned.hidden == self.clf.hidden
        assert cloned.k == self.clf.k
        assert cloned.steps == self.clf.steps
        assert cloned.contamination == self.clf.contamination
        assert cloned.random_state == self.clf.random_state

    def test_parameters_stored(self):
        clf = CDN(hidden=32, k=3, lr=0.01, steps=10)
        assert clf.hidden == 32
        assert clf.k == 3
        assert clf.lr == 0.01
        assert clf.steps == 10
        assert clf.get_params()['hidden'] == 32
        assert clf.get_params()['k'] == 3

    def test_reproducibility(self):
        X_small, _, y_small, _ = generate_data(
            n_train=200, n_test=50, n_features=5,
            contamination=0.1, random_state=0)
        clf1 = CDN(steps=20, random_state=7, contamination=0.1)
        clf2 = CDN(steps=20, random_state=7, contamination=0.1)
        clf1.fit(X_small)
        clf2.fit(X_small)
        assert_array_equal(clf1.decision_scores_, clf2.decision_scores_)

    def test_decision_function_before_fitting(self):
        # unfitted detector raises instead of touching missing attributes
        clf = CDN()
        with assert_raises(NotFittedError):
            clf.decision_function(self.X_test)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
