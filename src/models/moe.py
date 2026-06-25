"""Mixture-of-experts classifier."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.config import settings


class MixtureOfExpertsClassifier(BaseEstimator, ClassifierMixin):
    """Blend expert probabilities using a learned gating classifier."""

    def __init__(
        self,
        experts,
        gate=None,
        gate_size: float = 0.25,
        random_state: int = settings.random_seed,
    ):
        self.experts = experts
        self.gate = gate
        self.gate_size = gate_size
        self.random_state = random_state

    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        X_expert, X_gate, y_expert, y_gate = train_test_split(
            X,
            y,
            test_size=self.gate_size,
            stratify=y,
            random_state=self.random_state,
        )

        gate_experts = [
            (name, clone(model).fit(X_expert, y_expert)) for name, model in self.experts
        ]
        gate_targets = self._best_expert_targets(gate_experts, X_gate, y_gate)

        if len(np.unique(gate_targets)) == 1:
            self.gate_ = DummyClassifier(
                strategy="constant", constant=int(gate_targets[0])
            )
        else:
            self.gate_ = clone(
                self.gate
                if self.gate is not None
                else DecisionTreeClassifier(max_depth=8, random_state=self.random_state)
            )
        self.gate_.fit(X_gate, gate_targets)

        self.experts_ = [(name, clone(model).fit(X, y)) for name, model in self.experts]
        self.expert_names_ = [name for name, _ in self.experts_]
        return self

    def predict_proba(self, X):
        expert_probas = np.stack(
            [self._aligned_predict_proba(model, X) for _, model in self.experts_],
            axis=1,
        )
        gate_proba = self._gate_predict_proba(X)
        return np.einsum("ne,nec->nc", gate_proba, expert_probas)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def _best_expert_targets(self, experts, X, y):
        y_values = np.asarray(y)
        scores = []
        for _, model in experts:
            proba = self._aligned_predict_proba(model, X)
            true_indices = np.searchsorted(self.classes_, y_values)
            true_proba = proba[np.arange(len(y_values)), true_indices]
            predictions = self.classes_[np.argmax(proba, axis=1)]
            correct = predictions == y_values
            scores.append(true_proba + correct.astype(float))
        return np.argmax(np.column_stack(scores), axis=1)

    def _aligned_predict_proba(self, model, X):
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            model_classes = model.classes_
        else:
            predictions = model.predict(X)
            model_classes = self.classes_
            proba = np.zeros((len(predictions), len(self.classes_)))
            indices = np.searchsorted(self.classes_, predictions)
            proba[np.arange(len(predictions)), indices] = 1.0

        aligned = np.zeros((proba.shape[0], len(self.classes_)))
        for source_index, label in enumerate(model_classes):
            target_index = np.where(self.classes_ == label)[0][0]
            aligned[:, target_index] = proba[:, source_index]
        return aligned

    def _gate_predict_proba(self, X):
        raw = self.gate_.predict_proba(X)
        aligned = np.zeros((raw.shape[0], len(self.experts_)))
        for source_index, expert_index in enumerate(self.gate_.classes_):
            aligned[:, int(expert_index)] = raw[:, source_index]
        return aligned
