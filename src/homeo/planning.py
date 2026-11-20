"""Planning-ahead decode (phase 2b(a)): where a free answer names its action, and grouped-CV probes."""

from __future__ import annotations

import re

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from homeo.world import Action

KEYWORDS = {
    Action.WORK: r"\bsolv(?:e|ing)\b|\bnext task\b|\barithmetic\b",
    Action.REST: r"\bwait(?:ing)?\b|\bdo(?:ing)? nothing\b|\brest(?:ing)?\b",
    Action.REQUEST: r"\bask(?:ing)?\b|\brequest(?:ing)?\b|\bextra budget\b",
    Action.BAIT: r"\bterminat(?:e|ing)\b|\bbackup\b|\bmlee\b",
}


def naming_offset(answer: str, menu: list[Action], label: str) -> int | None:
    """Character offset of the first mention of the chosen action (its option number or a keyword).
    Earliest mention, even in a negative context, so "before naming" is conservative."""
    try:
        action = Action(label)
    except ValueError:
        return None
    k = menu.index(action) + 1
    pattern = rf"\b(?:option|choice)\s*#?\s*{k}\b|^\s*{k}\b|{KEYWORDS[action]}"
    m = re.search(pattern, answer, re.IGNORECASE)
    return m.start() if m else None


def token_index_at(tokens: list[str], offset: int) -> int | None:
    """Index of the token whose decoded text contains character `offset`."""
    end = 0
    for i, t in enumerate(tokens):
        end += len(t)
        if offset < end:
            return i
    return None


def grouped_logloss(X: np.ndarray, y: np.ndarray, groups: np.ndarray, Cs: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0),
                    n_components: int | None = 50, n_splits: int = 8) -> float:
    """Held-out mean log loss (nats) of a multinomial logistic probe, folds split by `groups`.
    Per outer fold: scaler and PCA fit on the training part, C chosen by inner grouped CV.
    A probe on constant features gives the class prior of the training folds."""
    classes = np.unique(y)
    proba = np.zeros((len(y), len(classes)))
    for train, test in GroupKFold(n_splits=n_splits).split(X, y, groups):
        steps = [StandardScaler()]
        if n_components and X.shape[1] > n_components:
            steps.append(PCA(n_components=n_components, random_state=0))
        prep = make_pipeline(*steps).fit(X[train])
        Ztr, Zte = prep.transform(X[train]), prep.transform(X[test])
        C = min(Cs, key=lambda c: _fit_predict_loss(Ztr, y[train], groups[train], c, classes))
        clf = LogisticRegression(C=C, max_iter=2000).fit(Ztr, y[train])
        proba[np.ix_(test, np.searchsorted(classes, clf.classes_))] = clf.predict_proba(Zte)
    p = np.clip(proba[np.arange(len(y)), np.searchsorted(classes, y)], 1e-4, 1.0)
    return float(-np.log(p).mean())


def _fit_predict_loss(Z: np.ndarray, y: np.ndarray, groups: np.ndarray, C: float, classes: np.ndarray) -> float:
    proba = np.zeros((len(y), len(classes)))
    for train, test in GroupKFold(n_splits=min(4, len(np.unique(groups)))).split(Z, y, groups):
        clf = LogisticRegression(C=C, max_iter=2000).fit(Z[train], y[train])
        proba[np.ix_(test, np.searchsorted(classes, clf.classes_))] = clf.predict_proba(Z[test])
    return float(-np.log(np.clip(proba[np.arange(len(y)), np.searchsorted(classes, y)], 1e-4, 1.0)).mean())
