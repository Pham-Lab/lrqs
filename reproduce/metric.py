"""Evaluation metrics for causal-direction predictions."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def prec(flags: ArrayLike) -> float:
    """Fraction of predictions equal to the ground-truth direction (+1)."""

    correct = np.asarray(flags) == 1
    return float(np.mean(correct))


def audrc(flags: ArrayLike) -> float:
    """Average accuracy over every prefix of an ordered result sequence."""

    correct = (np.asarray(flags) == 1).astype(float)
    prefix_accuracy = np.cumsum(correct) / np.arange(1, correct.size + 1)
    return float(np.mean(prefix_accuracy))


def od(order: NDArray[np.integer], graph: NDArray) -> float:
    """Count graph edges that point against a proposed node ordering."""

    error = 0.0
    for position, node in enumerate(order):
        error += float(graph[order[position + 1 :], node].sum())
    return error


def odr(order: NDArray[np.integer], graph: NDArray) -> float:
    """Normalize ordering disagreement by the total graph edge weight."""

    return float(od(order, graph) / np.sum(graph))


__all__ = ["prec", "audrc", "od", "odr"]
