"""Merge two LoRA adapters into one by rank-concatenation.

LoRA convention here: the weight delta is `B @ A`, with A of shape (r, in) and
B of shape (out, r). Two adapters trained separately give deltas B1·A1 and
B2·A2, and the model you want applies the sum of their effects, B1·A1 + B2·A2.

Concatenating along the rank dimension produces an adapter of rank r1+r2 whose
single delta equals that sum exactly:

    A = [[A1],        B = [B1 | B2]      B·A = B1·A1 + B2·A2
         [A2]]

No approximation, no shared-subspace assumption. Adding the adapters elementwise
(A1+A2, B1+B2) would assume they live in the same subspace, which independently
trained adapters do not.
"""

from __future__ import annotations

import numpy as np


def rank_concat(A1: np.ndarray, B1: np.ndarray, A2: np.ndarray, B2: np.ndarray):
    """Return (A, B) for the merged adapter. Raises ValueError on shape mismatch.

    A1, A2: (r, in).  B1, B2: (out, r).  Result A: (r1+r2, in), B: (out, r1+r2).
    """
    A1, B1, A2, B2 = (np.asarray(x) for x in (A1, B1, A2, B2))

    for name, arr in (("A1", A1), ("B1", B1), ("A2", A2), ("B2", B2)):
        if arr.ndim != 2:
            raise ValueError(f"{name} must be 2D, got shape {arr.shape}")

    r1, in1 = A1.shape
    r2, in2 = A2.shape
    out1, rb1 = B1.shape
    out2, rb2 = B2.shape

    if in1 != in2:
        raise ValueError(f"input dims differ: A1 in={in1}, A2 in={in2}")
    if out1 != out2:
        raise ValueError(f"output dims differ: B1 out={out1}, B2 out={out2}")
    if r1 != rb1:
        raise ValueError(f"adapter 1 rank mismatch: A1 r={r1}, B1 r={rb1}")
    if r2 != rb2:
        raise ValueError(f"adapter 2 rank mismatch: A2 r={r2}, B2 r={rb2}")

    A = np.concatenate([A1, A2], axis=0)     # (r1+r2, in)
    B = np.concatenate([B1, B2], axis=1)     # (out, r1+r2)
    return A, B


def delta(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """The weight delta B @ A for an adapter. Handy for verifying a merge."""
    return np.asarray(B) @ np.asarray(A)
