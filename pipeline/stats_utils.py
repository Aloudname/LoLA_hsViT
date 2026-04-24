"""lightweight statistics helpers for dataset / paper figures."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def format_anonymous_patient_id(raw_id: str, ordered_unique_ids: Sequence[str]) -> str:
    """Map raw patient id to P001-style label using stable sorted order."""
    oid = [str(x) for x in ordered_unique_ids]
    i = oid.index(str(raw_id))
    return f"P{i + 1:03d}"


def pairwise_cosine_matrix(curves: np.ndarray) -> np.ndarray:
    """curves: (n, d) row-wise cosine similarity matrix (n, n)."""
    x = np.asarray(curves, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    xn = x / norms
    return (xn @ xn.T).astype(np.float64)


def bootstrap_cosine_similarity_summary(
    curves: np.ndarray,
    n_bootstrap: int,
    seed: int,
    patient_labels: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Bootstrap over spectral band indices (with replacement) for each patient pair.
    curves: (n_patients, n_bands) mean spectra (e.g. one row per patient for a fixed class).
    """
    x = np.asarray(curves, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        return {"pairwise": [], "global_mean_cosine": None, "global_median_cosine": None}

    n, d = x.shape
    rng = np.random.default_rng(int(seed))
    triu_i, triu_j = np.triu_indices(n, k=1)
    pairwise: List[Dict[str, Any]] = []
    full_sims: List[float] = []

    for k in range(len(triu_i)):
        i, j = int(triu_i[k]), int(triu_j[k])
        vi, vj = x[i], x[j]
        sim_full = float(np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj) + 1e-12))
        full_sims.append(sim_full)
        boot_sims: List[float] = []
        for _ in range(max(1, int(n_bootstrap))):
            idx = rng.choice(d, size=d, replace=True)
            a, b = vi[idx], vj[idx]
            boot_sims.append(
                float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
            )
        bs = np.asarray(boot_sims, dtype=np.float64)
        entry: Dict[str, Any] = {
            "patient_index_i": i,
            "patient_index_j": j,
            "cosine": sim_full,
            "boot_mean": float(bs.mean()),
            "boot_p05": float(np.percentile(bs, 5)),
            "boot_p95": float(np.percentile(bs, 95)),
        }
        if patient_labels is not None and len(patient_labels) > max(i, j):
            entry["patient_id_i"] = str(patient_labels[i])
            entry["patient_id_j"] = str(patient_labels[j])
        pairwise.append(entry)

    return {
        "pairwise": pairwise,
        "global_mean_cosine": float(np.mean(full_sims)) if full_sims else None,
        "global_median_cosine": float(np.median(full_sims)) if full_sims else None,
        "n_bootstrap": int(n_bootstrap),
        "n_bands": int(d),
    }
