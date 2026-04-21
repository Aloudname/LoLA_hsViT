from __future__ import annotations

# metric analyzer for multi-class segmentation.
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


@dataclass
class MetricsBundle:
    """structured metric result."""

    summary: Dict[str, float]
    per_class: Dict[str, Dict[str, float]]
    confusion_matrix: np.ndarray
    roc_auc: Dict[str, float]
    roc_curves: Dict[str, Dict[str, np.ndarray]]


class Analyzer:
    """compute segmentation metrics and summary reports."""

    def __init__(
        self,
        class_names: Sequence[str],
        ignore_index: Optional[int] = None,
        small_target_class: int = 1,
    ) -> None:
        self.class_names = list(class_names)
        self.num_classes = len(self.class_names)
        self.ignore_index = ignore_index
        self.small_target_class = int(small_target_class)

    def compute_metrics(
        self,
        preds: Sequence[np.ndarray],
        targets: Sequence[np.ndarray],
        probs: Optional[Sequence[np.ndarray]] = None,
    ) -> MetricsBundle:
        """compute confusion-based metrics and optional roc-auc.

        input:
            preds: list of predicted masks (h, w).
            targets: list of ground-truth masks (h, w).
            probs: optional list of probabilities (c, h, w).
        output:
            MetricsBundle with summary, per-class metrics, cm, roc_auc, roc_curves.
        """
        y_pred, y_true = self._flatten(preds, targets)
        cm = self._confusion_matrix(y_true, y_pred)

        tp = np.diag(cm).astype(np.float64)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        tn = cm.sum() - (tp + fp + fn)

        precision = tp / np.maximum(tp + fp, 1.0)
        recall = tp / np.maximum(tp + fn, 1.0)
        f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-8)
        iou = tp / np.maximum(tp + fp + fn, 1.0)
        dice = 2.0 * tp / np.maximum(2.0 * tp + fp + fn, 1.0)
        acc = (tp + tn) / np.maximum(tp + tn + fp + fn, 1.0)

        class_range = list(range(self.num_classes))
        fg_range = [c for c in class_range if c != 0]

        per_class: Dict[str, Dict[str, float]] = {}
        for cls_idx, cls_name in enumerate(self.class_names):
            per_class[cls_name] = {
                "precision": float(precision[cls_idx]),
                "recall": float(recall[cls_idx]),
                "f1": float(f1[cls_idx]),
                "iou": float(iou[cls_idx]),
                "dice": float(dice[cls_idx]),
                "accuracy": float(acc[cls_idx]),
                "support": int(cm.sum(axis=1)[cls_idx]),
            }

        summary = {
            "pixel_accuracy": float(tp.sum() / np.maximum(cm.sum(), 1.0)),
            "dice_mean": float(np.mean(dice[fg_range] if fg_range else dice)),
            "iou_mean": float(np.mean(iou[fg_range] if fg_range else iou)),
            "precision_mean": float(np.mean(precision[fg_range] if fg_range else precision)),
            "recall_mean": float(np.mean(recall[fg_range] if fg_range else recall)),
            "f1_mean": float(np.mean(f1[fg_range] if fg_range else f1)),
            "small_target_recall": float(recall[self.small_target_class])
            if 0 <= self.small_target_class < self.num_classes
            else 0.0,
        }

        roc_auc: Dict[str, float] = {}
        roc_curves: Dict[str, Dict[str, np.ndarray]] = {}
        if probs is not None and len(probs) > 0:
            y_true_1d, y_prob_2d = self._flatten_probs(targets, probs)
            for cls_idx, cls_name in enumerate(self.class_names):
                y_true_bin = (y_true_1d == cls_idx).astype(np.int64)
                if np.unique(y_true_bin).size < 2:
                    roc_auc[cls_name] = float("nan")
                    continue
                try:
                    fpr, tpr, _ = roc_curve(y_true_bin, y_prob_2d[:, cls_idx], drop_intermediate=True)
                    roc_auc[cls_name] = float(roc_auc_score(y_true_bin, y_prob_2d[:, cls_idx]))
                    roc_curves[cls_name] = {
                        "fpr": fpr.astype(np.float32),
                        "tpr": tpr.astype(np.float32),
                    }
                except Exception:
                    roc_auc[cls_name] = float("nan")

        return MetricsBundle(
            summary=summary,
            per_class=per_class,
            confusion_matrix=cm,
            roc_auc=roc_auc,
            roc_curves=roc_curves,
        )

    def summarize(self, bundle: MetricsBundle) -> str:
        """format metric summary as text table."""
        s = bundle.summary
        lines = [
            "metric summary:",
            f"  pixel_accuracy: {s['pixel_accuracy']:.4f}",
            f"  dice_mean: {s['dice_mean']:.4f}",
            f"  iou_mean: {s['iou_mean']:.4f}",
            f"  precision_mean: {s['precision_mean']:.4f}",
            f"  recall_mean: {s['recall_mean']:.4f}",
            f"  f1_mean: {s['f1_mean']:.4f}",
            f"  small_target_recall: {s['small_target_recall']:.4f}",
        ]
        return "\n".join(lines)

    def _flatten(self, preds: Sequence[np.ndarray], targets: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """flatten masks to 1d arrays with optional ignore filtering."""
        pred_list: List[np.ndarray] = []
        true_list: List[np.ndarray] = []

        for pred, target in zip(preds, targets):
            p = np.asarray(pred).reshape(-1)
            t = np.asarray(target).reshape(-1)
            if self.ignore_index is not None:
                valid = t != self.ignore_index
                p = p[valid]
                t = t[valid]
            pred_list.append(p)
            true_list.append(t)

        return np.concatenate(pred_list), np.concatenate(true_list)

    def _flatten_probs(
        self,
        targets: Sequence[np.ndarray],
        probs: Sequence[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """flatten one-hot targets and probabilities for roc computation."""
        y_true_list: List[np.ndarray] = []
        y_prob_list: List[np.ndarray] = []

        for target, prob in zip(targets, probs):
            t = np.asarray(target).reshape(-1)
            p = np.asarray(prob)
            p = np.moveaxis(p, 0, -1).reshape(-1, p.shape[0])

            if self.ignore_index is not None:
                valid = t != self.ignore_index
                t = t[valid]
                p = p[valid]

            y_true_list.append(t)
            y_prob_list.append(p)

        return np.concatenate(y_true_list), np.concatenate(y_prob_list)

    def _confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """build confusion matrix with fixed class count."""
        cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        for t, p in zip(y_true, y_pred):
            if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                cm[t, p] += 1
        return cm
