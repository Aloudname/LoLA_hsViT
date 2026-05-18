from __future__ import annotations

from tqdm import tqdm
from munch import Munch
from pathlib import Path
from collections import OrderedDict
from scipy.linalg import eig
from dataclasses import dataclass
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
from tqdm import TqdmExperimentalWarning
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from pipeline.monitor import tprint
import os, cv2, json, torch, shutil, hashlib, warnings, itertools, numpy as np

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)


@dataclass
class SampleItem:
    """single sample metadata."""

    sample_id: str
    patient_id: str
    hsi_path: Path
    mask_path: Path
    rgb_path: Optional[Path]


@dataclass
class PatchRecord:
    """single patch index record."""

    sample_index: int
    top: int
    left: int
    weight: float = 1.0
    has_precious: bool = False
    fg_ratio: float = 0.0
    bg_ratio: float = 1.0
    is_hard_bg: bool = False


@dataclass
class PreparedData:
    """prepared split and patch cache shared by train/eval/test datasets."""

    modality: str
    samples_by_split: Dict[str, List[SampleItem]]
    patches_by_split: Dict[str, List[PatchRecord]]
    reducer: "SpectralReducer"
    stats: Dict[str, Any]


class KernelPCA:
    """
    Actually RFF, replace kernel with explicit proj.
    This doesn't do fit per se,
    do random proj with bandwidth gamma instead.
    """
    def __init__(self, input_dim: int,
                output_dim: int, gamma: float = 1.0):
        """
        Args:
            gamma (float): bandwidth parameter, larger -> narrower kernel, more aggressive reduction.
        """
        self.W = np.random.normal(
            scale=np.sqrt(2 * gamma),
            size=(input_dim, output_dim)
        )
        self.b = np.random.uniform(0, 2*np.pi, size=output_dim)

    def transform(self, X):
        return np.sqrt(2.0 / self.W.shape[1]) * np.cos(X @ self.W + self.b)



def _as_munch(config: Mapping[str, Any]) -> Munch:
    """convert mapping to munch."""
    return config if isinstance(config, Munch) else Munch.fromDict(dict(config))


def _config_section_plain(obj: Any) -> Dict[str, Any]:
    """nested munch → plain dict for json/cache keys."""
    if obj is None:
        return {}
    if isinstance(obj, Munch) and hasattr(obj, "toDict"):
        return dict(obj.toDict())
    if isinstance(obj, Mapping):
        return dict(obj)
    return {}


def _safe_ratio_triplet(train_ratio: float, eval_ratio: float, test_ratio: float) -> Tuple[float, float, float]:
    """normalize split ratios to sum 1."""
    total = train_ratio + eval_ratio + test_ratio
    if total <= 0:
        raise ValueError("split ratios must have positive sum")
    return train_ratio / total, eval_ratio / total, test_ratio / total


def _iter_patch_positions(height: int, width: int, patch_size: int, stride: int) -> Iterable[Tuple[int, int]]:
    """iterate patch top-left coordinates with tail coverage."""
    if height <= patch_size:
        ys = [0]
    else:
        ys = list(range(0, height - patch_size + 1, stride))
        if ys[-1] != height - patch_size:
            ys.append(height - patch_size)

    if width <= patch_size:
        xs = [0]
    else:
        xs = list(range(0, width - patch_size + 1, stride))
        if xs[-1] != width - patch_size:
            xs.append(width - patch_size)
    yield from itertools.product(ys, xs)


def _pad_to_patch(image: np.ndarray, mask: np.ndarray, patch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """pad image and mask to at least patch_size."""
    h, w = mask.shape
    pad_h = max(0, patch_size - h)
    pad_w = max(0, patch_size - w)
    if pad_h == 0 and pad_w == 0:
        return image, mask

    image_pad = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    mask_pad = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    return image_pad, mask_pad


def _compute_hist(mask: np.ndarray, num_classes: int) -> np.ndarray:
    """compute class histogram from mask."""
    return np.bincount(mask.reshape(-1), minlength=num_classes).astype(np.int64)


def _should_keep_patch(
    mask_patch: np.ndarray,
    foreground_ratio_threshold: float,
    max_background_ratio: float = 0.9,
    precious_class: int = 1,
) -> bool:
    """patch keep rule.

    keep when:
    - precious_class exists, or
    - foreground ratio >= threshold and background ratio <= max_background_ratio.
    """
    has_small_target = bool(np.any(mask_patch == precious_class))
    if has_small_target:
        return True
    bg_ratio = float(np.mean(mask_patch == 0))
    fg_ratio = float(np.mean(mask_patch > 0))
    return (
        False
        if bg_ratio > max_background_ratio
        else fg_ratio >= foreground_ratio_threshold
    )


class SpectralReducer:
    """fit/transform spectral channels for method-b preprocessing.

    Unified HSI order in ``prepare_data`` (``pipeline/dataset.prepare_data``):
    1) ``ensure_spectral_alignment_on_disk`` (when ``data.spectral_alignment.enabled``): optional
       spectral diff along bands, pad back to ``data.hsi_bands``, optional per-pixel SNV **on disk**
       (manifest v2: per-file SHA256 so aligned cubes are not rewritten on later runs).
    2) ``SpectralReducer.fit`` / ``transform``: never re-applies diff; when Stage A applied SNV on
       disk, ``enable_snv`` is false and ``derivative_order`` is 0. When Stage A SNV is on disk,
       ``enable_standardize`` is forced off so global z-score is not stacked on top of SNV
       (same contract as ``mode=none`` identity path).

    modes:
    - none: identity transform when Stage A diff+SNV is on disk; else SNV / derivative /
      standardize per ``data.preprocess``.
    - supervised_pca / lda_pca / kernel_lda / pca_nca: fit on pixels already Stage A when
       alignment is enabled (see above); then PCA / LDA / NCA as configured.
    """

    def __init__(
        self,
        mode: str,
        output_dim: int,
        max_fit_pixels: int,
        seed: int,
        preprocess_cfg: Optional[Mapping[str, Any]] = None,
        spectral_alignment_cfg: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.mode = mode.lower()
        self.output_dim = output_dim
        self.max_fit_pixels = max_fit_pixels
        self.seed = seed

        cfg = _as_munch(preprocess_cfg or {})
        align = _as_munch(spectral_alignment_cfg or {})
        self._spectral_alignment_enabled = bool(getattr(align, "enabled", False))
        self._stage_a_snv = self._spectral_alignment_enabled and bool(getattr(align, "snv", False))

        base_snv = bool(getattr(cfg, "snv", self.mode == "pca_nca"))
        self.enable_snv = base_snv and not (self._spectral_alignment_enabled and self._stage_a_snv)

        base_derivative = max(0, int(getattr(cfg, "derivative_order", 1 if self.mode == "pca_nca" else 0)))
        self.derivative_order = 0 if self._spectral_alignment_enabled else base_derivative

        self.savgol_window = max(3, int(getattr(cfg, "savgol_window", 7)))
        self.savgol_polyorder = max(1, int(getattr(cfg, "savgol_polyorder", 2)))
        self.enable_standardize = bool(getattr(cfg, "standardize", self.mode == "pca_nca"))

        self._none_identity = (
            self.mode == "none"
            and self._spectral_alignment_enabled
            and self._stage_a_snv
        )
        if self._spectral_alignment_enabled and self._stage_a_snv:
            self.enable_standardize = False
        self.pca_whiten = bool(getattr(cfg, "pca_whiten", self.mode == "pca_nca"))
        self.pca_dim_override = max(0, int(getattr(cfg, "pca_dim", 0)))
        self.nca_max_fit_pixels = max(256, int(getattr(cfg, "nca_max_fit_pixels", min(max_fit_pixels, 40000))))
        self.nca_max_iter = max(20, int(getattr(cfg, "nca_max_iter", 200)))
        self.nca_tol = float(getattr(cfg, "nca_tol", 1e-5))
        self.nca_init = str(getattr(cfg, "nca_init", "pca"))

        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[Any] = None
        self.lda: Optional[LinearDiscriminantAnalysis] = None
        self.nca: Optional[NeighborhoodComponentsAnalysis] = None
        self.pca_dim = 0
        self.lda_dim = 0
        self.final_dim = 0
        self.reference_projection_name = "PCA"
        self.reduced_projection_name = "Reducer"
        self.fitted = False

    def fit(
        self,
        samples: Sequence[SampleItem],
        num_classes: int,
        show_progress: bool = True,
    ) -> None:
        """fit reducer from train split pixels."""
        if self.mode == "none":
            if self._none_identity:
                self.fitted = True
                self.reference_projection_name = "Stage A (diff+SNV)"
                self.reduced_projection_name = "Stage A (diff+SNV)"
                return

            # none without full Stage A on disk: SNV / derivative / standardize only (no duplicate diff).
            x_fit, _ = self._collect_pixels(samples, num_classes, show_progress=show_progress)
            if x_fit.size == 0:
                raise RuntimeError("failed to collect fit pixels for none-mode normalization")
            _ = self._prepare_features(x_fit, fit=True)

            self.fitted = True
            self.reference_projection_name = "Raw spectra"
            self.reduced_projection_name = "Normalized spectra"
            return

        # --- Discriminative / projection modes (never reached when mode == "none").
        # Pixels come from ``SampleItem.hsi_path`` after ``ensure_spectral_alignment_on_disk``:
        # when ``spectral_alignment.enabled``, cubes are already diff [+ SNV] on disk.
        # ``_prepare_features`` then skips SNV + spectral derivative in that case; with Stage A SNV,
        # ``enable_standardize`` is already false (see __init__).
        x_fit, y_fit = self._collect_pixels(samples, num_classes, show_progress=show_progress)
        if x_fit.size == 0:
            raise RuntimeError("failed to collect fit pixels for spectral reducer")
        x_fit = self._prepare_features(x_fit, fit=True)

        if self.mode == "supervised_pca":
            self.pca_dim = min(self.output_dim, x_fit.shape[-1])
            self.pca = PCA(n_components=self.pca_dim, whiten=self.pca_whiten)
            self.pca.fit(x_fit)
            self.final_dim = self.pca_dim
            self.reference_projection_name = "PCA"
            self.reduced_projection_name = "PCA"
            self.fitted = True
            return

        if self.mode == "lda_pca":
            return self._do_lda_pca(x_fit, y_fit)

        if self.mode == "kernel_lda":
            return self._do_kernel_lda(x_fit, y_fit)

        if self.mode == "pca_nca":
            return self._do_pca_nca(x_fit, y_fit)

        raise ValueError(f"unknown reducer mode: {self.mode}")

    def _resolve_savgol_window(self, num_bands: int) -> int:
        window = min(self.savgol_window, num_bands)
        if window % 2 == 0:
            window -= 1
        min_window = self.savgol_polyorder + 2
        if min_window % 2 == 0:
            min_window += 1
        window = max(window, min_window)
        if window > num_bands:
            window = num_bands if num_bands % 2 == 1 else max(1, num_bands - 1)
        return max(1, window)

    def _prepare_features(self, x: np.ndarray, fit: bool = False) -> np.ndarray:
        feats = np.asarray(x, dtype=np.float32)
        if feats.ndim != 2:
            raise ValueError(f"expected 2d pixel matrix, got shape={tuple(feats.shape)}")

        if self.enable_snv:
            mean = feats.mean(axis=1, keepdims=True)
            std = feats.std(axis=1, keepdims=True)
            feats = (feats - mean) / np.maximum(std, 1e-6)

        if self.derivative_order > 0:
            window = self._resolve_savgol_window(feats.shape[1])
            polyorder = min(self.savgol_polyorder, max(1, window - 1))
            if window >= 3 and self.derivative_order <= polyorder:
                feats = savgol_filter(
                    feats,
                    window_length=window,
                    polyorder=polyorder,
                    deriv=self.derivative_order,
                    axis=1,
                    mode="interp",
                ).astype(np.float32)
            else:
                grad = feats
                for _ in range(self.derivative_order):
                    grad = np.gradient(grad, axis=1).astype(np.float32)
                feats = grad

        feats = np.asarray(feats, dtype=np.float32)

        if self.enable_standardize:
            if fit or self.scaler is None:
                self.scaler = StandardScaler(with_mean=True, with_std=True)
                feats = self.scaler.fit_transform(feats).astype(np.float32)
            else:
                feats = self.scaler.transform(feats).astype(np.float32)

        return feats

    def _fit_pca(self, x_fit: np.ndarray, target_dim: int) -> np.ndarray:
        self.pca_dim = int(min(max(1, target_dim), x_fit.shape[-1], max(1, x_fit.shape[0] - 1)))
        self.pca = PCA(n_components=self.pca_dim, whiten=self.pca_whiten)
        self.pca.fit(x_fit)
        return self.pca.transform(x_fit).astype(np.float32)

    def _balanced_subsample(
        self,
        x: np.ndarray,
        y: np.ndarray,
        budget: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if x.shape[0] <= budget:
            return x, y

        rng = np.random.default_rng(self.seed)
        classes = np.unique(y)
        per_class = max(32, budget // max(1, len(classes)))
        chosen_parts: List[np.ndarray] = []

        for cls_idx in classes:
            idx = np.where(y == cls_idx)[0]
            if idx.size == 0:
                continue
            take = min(idx.size, per_class)
            chosen_parts.append(rng.choice(idx, size=take, replace=False))

        if not chosen_parts:
            chosen = rng.choice(x.shape[0], size=budget, replace=False)
            return x[chosen], y[chosen]

        chosen = np.concatenate(chosen_parts, axis=0)
        if chosen.shape[0] > budget:
            chosen = rng.choice(chosen, size=budget, replace=False)
        elif chosen.shape[0] < budget:
            remaining = np.setdiff1d(np.arange(x.shape[0]), chosen, assume_unique=False)
            if remaining.size > 0:
                extra = rng.choice(remaining, size=min(budget - chosen.shape[0], remaining.size), replace=False)
                chosen = np.concatenate([chosen, extra], axis=0)

        rng.shuffle(chosen)
        return x[chosen], y[chosen]

    def _do_lda_pca(self, x_fit: np.ndarray, y_fit: np.ndarray) -> bool:
        if self.output_dim <= 1:
            tprint("Warning: output_dim <= 1, skipping LDA and fitting PCA with output_dim=1")
            self.pca_dim = min(1, x_fit.shape[-1])
            self.pca = PCA(n_components=self.pca_dim, whiten=self.pca_whiten)
            self.pca.fit(x_fit)
            self.lda = None
            self.lda_dim = 0
            self.final_dim = self.pca_dim
            self.reference_projection_name = "PCA"
            self.reduced_projection_name = "PCA"
            self.fitted = True
            return None

        lda_dim = int(min(max(1, len(np.unique(y_fit)) - 1), self.output_dim - 1))
        pca_dim = int(min(self.output_dim - lda_dim, x_fit.shape[-1]))
        pca_dim = max(1, pca_dim)
        tprint(
            "Fitting spectral reducer with:\n"
            f"\tPCA dim = {pca_dim}\n"
            f"\tLDA dim = {lda_dim}\n"
        )

        x_pca = self._fit_pca(x_fit, target_dim=pca_dim)

        try:
            lda_dim = int(min(lda_dim, x_pca.shape[-1]))
            self.lda = LinearDiscriminantAnalysis(n_components=lda_dim)
            self.lda.fit(x_pca, y_fit)
            self.lda_dim = lda_dim
        except Exception as exc:
            tprint(f"Warning: LDA fit failed, fallback to PCA only. reason={exc}")
            self.lda = None
            self.lda_dim = 0

        self.final_dim = self.pca_dim + self.lda_dim if self.lda is not None else self.pca_dim
        self.reference_projection_name = "PCA"
        self.reduced_projection_name = "PCA + LDA" if self.lda is not None else "PCA"
        self.fitted = True

    def _do_kernel_lda(self, x_fit: np.ndarray, y_fit: np.ndarray) -> bool:
        if self.output_dim <= 1:
            tprint("Warning: output_dim <= 1, skipping Kernel LDA and fitting PCA with output_dim=1")
            self.pca_dim = min(1, x_fit.shape[-1])
            self.pca = PCA(n_components=self.pca_dim, whiten=self.pca_whiten)
            self.pca.fit(x_fit)
            self.lda = None
            self.lda_dim = 0
            self.final_dim = self.pca_dim
            self.reference_projection_name = "Kernel PCA"
            self.reduced_projection_name = "Kernel PCA"
            self.fitted = True
            return None

        lda_dim = int(min(max(1, len(np.unique(y_fit)) - 1), self.output_dim - 1))
        pca_dim = int(min(self.output_dim - lda_dim, x_fit.shape[-1]))
        pca_dim = max(1, pca_dim)
        tprint(
            "Fitting kernel LDA spectral reducer with:\n"
            f"\tKernel PCA dim = {pca_dim}\n"
            f"\tKernel LDA dim = {lda_dim}\n"
        )

        self.pca_dim = pca_dim
        self.pca = KernelPCA(input_dim=x_fit.shape[-1], output_dim=self.pca_dim)
        x_pca = self.pca.transform(x_fit).astype(np.float32)

        try:
            lda_dim = int(min(lda_dim, x_pca.shape[-1]))
            self.lda = LinearDiscriminantAnalysis(n_components=lda_dim)
            self.lda.fit(x_pca, y_fit)
            self.lda_dim = lda_dim
        except Exception as exc:
            tprint(f"Warning: Kernel LDA fit failed, fallback to Kernel PCA only. reason={exc}")
            self.lda = None
            self.lda_dim = 0

        self.final_dim = self.pca_dim + self.lda_dim if self.lda is not None else self.pca_dim
        self.reference_projection_name = "Kernel PCA"
        self.reduced_projection_name = "Kernel PCA + LDA" if self.lda is not None else "Kernel PCA"
        self.fitted = True

    def _do_pca_nca(self, x_fit: np.ndarray, y_fit: np.ndarray) -> bool:
        base_dim = self.pca_dim_override if self.pca_dim_override > 0 else max(self.output_dim * 4, self.output_dim + 8)
        pca_dim = int(min(max(self.output_dim, base_dim), x_fit.shape[-1], max(1, x_fit.shape[0] - 1)))
        tprint(
            "Fitting PCA + NCA spectral reducer with:\n"
            f"\tSNV={self.enable_snv}, derivative_order={self.derivative_order}, standardize={self.enable_standardize}\n"
            f"\tPCA dim = {pca_dim}\n"
            f"\tNCA dim = {self.output_dim}\n"
            f"\tNCA fit budget = {self.nca_max_fit_pixels}\n"
        )

        x_pca = self._fit_pca(x_fit, target_dim=pca_dim)
        x_nca_fit, y_nca_fit = self._balanced_subsample(x_pca, y_fit, budget=self.nca_max_fit_pixels)

        try:
            nca_dim = int(min(self.output_dim, x_pca.shape[-1]))
            self.nca = NeighborhoodComponentsAnalysis(
                n_components=nca_dim,
                init=self.nca_init,
                max_iter=self.nca_max_iter,
                tol=self.nca_tol,
                random_state=self.seed,
            )
            self.nca.fit(x_nca_fit, y_nca_fit)
            self.final_dim = nca_dim
            self.reduced_projection_name = "PCA + NCA"
        except Exception as exc:
            tprint(f"Warning: NCA fit failed, fallback to PCA only. reason={exc}")
            self.nca = None
            self.final_dim = min(self.output_dim, x_pca.shape[-1])
            self.reduced_projection_name = "PCA"

        self.reference_projection_name = "PCA"
        self.lda = None
        self.lda_dim = 0
        self.fitted = True

    def project_pixels(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """project flat spectral vectors for visualization or debugging."""
        if self.mode == "none":
            x_arr = np.asarray(x, dtype=np.float32)
            return x_arr, x_arr
        if not self.fitted:
            raise RuntimeError("spectral reducer must be fitted before projection")

        feats = self._prepare_features(x, fit=False)
        if self.pca is None:
            return feats, feats

        x_ref = self.pca.transform(feats).astype(np.float32)

        if self.mode == "pca_nca":
            if self.nca is None:
                return x_ref, x_ref[:, : min(self.output_dim, x_ref.shape[1])]
            return x_ref, self.nca.transform(x_ref).astype(np.float32)

        if self.lda is not None:
            return x_ref, self.lda.transform(x_ref).astype(np.float32)

        return x_ref, x_ref[:, : min(self.output_dim, x_ref.shape[1])]

    def transform(self, image: np.ndarray) -> np.ndarray:
        """transform image channels.

        input:
            image: (h, w, c).
        output:
            transformed image: (c, h, w) for reducer modes,
        """
        if self.mode == "none":
            if self._none_identity:
                h, w, c = image.shape
                return np.asarray(image, dtype=np.float32).transpose(2, 0, 1)
            # do simple norm for none-mode
            h, w, c = image.shape
            x = image.reshape(-1, c).astype(np.float32)
            x = self._prepare_features(x, fit=False)
            # expected output: (c, h, w)
            return x.reshape(h, w, c).transpose(2, 0, 1).astype(np.float32)
        if not self.fitted:
            raise RuntimeError("spectral reducer must be fitted before transform")

        h, w, c = image.shape
        x = image.reshape(-1, c)
        x_ref, x_red = self.project_pixels(x)

        if self.mode in {"lda_pca", "kernel_lda"} and self.lda is not None:
            y = np.concatenate([x_ref, x_red], axis=1).astype(np.float32)
        else:
            y = np.asarray(x_red, dtype=np.float32)

        if y.shape[1] < self.output_dim:
            pad = np.zeros((y.shape[0], self.output_dim - y.shape[1]), dtype=np.float32)
            y = np.concatenate([y, pad], axis=1)
        if y.shape[1] > self.output_dim:
            y = y[:, : self.output_dim]

        return y.reshape(h, w, self.output_dim).transpose(2, 0, 1)

    def _collect_pixels(
        self,
        samples: Sequence[SampleItem],
        num_classes: int,
        show_progress: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """collect class-balanced train pixels for fitting."""
        rng = np.random.default_rng(self.seed)
        per_class_budget = max(256, self.max_fit_pixels // max(1, num_classes))
        x_by_class: List[List[np.ndarray]] = [[] for _ in range(num_classes)]

        sample_iter = tqdm(
            samples,
            total=len(samples),
            desc="reducer-fit",
            dynamic_ncols=True,
            leave=False,
            disable=not show_progress,
            mininterval=0.2,
        )

        for sample in sample_iter:
            cube = np.load(sample.hsi_path, mmap_mode="r")
            mask = np.load(sample.mask_path).astype(np.int64)
            cube, mask = _pad_to_patch(cube, mask, patch_size=1)

            h, w, c = cube.shape
            flat = cube.reshape(-1, c)
            y = mask.reshape(-1)

            for cls_idx in range(num_classes):
                idx = np.where(y == cls_idx)[0]
                if idx.size == 0:
                    continue
                take = min(idx.size, max(32, per_class_budget // max(1, len(samples))))
                chosen = rng.choice(idx, size=take, replace=False)
                x_by_class[cls_idx].append(np.asarray(flat[chosen], dtype=np.float32))

        sample_iter.close()

        x_parts: List[np.ndarray] = []
        y_parts: List[np.ndarray] = []
        for cls_idx in range(num_classes):
            if not x_by_class[cls_idx]:
                continue
            cls_x = np.concatenate(x_by_class[cls_idx], axis=0)
            if cls_x.shape[0] > per_class_budget:
                chosen = rng.choice(cls_x.shape[0], size=per_class_budget, replace=False)
                cls_x = cls_x[chosen]
            x_parts.append(cls_x)
            y_parts.append(np.full(cls_x.shape[0], cls_idx, dtype=np.int64))

        if not x_parts:
            return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)

        x = np.concatenate(x_parts, axis=0).astype(np.float32)
        y = np.concatenate(y_parts, axis=0)

        if x.shape[0] > self.max_fit_pixels:
            chosen = rng.choice(x.shape[0], size=self.max_fit_pixels, replace=False)
            x = x[chosen]
            y = y[chosen]
        return x, y


def _discover_hsi_samples(config: Munch) -> List[SampleItem]:
    """discover paired hsi and mask files."""
    hsi_dir = Path(config.path.hsi_dir)
    label_dir = Path(config.path.label_dir)
    rgb_dir = Path(config.path.rgb_dir)
    return _discover_hsi_samples_from_dirs(hsi_dir=hsi_dir, label_dir=label_dir, rgb_dir=rgb_dir)


def _require_config_path(cfg: Munch, key: str) -> Path:
    """Read required path from config and fail loudly when missing/empty."""
    try:
        value = getattr(cfg.path, key)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"missing required config path: path.{key}") from exc
    text = str(value).strip()
    if not text:
        raise RuntimeError(f"empty required config path: path.{key}")
    return Path(text).resolve()


def _discover_hsi_samples_from_dirs(hsi_dir: Path, label_dir: Path, rgb_dir: Path) -> List[SampleItem]:
    """discover paired hsi and mask files from explicit directories."""

    if not hsi_dir.exists():
        raise FileNotFoundError(f"hsi_dir not found: {hsi_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"label_dir not found: {label_dir}")

    gt_files = sorted(label_dir.glob("*_gt.npy"))
    samples: List[SampleItem] = []

    for gt_path in gt_files:
        sample_id = gt_path.name[: -len("_gt.npy")]
        hsi_path = hsi_dir / f"{sample_id}.npy"
        if not hsi_path.exists():
            continue

        rgb_path = rgb_dir / f"{sample_id}_Merged_rgb.png"
        rgb_value = rgb_path if rgb_path.exists() else None

        patient_id = sample_id.split("_")[0]
        samples.append(
            SampleItem(
                sample_id=sample_id,
                patient_id=patient_id,
                hsi_path=hsi_path,
                mask_path=gt_path,
                rgb_path=rgb_value,
            )
        )

    if not samples:
        raise RuntimeError("no sample pairs found in configured directories")

    return samples


def _prepare_diffed_hsi_to_dir(
    config: Munch,
    raw_samples: Sequence[SampleItem],
    show_progress: bool,
) -> Dict[str, Any]:
    """
    Build debiased HSI dataset in ``path.diffed_dir`` from raw ``path.hsi_dir``.
    Fixed processing order:
      1) patient-level z-score normalization
      2) Savitzky-Golay smoothing
      3) first-order spectral difference
    No in-place overwrite of raw files, no hash mapping.
    """
    out_bands = int(config.data.hsi_bands)
    raw_hsi_dir = _require_config_path(config, "hsi_dir")
    diffed_dir = _require_config_path(config, "diffed_dir")
    if raw_hsi_dir == diffed_dir:
        raise RuntimeError(
            "strict safety mode: path.hsi_dir and path.diffed_dir must be different "
            f"(both resolve to: {raw_hsi_dir})"
        )
    diffed_dir.mkdir(parents=True, exist_ok=True)

    sentinel_path = diffed_dir / ".stagea_meta.json"
    n_written = 0
    n_labels_copied = 0
    pre_cfg = _as_munch(getattr(config.data, "preprocess", Munch()))
    smooth_window = int(getattr(pre_cfg, "savgol_window", 11))
    smooth_polyorder = int(getattr(pre_cfg, "savgol_polyorder", 3))
    patient_stats = _collect_patient_zscore_stats(raw_samples, out_bands, show_progress=show_progress)
    iterator = tqdm(
        list(raw_samples),
        desc="zscore+savgol+diff->diffed_dir",
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress,
    )
    for sample in iterator:
        cube = np.load(sample.hsi_path).astype(np.float32)
        if cube.ndim != 3:
            raise ValueError(f"hsi must be HWC for {sample.sample_id}, got {cube.shape}")
        if int(cube.shape[2]) != out_bands:
            raise ValueError(
                f"{sample.sample_id}: hsi has {cube.shape[2]} bands; expected raw width=data.hsi_bands={out_bands}"
            )
        pid = str(sample.patient_id)
        if pid not in patient_stats:
            raise RuntimeError(f"missing patient z-score stats for patient_id={pid!r}")
        mean, std = patient_stats[pid]
        out = _hsi_patient_zscore_savgol_diff_array(
            cube,
            patient_mean=mean,
            patient_std=std,
            smooth_window=smooth_window,
            smooth_polyorder=smooth_polyorder,
        )
        out_hsi = diffed_dir / f"{sample.sample_id}.npy"
        tmp_hsi = diffed_dir / f"{sample.sample_id}_tmp_diffed.npy"
        np.save(tmp_hsi, out.astype(np.float32))
        os.replace(tmp_hsi, out_hsi)
        n_written += 1

        out_gt = diffed_dir / f"{sample.sample_id}_gt.npy"
        if sample.mask_path.exists():
            shutil.copy2(sample.mask_path, out_gt)
            n_labels_copied += 1
    iterator.close()
    sentinel_payload = {
        "stage": "stage_a_patient_zscore_savgol_diff",
        "mode": "generated_from_raw",
        "source_hsi_dir": str(raw_hsi_dir),
        "source_label_dir": str(Path(str(config.path.label_dir)).resolve()),
        "diffed_dir": str(diffed_dir),
        "num_hsi_written": int(n_written),
        "num_labels_copied": int(n_labels_copied),
        "pipeline": [
            "patient_level_zscore",
            f"savgol(window={smooth_window},polyorder={smooth_polyorder})",
            "first_order_difference",
            "zero_pad_to_same_bands",
        ],
        "output_bands": int(out_bands),
    }
    with sentinel_path.open("w", encoding="utf-8") as f:
        json.dump(sentinel_payload, f, ensure_ascii=False, indent=2)
    return {
        "enabled": True,
        "mode": "generated_from_raw",
        "source_hsi_dir": str(raw_hsi_dir),
        "source_label_dir": str(config.path.label_dir),
        "diffed_dir": str(diffed_dir),
        "num_hsi_written": int(n_written),
        "num_labels_copied": int(n_labels_copied),
        "pipeline": [
            "patient_level_zscore",
            f"savgol(window={smooth_window},polyorder={smooth_polyorder})",
            "first_order_difference",
            "zero_pad_to_same_bands",
        ],
        "output_bands": int(out_bands),
        "sentinel_path": str(sentinel_path),
    }


def _apply_classwise_patient_mean_align(
    diffed_dir: Path,
    samples: Sequence[SampleItem],
    num_classes: int,
    show_progress: bool,
) -> Dict[str, Any]:
    """
    Supervised class-wise patient mean alignment on already diffed cubes.
    For each foreground class c:
      x <- x - mean_patient(c) + mean_global(c)
    """
    sums_global: Dict[int, np.ndarray] = {}
    counts_global: Dict[int, int] = {}
    sums_patient: Dict[Tuple[str, int], np.ndarray] = {}
    counts_patient: Dict[Tuple[str, int], int] = {}
    n_bands: Optional[int] = None

    iter1 = tqdm(list(samples), desc="classwise-align pass1", dynamic_ncols=True, leave=False, disable=not show_progress)
    for s in iter1:
        hsi_path = diffed_dir / f"{s.sample_id}.npy"
        gt_path = diffed_dir / f"{s.sample_id}_gt.npy"
        if not hsi_path.exists() or not gt_path.exists():
            continue
        cube = np.load(hsi_path).astype(np.float32)
        mask = np.load(gt_path).astype(np.int64)
        if cube.ndim != 3 or mask.ndim != 2:
            continue
        if cube.shape[:2] != mask.shape and cube.shape[1:] == mask.shape:
            cube = np.moveaxis(cube, 0, -1)
        if cube.shape[:2] != mask.shape:
            continue
        if n_bands is None:
            n_bands = int(cube.shape[-1])
        pid = str(s.patient_id)
        for cls_idx in range(1, int(num_classes)):
            m = (mask == cls_idx)
            if not np.any(m):
                continue
            rows = cube[m]
            k = (pid, cls_idx)
            sums_patient[k] = rows.sum(axis=0) if k not in sums_patient else (sums_patient[k] + rows.sum(axis=0))
            counts_patient[k] = int(counts_patient.get(k, 0) + rows.shape[0])
            sums_global[cls_idx] = rows.sum(axis=0) if cls_idx not in sums_global else (sums_global[cls_idx] + rows.sum(axis=0))
            counts_global[cls_idx] = int(counts_global.get(cls_idx, 0) + rows.shape[0])
    iter1.close()

    if n_bands is None:
        return {"classwise_align": True, "classwise_align_applied": False, "classwise_align_reason": "no_valid_rows"}

    mean_global: Dict[int, np.ndarray] = {}
    mean_patient: Dict[Tuple[str, int], np.ndarray] = {}
    for cls_idx, ssum in sums_global.items():
        cnt = max(1, int(counts_global.get(cls_idx, 0)))
        mean_global[cls_idx] = (ssum / float(cnt)).astype(np.float32)
    for k, ssum in sums_patient.items():
        cnt = max(1, int(counts_patient.get(k, 0)))
        mean_patient[k] = (ssum / float(cnt)).astype(np.float32)

    aligned_samples = 0
    iter2 = tqdm(list(samples), desc="classwise-align pass2", dynamic_ncols=True, leave=False, disable=not show_progress)
    for s in iter2:
        hsi_path = diffed_dir / f"{s.sample_id}.npy"
        gt_path = diffed_dir / f"{s.sample_id}_gt.npy"
        if not hsi_path.exists() or not gt_path.exists():
            continue
        cube = np.load(hsi_path).astype(np.float32)
        mask = np.load(gt_path).astype(np.int64)
        moved = False
        if cube.ndim != 3 or mask.ndim != 2:
            continue
        if cube.shape[:2] != mask.shape and cube.shape[1:] == mask.shape:
            cube = np.moveaxis(cube, 0, -1)
            moved = True
        if cube.shape[:2] != mask.shape:
            continue
        pid = str(s.patient_id)
        changed = False
        for cls_idx in range(1, int(num_classes)):
            k = (pid, cls_idx)
            if cls_idx not in mean_global or k not in mean_patient:
                continue
            m = (mask == cls_idx)
            if not np.any(m):
                continue
            delta = (mean_global[cls_idx] - mean_patient[k]).astype(np.float32)
            cube[m] = cube[m] + delta
            changed = True
        if changed:
            out_arr = np.moveaxis(cube, -1, 0) if moved else cube
            tmp = diffed_dir / f"{s.sample_id}_tmp_classalign.npy"
            np.save(tmp, out_arr.astype(np.float32))
            os.replace(tmp, hsi_path)
            aligned_samples += 1
    iter2.close()
    return {
        "classwise_align": True,
        "classwise_align_applied": bool(aligned_samples > 0),
        "classwise_align_aligned_samples": int(aligned_samples),
    }


def _diffed_dataset_ready(diffed_dir: Path) -> Tuple[bool, int]:
    """
    Check whether ``diffed_dir`` already contains valid processed pairs.
    A pair is counted when both ``<sample>.npy`` and ``<sample>_gt.npy`` exist.
    """
    if not diffed_dir.exists() or not diffed_dir.is_dir():
        return False, 0
    sentinel_path = diffed_dir / ".stagea_meta.json"
    if not sentinel_path.exists():
        return False, 0
    pair_count = 0
    for gt_path in diffed_dir.glob("*_gt.npy"):
        sample_id = gt_path.name[:-7]
        if (diffed_dir / f"{sample_id}.npy").exists():
            pair_count += 1
    return pair_count > 0, int(pair_count)


def _collect_patient_zscore_stats(
    samples: Sequence[SampleItem],
    raw_bands: int,
    show_progress: bool,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute patient-level per-band mean/std on raw spectra.
    Stats are aggregated over all pixels from all samples of each patient.
    """
    sums: Dict[str, np.ndarray] = {}
    sums_sq: Dict[str, np.ndarray] = {}
    counts: Dict[str, int] = {}

    iterator = tqdm(
        list(samples),
        desc="patient-zscore-stats",
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress,
    )
    for sample in iterator:
        cube = np.load(sample.hsi_path).astype(np.float32)
        if cube.ndim != 3 or int(cube.shape[2]) != int(raw_bands):
            raise ValueError(
                f"{sample.sample_id}: expected HWC with {raw_bands} bands for patient stats, got {tuple(cube.shape)}"
            )
        rows = cube.reshape(-1, int(raw_bands)).astype(np.float64)
        pid = str(sample.patient_id)
        band_sum = rows.sum(axis=0)
        band_sq_sum = np.square(rows).sum(axis=0)
        if pid not in sums:
            sums[pid] = band_sum
            sums_sq[pid] = band_sq_sum
            counts[pid] = int(rows.shape[0])
        else:
            sums[pid] += band_sum
            sums_sq[pid] += band_sq_sum
            counts[pid] = int(counts[pid] + rows.shape[0])
    iterator.close()

    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for pid, band_sum in sums.items():
        n = max(1, int(counts.get(pid, 0)))
        mean = (band_sum / float(n)).astype(np.float32)
        var = (sums_sq[pid] / float(n)) - np.square(mean.astype(np.float64))
        std = np.sqrt(np.maximum(var, 1e-12)).astype(np.float32)
        out[pid] = (mean, std)
    return out


def _hsi_patient_zscore_savgol_diff_array(
    cube: np.ndarray,
    patient_mean: np.ndarray,
    patient_std: np.ndarray,
    smooth_window: int = 11,
    smooth_polyorder: int = 3,
) -> np.ndarray:
    """
    Fixed debias chain:
      patient-level z-score -> Savitzky-Golay smoothing -> first-order difference.
    Then pad one zero band to keep band count unchanged.
    """
    if cube.ndim != 3:
        raise ValueError(f"expected HWC cube, got shape={tuple(cube.shape)}")
    raw_bands = int(cube.shape[2])
    if patient_mean.shape[0] != raw_bands or patient_std.shape[0] != raw_bands:
        raise ValueError(
            f"patient stats mismatch: expected {raw_bands} bands, got mean={patient_mean.shape[0]}, std={patient_std.shape[0]}"
        )
    x = np.asarray(cube, dtype=np.float64)
    x = (x - patient_mean.reshape(1, 1, -1)) / np.maximum(patient_std.reshape(1, 1, -1), 1e-6)

    win = int(max(3, smooth_window))
    if win % 2 == 0:
        win += 1
    if win > int(raw_bands):
        win = int(raw_bands) if int(raw_bands) % 2 == 1 else int(raw_bands) - 1
    poly = int(max(1, smooth_polyorder))
    if win >= 3 and poly < win:
        x = savgol_filter(x, window_length=win, polyorder=poly, axis=2, mode="interp")
    d = np.diff(x, axis=2, n=1)
    z = np.zeros_like(d[..., :1])
    out = np.concatenate([d, z], axis=2)
    if int(out.shape[2]) != raw_bands:
        raise RuntimeError(
            f"internal error: expected {raw_bands} bands after zero pad, got {out.shape[2]}"
        )
    return out.astype(np.float32)


def _hsi_diff_pad_snv_array(
    cube: np.ndarray,
    hsi_diff_order: int,
    snv: bool,
    snv_eps: float,
    pad_mode: str,
    raw_bands: int,
    smooth_window: int = 7,
    smooth_polyorder: int = 2,
) -> np.ndarray:
    """
    Legacy helper kept for compatibility with non--d code paths.
    Applies Savitzky-Golay smoothing + spectral diff, optional pad, optional per-pixel SNV.
    """
    if int(cube.shape[2]) != int(raw_bands):
        raise ValueError(f"expected {raw_bands} spectral bands, got {cube.shape[2]}")
    order = max(1, int(hsi_diff_order))
    x = np.asarray(cube, dtype=np.float64)
    win = int(max(3, smooth_window))
    if win % 2 == 0:
        win += 1
    if win > int(raw_bands):
        win = int(raw_bands) if int(raw_bands) % 2 == 1 else int(raw_bands) - 1
    poly = int(max(1, smooth_polyorder))
    if win >= 3 and poly < win:
        x = savgol_filter(x, window_length=win, polyorder=poly, axis=2, mode="interp")
    d = np.diff(x, axis=2, n=order)
    need = int(raw_bands) - int(d.shape[2])
    if need < 0:
        raise ValueError("hsi_diff_order too large for hsi_bands")
    pad_mode = str(pad_mode)
    if need > 0:
        if pad_mode == "repeat_last":
            tail = d[..., -1:]
            d = np.concatenate([d] + [tail] * need, axis=2)
        elif pad_mode in ("zero_pad", "zero"):
            z = np.zeros_like(d[..., :1])
            d = np.concatenate([d] + [z] * need, axis=2)
        else:
            raise ValueError(f"unknown hsi_pad_mode={pad_mode!r}; use repeat_last or zero_pad")
    out = d.astype(np.float32)
    if snv:
        h, w, bands = out.shape
        flat = out.reshape(-1, bands)
        mean = flat.mean(axis=1, keepdims=True)
        std = flat.std(axis=1, keepdims=True)
        flat = (flat - mean) / np.maximum(std, float(snv_eps))
        out = flat.reshape(h, w, bands).astype(np.float32)
    if int(out.shape[2]) != int(raw_bands):
        raise RuntimeError(f"internal error: expected {raw_bands} bands after pad, got {out.shape[2]}")
    return out


def _file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA256 of file bytes (stable across runs; used to detect already-aligned HSI)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _alignment_stat_sig(path: Path) -> Tuple[int, int]:
    st = path.stat()
    return (int(st.st_mtime_ns), int(st.st_size))


def _spectral_alignment_cfg(config: Munch) -> Munch:
    """spec §4: safe access to data.spectral_alignment."""
    return _as_munch(getattr(config.data, "spectral_alignment", Munch()))


def _hsi_spectral_diff_manifest_path(config: Munch) -> Path:
    al = _spectral_alignment_cfg(config)
    name = str(getattr(al, "manifest_basename_hsi", ".hsi_spectral_diff_manifest.json"))
    return Path(config.path.hsi_dir) / name


def _rgb_spectral_diff_manifest_path(config: Munch) -> Path:
    al = _spectral_alignment_cfg(config)
    name = str(getattr(al, "manifest_basename_rgb", ".rgb_spectral_diff_manifest.json"))
    return Path(config.path.rgb_dir) / name


def _spectral_alignment_key_digest(config_key: Mapping[str, Any]) -> str:
    raw = json.dumps(config_key, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def ensure_spectral_alignment_on_disk(
    config: Munch,
    samples: Sequence[SampleItem],
    show_progress: bool,
) -> Dict[str, Any]:
    """
    In-place HSI: raw cubes have ``data.hsi_bands`` channels (e.g. 96). Apply spectral diff + pad
    back to the same width + optional per-pixel SNV. Idempotency: manifest v2 stores per-file
    SHA256 of aligned bytes; matching hashes skip rewrites.
    """
    al = _spectral_alignment_cfg(config)
    if not bool(getattr(al, "enabled", False)):
        return {
            "skipped": True,
            "enabled": False,
            "reason": "spectral_alignment.disabled",
            "hsi_manifest": None,
            "rgb_manifest": None,
            "alignment_key": None,
            "snv": bool(getattr(al, "snv", False)),
            "snv_eps": float(getattr(al, "snv_eps", 1e-6)),
            "rgb_snv": bool(getattr(al, "rgb_snv", False)),
            "num_hsi_written": 0,
            "num_rgb_written": 0,
            "num_hsi_skipped_unchanged": 0,
            "manifest_format": None,
        }

    out_bands = int(config.data.hsi_bands)
    hsi_diff_order = max(1, int(getattr(al, "hsi_diff_order", 1)))
    snv = bool(getattr(al, "snv", True))
    snv_eps = float(getattr(al, "snv_eps", 1e-6))
    version_tag = str(getattr(al, "version_tag", "v1"))
    process_rgb = bool(getattr(al, "process_rgb", False))
    rgb_pad_mode = str(getattr(al, "rgb_pad_mode", "repeat_last"))
    rgb_snv = bool(getattr(al, "rgb_snv", False))
    hsi_pad_mode = str(getattr(al, "hsi_pad_mode", "repeat_last"))
    hsi_backup_dir = Path(str(getattr(al, "hsi_raw_backup_dir", Path(config.path.hsi_dir) / ".stageA_raw_backup")))

    manifest_path = _hsi_spectral_diff_manifest_path(config)
    rgb_manifest_path = _rgb_spectral_diff_manifest_path(config)

    config_key = {
        "version_tag": version_tag,
        "hsi_spectral_layout": "diff_pad_snv_same_dim_v1",
        "hsi_diff_order": hsi_diff_order,
        "hsi_pad_mode": hsi_pad_mode,
        "snv": snv,
        "snv_eps": snv_eps,
        "hsi_bands": out_bands,
        "rgb_pad_mode": rgb_pad_mode,
        "rgb_snv": rgb_snv,
        "process_rgb": process_rgb,
    }
    alignment_key = _spectral_alignment_key_digest(config_key)

    existing: Dict[str, Any] = {}
    if manifest_path.exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = {}

    per_sample_prev: Dict[str, Any] = {}
    if (
        existing.get("config_key") == config_key
        and existing.get("alignment_key") == alignment_key
        and existing.get("manifest_format") == "v2"
    ):
        per_sample_prev = existing.get("per_sample", {}) or {}

    fast_ok = (
        existing.get("config_key") == config_key
        and existing.get("alignment_key") == alignment_key
        and existing.get("manifest_format") == "v2"
    )
    if fast_ok:
        for sample in samples:
            try:
                cube = np.load(sample.hsi_path, mmap_mode="r")
                if int(cube.shape[2]) != out_bands:
                    fast_ok = False
                    break
                prev = per_sample_prev.get(sample.sample_id)
                if not prev or not prev.get("hsi_aligned_sha256"):
                    fast_ok = False
                    break
                if _file_sha256(sample.hsi_path) != str(prev["hsi_aligned_sha256"]):
                    fast_ok = False
                    break
            except OSError:
                fast_ok = False
                break
        if fast_ok:
            return {
                "skipped": False,
                "cache_hit": True,
                "enabled": True,
                "alignment_key": alignment_key,
                "config_key": config_key,
                "snv": snv,
                "snv_eps": snv_eps,
                "rgb_snv": rgb_snv,
                "hsi_manifest": str(manifest_path),
                "rgb_manifest": str(rgb_manifest_path) if process_rgb else None,
                "num_hsi_written": 0,
                "num_rgb_written": 0,
                "num_hsi_skipped_unchanged": len(samples),
                "manifest_format": "v2",
                "per_sample": per_sample_prev,
            }

    num_hsi_written = 0
    num_hsi_skipped = 0
    num_rgb_written = 0
    per_sample_out: Dict[str, Any] = {}

    sample_iter = tqdm(
        list(samples),
        desc="spectral-align",
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress,
    )
    for sample in sample_iter:
        cube = np.load(sample.hsi_path, mmap_mode="r")
        if cube.ndim != 3:
            raise ValueError(f"hsi must be HWC for {sample.sample_id}, got {cube.shape}")
        c = int(cube.shape[2])
        if c != out_bands:
            raise ValueError(
                f"{sample.sample_id}: hsi has {c} bands; with spectral_alignment enabled, "
                f"expect raw (or aligned) width == data.hsi_bands ({out_bands}). "
                f"Legacy (out+diff_order) inputs are no longer supported."
            )
        cur_hash = _file_sha256(sample.hsi_path)
        prev = per_sample_prev.get(sample.sample_id, {})
        if prev.get("hsi_aligned_sha256") == cur_hash:
            per_sample_out[sample.sample_id] = {
                "hsi_raw_sha256": prev.get("hsi_raw_sha256"),
                "hsi_aligned_sha256": prev.get("hsi_aligned_sha256"),
                "hsi_raw_backup_path": prev.get("hsi_raw_backup_path"),
            }
            num_hsi_skipped += 1
        else:
            raw_fp = cur_hash
            arr = np.asarray(cube, dtype=np.float32)
            raw_backup_path = hsi_backup_dir / f"{sample.sample_id}__{raw_fp[:16]}.npy"
            if not raw_backup_path.exists():
                raw_backup_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(raw_backup_path, arr.astype(np.float32))
            out = _hsi_diff_pad_snv_array(
                arr, hsi_diff_order, snv, snv_eps, hsi_pad_mode, raw_bands=out_bands
            )
            # np.save appends .npy when suffix is missing/unknown, so temp path must end with .npy.
            tmp = sample.hsi_path.with_name(f"{sample.hsi_path.stem}_spectral_align_tmp.npy")
            np.save(tmp, out.astype(np.float32))
            os.replace(tmp, sample.hsi_path)
            num_hsi_written += 1
            aligned_fp = _file_sha256(sample.hsi_path)
            per_sample_out[sample.sample_id] = {
                "hsi_raw_sha256": raw_fp,
                "hsi_aligned_sha256": aligned_fp,
                "hsi_raw_backup_path": str(raw_backup_path),
            }

        if process_rgb and sample.rgb_path is not None and sample.rgb_path.exists():
            rgb_bgr = cv2.imread(str(sample.rgb_path), cv2.IMREAD_COLOR)
            if rgb_bgr is None:
                continue
            rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            d = np.diff(rgb.astype(np.float64), axis=2, n=1).astype(np.float32)
            if rgb_pad_mode == "zero_third":
                rgb_out = np.concatenate([d, np.zeros_like(d[..., :1])], axis=2)
            else:
                rgb_out = np.concatenate([d, d[..., -1:]], axis=2)
            if rgb_snv:
                h, w, ch = rgb_out.shape
                flat = rgb_out.reshape(-1, ch)
                mean = flat.mean(axis=1, keepdims=True)
                std = flat.std(axis=1, keepdims=True)
                flat = (flat - mean) / np.maximum(std, snv_eps)
                rgb_out = flat.reshape(h, w, ch)
            rgb_u8 = np.clip(rgb_out * 255.0, 0.0, 255.0).astype(np.uint8)
            rgb_u8_bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
            # cv2.imwrite selects codec by extension; must end in .png (not .png.tmp_align).
            tmp_png = sample.rgb_path.with_name(f"{sample.rgb_path.stem}_spectral_align_tmp.png")
            cv2.imwrite(str(tmp_png), rgb_u8_bgr)
            os.replace(tmp_png, sample.rgb_path)
            num_rgb_written += 1
            per_sample_out[sample.sample_id]["rgb_sig"] = list(_alignment_stat_sig(sample.rgb_path))

    payload = {
        "manifest_format": "v2",
        "alignment_key": alignment_key,
        "config_key": config_key,
        "per_sample": per_sample_out,
        "num_hsi_written": num_hsi_written,
        "num_hsi_skipped_unchanged": num_hsi_skipped,
        "num_rgb_written": num_rgb_written,
    }
    try:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    except OSError as exc:
        tprint(f"Warning: could not write spectral alignment manifest: {exc}")

    if process_rgb:
        try:
            rgb_manifest_path.parent.mkdir(parents=True, exist_ok=True)
            rgb_payload = {
                "manifest_format": "v2",
                "alignment_key": alignment_key,
                "config_key": config_key,
                "per_sample": per_sample_out,
            }
            with rgb_manifest_path.open("w", encoding="utf-8") as f:
                json.dump(rgb_payload, f, indent=2, ensure_ascii=False)
        except OSError as exc:
            tprint(f"Warning: could not write RGB spectral alignment manifest: {exc}")

    return {
        "skipped": False,
        "cache_hit": False,
        "enabled": True,
        "alignment_key": alignment_key,
        "config_key": config_key,
        "snv": snv,
        "snv_eps": snv_eps,
        "rgb_snv": rgb_snv,
        "hsi_manifest": str(manifest_path),
        "rgb_manifest": str(rgb_manifest_path) if process_rgb else None,
        "num_hsi_written": num_hsi_written,
        "num_hsi_skipped_unchanged": num_hsi_skipped,
        "num_rgb_written": num_rgb_written,
        "manifest_format": "v2",
        "per_sample": per_sample_out,
    }


def _split_subjects(samples: Sequence[SampleItem], config: Munch) -> Dict[str, List[SampleItem]]:
    """strict subject-level split into train/eval/test."""
    train_ratio = float(config.data.split.train)
    eval_ratio = float(config.data.split.eval)
    test_ratio = float(config.data.split.test)
    train_ratio, eval_ratio, test_ratio = _safe_ratio_triplet(train_ratio, eval_ratio, test_ratio)

    seed = int(config.data.split.seed)
    rng = np.random.default_rng(seed)

    patients = sorted({s.patient_id for s in samples})
    rng.shuffle(patients)

    n = len(patients)
    n_train = max(1, int(round(n * train_ratio)))
    n_eval = max(1, int(round(n * eval_ratio))) if n >= 3 else max(0, n - n_train - 1)
    n_test = n - n_train - n_eval
    if n_test <= 0:
        n_test = 1
        if n_eval > 1:
            n_eval -= 1
        else:
            n_train = max(1, n_train - 1)

    train_patients = set(patients[:n_train])
    eval_patients = set(patients[n_train : n_train + n_eval])
    test_patients = set(patients[n_train + n_eval :])

    out = {"train": [], "eval": [], "test": []}
    for item in samples:
        if item.patient_id in train_patients:
            out["train"].append(item)
        elif item.patient_id in eval_patients:
            out["eval"].append(item)
        elif item.patient_id in test_patients:
            out["test"].append(item)

    for split in ["train", "eval", "test"]:
        if not out[split]:
            raise RuntimeError(f"empty split '{split}' after subject split")

    return out


def _build_patch_records(
    samples: Sequence[SampleItem],
    patch_size: int,
    stride: int,
    foreground_ratio_threshold: float,
    num_classes: int,
    tracked_sample_ids: Sequence[str],
    split_name: str = "split",
    show_progress: bool = True,
    max_background_ratio: float = 1.0,
    precious_repeat: int = 1,
    precious_class: int = 1,
    advanced_sampling: Optional[Mapping[str, Any]] = None,
) -> Tuple[List[PatchRecord], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """build patch index list and class histograms.
    input:
        samples: list of SampleItem.
        patch_size: int.
        stride: int.
        foreground_ratio_threshold: float.
        num_classes: int.
        tracked_sample_ids: list of sample ids.
        split_name: str.
        show_progress: bool.
        max_background_ratio: float.
        precious_repeat: int.
        precious_class: int.
        advanced_sampling: dict.
    """
    tracked_set = set(tracked_sample_ids)
    records: List[PatchRecord] = []

    raw_hist = np.zeros(num_classes, dtype=np.int64)
    patch_hist = np.zeros(num_classes, dtype=np.int64)
    sampled_hist = np.zeros(num_classes, dtype=np.float64)
    tracked_pre = np.zeros(num_classes, dtype=np.int64)
    tracked_post = np.zeros(num_classes, dtype=np.int64)

    precious_repeat = max(1, precious_repeat)

    sample_iter = tqdm(
        enumerate(samples),
        total=len(samples),
        desc=f"patch-{split_name}",
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress,
        mininterval=0.2,
    )

    adv = _as_munch(advanced_sampling or {})
    use_weighted_sampler = bool(getattr(adv, "use_weighted_sampler", False)) and split_name == "train"
    rho = float(getattr(adv, "rho", 0.35))
    pg_min_ratio = float(getattr(adv, "pg_min_ratio", 0.12))
    hard_bg_ratio = float(getattr(adv, "hard_bg_ratio", 0.20))
    weight_caps = getattr(adv, "weight_caps", [0.5, 8.0])
    w_min = float(weight_caps[0]) if isinstance(weight_caps, (list, tuple)) and len(weight_caps) > 0 else 0.5
    w_max = float(weight_caps[1]) if isinstance(weight_caps, (list, tuple)) and len(weight_caps) > 1 else 8.0

    for sample_index, sample in sample_iter:
        mask = np.load(sample.mask_path).astype(np.int64)
        raw_hist += _compute_hist(mask, num_classes)

        # Patch indexing only needs spatial shape; avoid extra HSI file reads here.
        h, w = mask.shape

        if h < patch_size or w < patch_size:
            pad_h = max(0, patch_size - h)
            pad_w = max(0, patch_size - w)
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
            h, w = mask.shape

        sample_kept = 0
        for top, left in _iter_patch_positions(h, w, patch_size, stride):
            patch_mask = mask[top : top + patch_size, left : left + patch_size]
            patch_hist_once = _compute_hist(patch_mask, num_classes)
            has_precious = bool(np.any(patch_mask == precious_class))
            fg_ratio = float(np.mean(patch_mask > 0))
            bg_ratio = float(np.mean(patch_mask == 0))
            is_hard_bg = bool(bg_ratio >= 0.85 and fg_ratio > 0.0 and fg_ratio <= hard_bg_ratio)
            if sample.sample_id in tracked_set:
                tracked_pre += patch_hist_once

            if keep := _should_keep_patch(
                patch_mask,
                foreground_ratio_threshold,
                max_background_ratio=max_background_ratio,
                precious_class=precious_class,
            ):
                repeat_count = precious_repeat if has_precious else 1
                weight = 1.0
                if use_weighted_sampler:
                    # mix original and balanced targets while enforcing pg emphasis
                    pg_bonus = max(pg_min_ratio, rho) if has_precious else 0.0
                    weight = 1.0 + 2.5 * pg_bonus + rho * fg_ratio + 0.8 * float(is_hard_bg)
                    weight = float(np.clip(weight, w_min, w_max))
                for _ in range(repeat_count):
                    records.append(
                        PatchRecord(
                            sample_index=sample_index,
                            top=top,
                            left=left,
                            weight=weight,
                            has_precious=has_precious,
                            fg_ratio=fg_ratio,
                            bg_ratio=bg_ratio,
                            is_hard_bg=is_hard_bg,
                        )
                    )
                    patch_hist += patch_hist_once
                    sampled_hist += patch_hist_once.astype(np.float64) * float(weight)
                    sample_kept += 1
                if sample.sample_id in tracked_set:
                    tracked_post += patch_hist_once

        if sample_kept == 0:
            center_top = max(0, (h - patch_size) // 2)
            center_left = max(0, (w - patch_size) // 2)
            patch_mask = mask[center_top : center_top + patch_size, center_left : center_left + patch_size]
            patch_hist_once = _compute_hist(patch_mask, num_classes)
            has_precious = bool(np.any(patch_mask == precious_class))
            repeat_count = precious_repeat if has_precious else 1
            fg_ratio = float(np.mean(patch_mask > 0))
            bg_ratio = float(np.mean(patch_mask == 0))
            is_hard_bg = bool(bg_ratio >= 0.85 and fg_ratio > 0.0 and fg_ratio <= hard_bg_ratio)
            weight = 1.0
            if use_weighted_sampler:
                pg_bonus = max(pg_min_ratio, rho) if has_precious else 0.0
                weight = 1.0 + 2.5 * pg_bonus + rho * fg_ratio + 0.8 * float(is_hard_bg)
                weight = float(np.clip(weight, w_min, w_max))
            for _ in range(repeat_count):
                records.append(
                    PatchRecord(
                        sample_index=sample_index,
                        top=center_top,
                        left=center_left,
                        weight=weight,
                        has_precious=has_precious,
                        fg_ratio=fg_ratio,
                        bg_ratio=bg_ratio,
                        is_hard_bg=is_hard_bg,
                    )
                )
                patch_hist += patch_hist_once
                sampled_hist += patch_hist_once.astype(np.float64) * float(weight)
            if sample.sample_id in tracked_set:
                tracked_post += patch_hist_once

        sample_iter.set_postfix(kept=len(records), refresh=False)

    sample_iter.close()

    return records, raw_hist, patch_hist, tracked_pre, tracked_post, sampled_hist.astype(np.float64)


def _pick_tracked_samples(samples: Sequence[SampleItem], ratio: float, seed: int) -> List[str]:
    """sample ids used for before/after tracking report."""
    if not samples:
        return []
    n = max(1, int(round(len(samples) * ratio)))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(samples), size=min(n, len(samples)), replace=False)
    return [samples[i].sample_id for i in chosen.tolist()]


_PREPARED_CACHE: Dict[str, PreparedData] = {}


def _build_cache_key(config: Munch, modality: str) -> str:
    """build deterministic cache key for prepared data."""
    sampling_cfg = getattr(config.data, "sampling", Munch())
    payload = {
        "modality": modality,
        "paths": {
            "hsi": str(config.path.hsi_dir),
            "label": str(config.path.label_dir),
            "rgb": str(config.path.rgb_dir),
            "diffed": str(config.path.diffed_dir),
        },
        "diff_mode": "strict_read_diffed",
        "split": dict(config.data.split),
        "patch_size": int(config.data.patch_size),
        "stride": int(config.data.stride),
        "threshold": float(config.data.foreground_ratio_threshold),
        "sampling": {
            "train_fg_ratio_threshold": float(
                getattr(sampling_cfg, "train_fg_ratio_threshold", float(config.data.foreground_ratio_threshold))
            ),
            "train_max_background_ratio": float(
                getattr(sampling_cfg, "train_max_background_ratio", 1.0)
            ),
            "train_precious_repeat": int(
                getattr(sampling_cfg, "train_precious_repeat", 1)
            ),
            "precious_class": int(getattr(sampling_cfg, "precious_class", 1)),
        },
        "preprocess": dict(config.data.preprocess),
        "spectral_alignment": _config_section_plain(getattr(config.data, "spectral_alignment", None)),
    }
    return json.dumps(payload, sort_keys=True)


def _cache_signature(samples_by_split: Mapping[str, Sequence[SampleItem]]) -> Dict[str, Any]:
    """build signature for preprocessed samples cache"""
    signature: Dict[str, Any] = {}
    for split_name, split_samples in samples_by_split.items():
        rows = []
        for sample in split_samples:
            st = sample.hsi_path.stat()
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "hsi_path": str(sample.hsi_path),
                    "mtime_ns": int(st.st_mtime_ns),
                    "size": int(st.st_size),
                }
            )
        signature[split_name] = rows
    return signature


def build_diffed_dataset(config: Munch) -> Dict[str, Any]:
    """CLI helper: generate diffed HSI+labels to ``path.diffed_dir`` from raw directories."""
    cfg = config
    raw_hsi_dir = _require_config_path(cfg, "hsi_dir")
    diffed_dir = _require_config_path(cfg, "diffed_dir")
    if raw_hsi_dir == diffed_dir:
        raise RuntimeError(
            "strict safety mode: path.hsi_dir and path.diffed_dir must be different "
            f"(both resolve to: {raw_hsi_dir})"
        )
    if diffed_dir.exists():
        sentinel = diffed_dir / ".stagea_meta.json"
        if sentinel.exists():
            raise RuntimeError(
                f"diffed_dir already has Stage A sentinel ({sentinel}); refusing to run -d twice."
            )
        if any(diffed_dir.iterdir()):
            raise RuntimeError(
                f"diffed_dir is not empty ({diffed_dir}); treat as polluted and abort. "
                "Please clean diffed_dir before running -d."
            )
    raw_samples = _discover_hsi_samples(cfg)
    show_progress = bool(getattr(cfg.runtime, "progress_bar", True))
    meta = _prepare_diffed_hsi_to_dir(cfg, raw_samples, show_progress=show_progress)
    return meta


def prepare_data(config: Munch, modality: str) -> PreparedData:
    """prepare split, patch records, and reducer once."""
    cfg = config
    modality = modality.lower()
    show_progress = bool(getattr(cfg.runtime, "progress_bar", True))

    tprint(
        "prepare data start:\n"
        f"\tmodality={modality}\n"
        f"\tpaths:\n"
        f"\t  hsi={cfg.path.hsi_dir},\n"
        f"\t  label={cfg.path.label_dir},\n"
        f"\t  rgb={cfg.path.rgb_dir}"
    )

    cache_key = _build_cache_key(cfg, modality)
    if cache_key in _PREPARED_CACHE:
        tprint(f"prepare data cache hit:\n"
               f"\tmodality={modality}")
        return _PREPARED_CACHE[cache_key]

    alignment_meta: Dict[str, Any] = {
        "enabled": True,
        "mode": "read_diffed_only",
        "diffed_dir": None,
    }
    if modality == "hsi":
        raw_hsi_dir = _require_config_path(cfg, "hsi_dir")
        diffed_dir = _require_config_path(cfg, "diffed_dir")
        alignment_meta["diffed_dir"] = str(diffed_dir)
        if raw_hsi_dir == diffed_dir:
            raise RuntimeError(
                "strict safety mode: path.hsi_dir and path.diffed_dir must be different "
                f"(both resolve to: {raw_hsi_dir})"
            )
        has_diffed, ready_pairs = _diffed_dataset_ready(diffed_dir)
        if not has_diffed:
            raise RuntimeError(
                "diffed dataset is not ready: require '.stagea_meta.json' and at least one "
                f"HSI/GT pair in path.diffed_dir={diffed_dir}. Run `python run.py -d` first."
            )
        alignment_meta.update(
            {
                "mode": "read_diffed_only",
                "source_hsi_dir": str(raw_hsi_dir),
                "source_label_dir": str(_require_config_path(cfg, "label_dir")),
                "ready_pairs": int(ready_pairs),
            }
        )
        tprint(
            "spectral preprocess routing (strict):\n"
            f"\tread-only from diffed_dir={diffed_dir}\n"
            f"\tready_pairs={ready_pairs}\n"
            "\tno in-pipeline debiasing; use `python run.py -d` to generate when needed"
        )
        all_samples = _discover_hsi_samples_from_dirs(
            hsi_dir=diffed_dir,
            label_dir=diffed_dir,
            rgb_dir=Path(cfg.path.rgb_dir),
        )
        unique_patients = len({s.patient_id for s in all_samples})
        tprint(
            "discovered effective HSI samples:\n"
            f"\ttotal={len(all_samples)}, unique_patients={unique_patients}"
        )
    else:
        alignment_meta["enabled"] = False
        all_samples = _discover_hsi_samples(cfg)
        unique_patients = len({s.patient_id for s in all_samples})
        tprint(f"discovered samples:\n"
               f"\ttotal={len(all_samples)}, unique_patients={unique_patients}")

    samples_by_split = _split_subjects(all_samples, cfg)
    tprint(
        "subject split:\n"
        f"\ttrain={len(samples_by_split['train'])}\n"
        f"\teval={len(samples_by_split['eval'])}\n"
        f"\ttest={len(samples_by_split['test'])}"
    )

    patch_size = int(cfg.data.patch_size)
    stride = int(cfg.data.stride)
    threshold = float(cfg.data.foreground_ratio_threshold)
    num_classes = int(cfg.data.num_classes)
    seed = int(cfg.data.split.seed)
    track_ratio = float(cfg.data.track_sample_ratio)
    sampling_cfg = getattr(cfg.data, "sampling", Munch())
    advanced_sampling_cfg = _as_munch(getattr(sampling_cfg, "advanced", Munch()))

    train_fg_threshold = float(getattr(sampling_cfg, "train_fg_ratio_threshold", threshold))
    train_max_background_ratio = float(getattr(sampling_cfg, "train_max_background_ratio", 1.0))
    train_max_background_ratio = min(1.0, max(0.0, train_max_background_ratio))
    train_precious_repeat = max(1, int(getattr(sampling_cfg, "train_precious_repeat", 1)))
    precious_class = int(getattr(sampling_cfg, "precious_class", 1))

    spectral_alignment_cfg = _config_section_plain(getattr(cfg.data, "spectral_alignment", None))
    if modality == "hsi":
        # HSI branch always reads from diffed_dir (already diff+SNV), so Stage B should not repeat SNV/derivative.
        spectral_alignment_cfg["enabled"] = True
        spectral_alignment_cfg["snv"] = bool(getattr(_spectral_alignment_cfg(cfg), "snv", True))
    reducer = SpectralReducer(
        mode=str(cfg.data.preprocess.mode),
        output_dim=int(cfg.data.preprocess.output_dim),
        max_fit_pixels=int(cfg.data.preprocess.max_fit_pixels),
        seed=seed,
        preprocess_cfg=dict(cfg.data.preprocess),
        spectral_alignment_cfg=spectral_alignment_cfg,
    )
    if modality == "hsi":
        tprint(
            "spectral reducer:\n"
            f"\tmode={reducer.mode}, output_dim={reducer.output_dim}, max_fit_pixels={reducer.max_fit_pixels}"
        )
        reducer.fit(samples_by_split["train"], num_classes=num_classes, show_progress=show_progress)
        tprint("spectral reducer fit done")
    else:
        reducer.mode = "none"
        reducer.fitted = True
        tprint("spectral reducer skipped for rgb modality")

    patches_by_split: Dict[str, List[PatchRecord]] = {}
    split_stats: Dict[str, Any] = {}

    for split_idx, split_name in enumerate(["train", "eval", "test"]):
        split_samples = samples_by_split[split_name]
        tracked_ids = _pick_tracked_samples(split_samples, ratio=track_ratio, seed=seed + split_idx)

        split_threshold = threshold
        split_max_background_ratio = 1.0
        split_precious_repeat = 1
        if split_name == "train":
            split_threshold = train_fg_threshold
            split_max_background_ratio = train_max_background_ratio
            split_precious_repeat = train_precious_repeat

        records, raw_hist, patch_hist, tracked_pre, tracked_post, sampled_hist = _build_patch_records(
            samples=split_samples,
            patch_size=patch_size,
            stride=stride,
            foreground_ratio_threshold=split_threshold,
            num_classes=num_classes,
            tracked_sample_ids=tracked_ids,
            split_name=split_name,
            show_progress=show_progress,
            max_background_ratio=split_max_background_ratio,
            precious_repeat=split_precious_repeat,
            precious_class=precious_class,
            advanced_sampling=dict(advanced_sampling_cfg),
        )

        tprint(
            f"patch prep [{split_name}]: samples={len(split_samples)}\n"
            f"\tpatches={len(records)}, tracked={len(tracked_ids)}\n"
            f"\tpolicy(fg_thr={split_threshold:.3f}, bg_max={split_max_background_ratio:.3f}, repeat={split_precious_repeat})"
        )

        patches_by_split[split_name] = records
        split_stats[split_name] = {
            "num_samples": len(split_samples),
            "num_patches": len(records),
            "raw_pixel_hist": raw_hist.tolist(),
            "kept_patch_pixel_hist": patch_hist.tolist(),
            "sampled_patch_pixel_hist": sampled_hist.tolist(),
            "tracked_sample_ids": tracked_ids,
            "tracked_pre_hist": tracked_pre.tolist(),
            "tracked_post_hist": tracked_post.tolist(),
            "sampling_policy": {
                "foreground_ratio_threshold": split_threshold,
                "max_background_ratio": float(split_max_background_ratio),
                "precious_repeat": int(split_precious_repeat),
                "precious_class": precious_class,
                "advanced": dict(advanced_sampling_cfg),
            },
            "sampled_pg_patch_ratio": float(np.mean([1.0 if r.has_precious else 0.0 for r in records])) if records else 0.0,
        }

    prepared = PreparedData(
        modality=modality,
        samples_by_split=samples_by_split,
        patches_by_split=patches_by_split,
        reducer=reducer,
        stats={
            "split": split_stats,
            "class_names": list(cfg.data.class_names),
            "num_classes": num_classes,
            "spectral_alignment": alignment_meta,
        },
    )

    _PREPARED_CACHE[cache_key] = prepared
    tprint(
        f"prepared {modality} data patches:\n"
        f"\ttrain={len(patches_by_split['train'])}\n"
        f"\teval={len(patches_by_split['eval'])}\n"
        f"\ttest={len(patches_by_split['test'])}"
    )
    return prepared


class BasePatchDataset(Dataset):
    """base patch dataset with optional train-time augmentation."""

    def __init__(
        self,
        config: Munch,
        split: str,
        modality: str,
        prepared: Optional[PreparedData] = None,
        training: bool = False,
    ) -> None:
        self.config = config
        self.split = split
        self.modality = modality
        self.training = training

        self.prepared = prepared or prepare_data(self.config, modality=modality)
        self.samples = self.prepared.samples_by_split[split]
        self.records = self.prepared.patches_by_split[split]

        self.patch_size = int(self.config.data.patch_size)
        self.num_classes = int(self.config.data.num_classes)

        self.enable_train_aug = bool(self.config.augment.enable_train_aug)
        self.flip_prob = float(self.config.augment.flip_prob)
        self.rotate_prob = float(self.config.augment.rotate_prob)
        self.noise_prob = float(self.config.augment.noise_prob)
        self.noise_std = float(self.config.augment.noise_std)
        self.shift_prob = float(self.config.augment.spectral_shift_prob)
        self.shift_std = float(self.config.augment.spectral_shift_std)
        self.band_drop_prob = float(self.config.augment.band_drop_prob)

        self.rng = np.random.default_rng(int(self.config.data.split.seed) + hash(split) % 1000)

        # Bounded cache to avoid unbounded host RAM growth on large cohorts.
        cache_ns = _as_munch(getattr(self.config.runtime, "dataset_cache", Munch()))
        self.max_cached_samples = max(1, int(getattr(cache_ns, "max_items", 12)))
        self._cube_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._mask_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[index]
        sample = self.samples[rec.sample_index]

        image, mask = self._load_sample(sample)
        image, mask = _pad_to_patch(image, mask, self.patch_size)

        img_patch = image[rec.top : rec.top + self.patch_size, rec.left : rec.left + self.patch_size, :]
        mask_patch = mask[rec.top : rec.top + self.patch_size, rec.left : rec.left + self.patch_size]

        # Keep HSI patch casting local to the sampled window to avoid full-cube copies.
        img_patch = np.asarray(img_patch, dtype=np.float32)

        if self.training and self.enable_train_aug:
            img_patch, mask_patch = self._apply_augment(img_patch, mask_patch)

        if self.modality == "hsi":
            img_patch = self.prepared.reducer.transform(img_patch)
        else:
            img_patch = img_patch.transpose(2, 0, 1)

        return torch.from_numpy(np.ascontiguousarray(img_patch, dtype=np.float32)), torch.from_numpy(mask_patch.astype(np.int64))

    def _load_sample(self, sample: SampleItem) -> Tuple[np.ndarray, np.ndarray]:
        """load image and mask for one sample."""
        sid = sample.sample_id
        if sid in self._mask_cache:
            self._mask_cache.move_to_end(sid)
        else:
            self._mask_cache[sid] = np.load(sample.mask_path).astype(np.int64)
            while len(self._mask_cache) > self.max_cached_samples:
                self._mask_cache.popitem(last=False)

        if sid in self._cube_cache:
            self._cube_cache.move_to_end(sid)
        else:
            if self.modality == "hsi":
                cube = np.load(sample.hsi_path, mmap_mode="r")
            else:
                if sample.rgb_path is None:
                    raise FileNotFoundError(f"rgb file not found for sample: {sample.sample_id}")
                rgb = cv2.imread(str(sample.rgb_path), cv2.IMREAD_COLOR)
                if rgb is None:
                    raise FileNotFoundError(f"failed to read rgb image: {sample.rgb_path}")
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                cube = rgb.astype(np.float32) / 255.0
            self._cube_cache[sid] = cube
            while len(self._cube_cache) > self.max_cached_samples:
                self._cube_cache.popitem(last=False)

        return self._cube_cache[sid], self._mask_cache[sid]

    def _apply_augment(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """apply configured train-time augmentation to one patch."""
        img = image.copy()
        msk = mask.copy()

        if self.rng.random() < self.flip_prob:
            if self.rng.random() < 0.5:
                img = img[:, ::-1, :]
                msk = msk[:, ::-1]
            else:
                img = img[::-1, :, :]
                msk = msk[::-1, :]

        if self.rng.random() < self.rotate_prob:
            k = int(self.rng.integers(1, 4))
            img = np.rot90(img, k=k, axes=(0, 1)).copy()
            msk = np.rot90(msk, k=k).copy()

        if self.modality == "hsi":
            if self.rng.random() < self.noise_prob:
                img += self.rng.normal(0.0, self.noise_std, size=img.shape).astype(np.float32)

            if self.rng.random() < self.shift_prob:
                shift = self.rng.normal(0.0, self.shift_std, size=(1, 1, img.shape[2])).astype(np.float32)
                img += shift

            if self.rng.random() < self.band_drop_prob:
                n_drop = int(self.rng.integers(1, 3))
                channels = self.rng.choice(img.shape[2], size=n_drop, replace=False)
                img[:, :, channels] = 0.0

        return img.astype(np.float32), msk.astype(np.int64)


class NpyHSIDataset(BasePatchDataset):
    """hsi dataset with strict subject split and patch filtering.

    input:
        config(munch): runtime config tree.
        split(str): train/eval/test.
    output item:
        image tensor (c, h, w), mask tensor (h, w).
    """

    def __init__(self, config: Munch, split: str = "train", prepared: Optional[PreparedData] = None):
        super().__init__(
            config=config,
            split=split,
            modality="hsi",
            prepared=prepared,
            training=(split == "train"),
        )


class RGBDataset(BasePatchDataset):
    """rgb dataset using same masks and patch policy as hsi dataset."""

    def __init__(self, config: Munch, split: str = "train", prepared: Optional[PreparedData] = None):
        super().__init__(
            config=config,
            split=split,
            modality="rgb",
            prepared=prepared,
            training=(split == "train"),
        )


def build_dataloaders(config: Munch, modality: str) -> Tuple[Dict[str, DataLoader], PreparedData]:
    """build train/eval/test dataloaders for selected modality."""
    cfg = config
    prepared = prepare_data(cfg, modality=modality)

    if modality == "hsi":
        train_ds = NpyHSIDataset(cfg, split="train", prepared=prepared)
        eval_ds = NpyHSIDataset(cfg, split="eval", prepared=prepared)
        test_ds = NpyHSIDataset(cfg, split="test", prepared=prepared)
    else:
        train_ds = RGBDataset(cfg, split="train", prepared=prepared)
        eval_ds = RGBDataset(cfg, split="eval", prepared=prepared)
        test_ds = RGBDataset(cfg, split="test", prepared=prepared)

    train_bs = int(cfg.train.batch_size)
    eval_bs = int(cfg.train.eval_batch_size)
    num_workers = int(cfg.train.num_workers)
    prefetch_factor = max(1, int(getattr(cfg.train, "prefetch_factor", 2)))
    dl_cfg = _as_munch(getattr(cfg.runtime, "dataloader", Munch()))
    max_workers = int(getattr(dl_cfg, "max_workers", 4))
    prefetch_cap = int(getattr(dl_cfg, "prefetch_cap", 2))
    persistent_workers_cfg = bool(getattr(dl_cfg, "persistent_workers", False))
    if max_workers > 0:
        num_workers = min(num_workers, max_workers)
    prefetch_factor = max(1, min(prefetch_factor, prefetch_cap))

    runtime_device = str(getattr(cfg.runtime, "device", "auto")).lower()
    use_cuda = torch.cuda.is_available() and runtime_device != "cpu"

    loader_kwargs: Dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": bool(use_cuda),
        "persistent_workers": bool(num_workers > 0 and persistent_workers_cfg),
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    advanced_sampling_cfg = _as_munch(getattr(getattr(cfg.data, "sampling", Munch()), "advanced", Munch()))
    use_weighted_sampler = bool(getattr(advanced_sampling_cfg, "use_weighted_sampler", False))
    train_sampler = None
    if use_weighted_sampler and modality == "hsi" and len(train_ds.records) > 0:
        weights = np.asarray([max(1e-6, float(getattr(rec, "weight", 1.0))) for rec in train_ds.records], dtype=np.float64)
        train_sampler = WeightedRandomSampler(
            weights=torch.from_numpy(weights).double(),
            num_samples=len(train_ds.records),
            replacement=True,
        )

    loaders = {
        "train": DataLoader(train_ds, batch_size=train_bs, shuffle=(train_sampler is None), sampler=train_sampler, **loader_kwargs),
        "eval": DataLoader(eval_ds, batch_size=eval_bs, shuffle=False, **loader_kwargs),
        "test": DataLoader(test_ds, batch_size=eval_bs, shuffle=False, **loader_kwargs),
    }

    tprint(
        f"dataloaders[{modality}]:\n"
        f"\ttrain_samples={len(train_ds)}, eval_samples={len(eval_ds)}, test_samples={len(test_ds)}\n"
        f"\ttrain_batches={len(loaders['train'])}, eval_batches={len(loaders['eval'])}, test_batches={len(loaders['test'])}\n"
        f"\tbatch_size(train/eval)={train_bs}/{eval_bs}, workers={num_workers}, prefetch={prefetch_factor}\n"
        f"\tpin_memory={loader_kwargs['pin_memory']}, persistent_workers={loader_kwargs['persistent_workers']}\n"
        f"\tweighted_sampler={train_sampler is not None}\n"
    )
    return loaders, prepared


def generate_synthetic_dataset(
    root_dir: str,
    num_subjects: int = 12,
    samples_per_subject: int = 2,
    image_size: int = 128,
    num_bands: int = 96,
    seed: int = 3407,
    domain_shift_strength: float = 0.12,
    noise_std: float = 0.03,
    boundary_mix_sigma: float = 1.2,
    label_noise_prob: float = 0.01,
) -> Dict[str, str]:
    """generate synthetic hsi/rgb paired dataset for end-to-end validation.

    Compared with the previous synthetic generator, this version intentionally
    introduces stronger variability and domain shift so the task is no longer
    trivially solved by memorizing class spectral templates.
    """
    root = Path(root_dir)
    fisher_dir = root / "fisher"
    rgb_dir = root / "rgb"

    # Ensure each generation run produces an isolated batch instead of accumulating stale samples.
    if fisher_dir.exists():
        shutil.rmtree(fisher_dir)
    if rgb_dir.exists():
        shutil.rmtree(rgb_dir)

    fisher_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    directions = ["LU", "RU", "LD", "RD", "CF", "AX"]

    # Class-specific spectral templates.
    grid = np.linspace(0.0, 1.0, num_bands, dtype=np.float32)
    base_templates = np.stack(
        [
            0.15 + 0.05 * np.sin(grid * 2.0 * np.pi),
            0.45 + 0.10 * np.cos(grid * 3.0 * np.pi + 0.2),
            0.35 + 0.08 * np.sin(grid * 4.0 * np.pi + 0.8),
            0.55 + 0.12 * np.cos(grid * 2.5 * np.pi + 1.2),
        ],
        axis=0,
    ).astype(np.float32)

    # Slightly reduce deterministic separability at initialization.
    base_templates += rng.normal(0.0, 0.01, size=base_templates.shape).astype(np.float32)

    num_classes = 4

    def _sample_variant_templates(patient_rng: np.random.Generator) -> np.ndarray:
        """Sample per-class spectral templates with patient/sample-specific shifts."""
        patient_bias = patient_rng.normal(
            0.0,
            domain_shift_strength * 0.25,
            size=(num_bands,),
        ).astype(np.float32)
        patient_gain = float(np.clip(1.0 + patient_rng.normal(0.0, domain_shift_strength * 0.30), 0.75, 1.25))

        var = np.zeros((num_classes, num_bands), dtype=np.float32)
        for cls_idx in range(num_classes):
            phase = float(patient_rng.uniform(0.0, 2.0 * np.pi))
            freq = float(1.0 + 3.0 * patient_rng.random())
            harmonic = 0.03 * np.sin(grid * freq * 2.0 * np.pi + phase)
            band_scale = 1.0 + patient_rng.normal(0.0, 0.05, size=(num_bands,)).astype(np.float32)
            band_bias = patient_rng.normal(0.0, 0.015, size=(num_bands,)).astype(np.float32)

            tmpl = base_templates[cls_idx] * band_scale
            tmpl = tmpl + harmonic.astype(np.float32)
            tmpl = tmpl + band_bias + patient_bias
            tmpl = tmpl * patient_gain
            var[cls_idx] = np.clip(tmpl, 0.0, 1.0)

        # Add mild class overlap by linear mixing.
        mix = np.eye(num_classes, dtype=np.float32) * 0.84 + 0.16 / float(num_classes)
        mix += patient_rng.normal(0.0, 0.025, size=(num_classes, num_classes)).astype(np.float32)
        mix = np.clip(mix, 0.01, None)
        mix = mix / np.maximum(mix.sum(axis=1, keepdims=True), 1e-6)
        var = mix @ var
        return np.clip(var, 0.0, 1.0)

    def _soft_labels(mask_2d: np.ndarray, sigma: float) -> np.ndarray:
        """Build soft class weights to simulate spectral mixing near boundaries."""
        one_hot = np.stack([(mask_2d == c).astype(np.float32) for c in range(num_classes)], axis=0)
        soft = np.empty_like(one_hot)
        for c in range(num_classes):
            soft[c] = cv2.GaussianBlur(one_hot[c], (0, 0), sigmaX=sigma, sigmaY=sigma)
        soft_sum = np.maximum(soft.sum(axis=0, keepdims=True), 1e-6)
        return soft / soft_sum

    def _add_artifacts(cube: np.ndarray, sample_rng: np.random.Generator) -> np.ndarray:
        """Add realistic illumination and sensor artifacts."""
        _, h, w = cube.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        yy /= max(1.0, float(h - 1))
        xx /= max(1.0, float(w - 1))

        grad_x = float(sample_rng.normal(0.0, 0.10))
        grad_y = float(sample_rng.normal(0.0, 0.10))
        wave_amp = float(sample_rng.uniform(0.03, 0.10))
        wave_fx = float(sample_rng.uniform(0.4, 1.6))
        wave_fy = float(sample_rng.uniform(0.4, 1.6))
        wave_phase = float(sample_rng.uniform(0.0, 2.0 * np.pi))

        illum = 1.0 + grad_x * (xx - 0.5) + grad_y * (yy - 0.5)
        illum += wave_amp * np.sin(2.0 * np.pi * (wave_fx * xx + wave_fy * yy) + wave_phase)
        cube = cube * illum[None, :, :]

        # Correlated band noise + per-pixel noise.
        band_noise = sample_rng.normal(0.0, noise_std, size=(cube.shape[0], 1, 1)).astype(np.float32)
        pixel_noise = sample_rng.normal(0.0, noise_std * 0.40, size=cube.shape).astype(np.float32)
        cube = cube + band_noise + pixel_noise

        # Random bright spots (specular-like artifacts).
        n_spots = int(sample_rng.integers(0, 3))
        for _ in range(n_spots):
            cx = float(sample_rng.uniform(0.0, w - 1))
            cy = float(sample_rng.uniform(0.0, h - 1))
            rad = float(sample_rng.uniform(max(2.0, image_size * 0.04), max(3.0, image_size * 0.12)))
            dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
            spot = np.exp(-dist2 / max(1e-6, 2.0 * (rad ** 2))).astype(np.float32)
            spec_gain = sample_rng.uniform(0.01, 0.08, size=(cube.shape[0], 1, 1)).astype(np.float32)
            cube = cube + spec_gain * spot[None, :, :]

        # Column stripe artifact.
        if sample_rng.random() < 0.35:
            col = int(sample_rng.integers(0, w))
            half_width = int(sample_rng.integers(1, 4))
            lo = max(0, col - half_width)
            hi = min(w, col + half_width + 1)
            stripe = sample_rng.normal(0.0, noise_std * 1.2, size=(cube.shape[0], 1, 1)).astype(np.float32)
            cube[:, :, lo:hi] += stripe

        return np.clip(cube, 0.0, 1.0)

    def _inject_boundary_label_noise(mask_2d: np.ndarray, sample_rng: np.random.Generator) -> np.ndarray:
        """Apply tiny boundary-only label flips to avoid unrealistically clean labels."""
        if label_noise_prob <= 0.0:
            return mask_2d

        kernel = np.ones((3, 3), dtype=np.uint8)
        boundary = cv2.morphologyEx(mask_2d.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
        random_flip = sample_rng.random(size=mask_2d.shape) < label_noise_prob
        flip_mask = boundary & random_flip
        if not np.any(flip_mask):
            return mask_2d

        out = mask_2d.copy()
        old = out[flip_mask]
        rnd = sample_rng.integers(0, num_classes, size=old.shape[0], dtype=np.int64)
        rnd = (rnd + (rnd == old).astype(np.int64)) % num_classes
        out[flip_mask] = rnd.astype(out.dtype)
        return out

    for s_idx in range(num_subjects):
        patient_name = f"P{s_idx:03d}"
        patient_seed = int(seed + s_idx * 100003)
        patient_rng = np.random.default_rng(patient_seed)
        for k in range(samples_per_subject):
            sample_rng = np.random.default_rng(patient_seed + k * 1009)

            date = f"20260{1 + (s_idx % 6):02d}{1 + (k % 28):02d}"
            direction = directions[(s_idx + k) % len(directions)]
            sample_id = f"{patient_name}_{date}_{direction}"

            h, w = image_size, image_size
            mask = np.zeros((h, w), dtype=np.uint8)

            # thyroid region (class 3).
            center = (
                int(w * (0.35 + 0.25 * sample_rng.random())),
                int(h * (0.40 + 0.20 * sample_rng.random())),
            )
            axes = (
                int(w * (0.22 + 0.08 * sample_rng.random())),
                int(h * (0.18 + 0.07 * sample_rng.random())),
            )
            cv2.ellipse(mask, center, axes, 0.0, 0.0, 360.0, color=3, thickness=-1)

            # trachea region (class 2).
            x0 = int(w * (0.58 + 0.12 * sample_rng.random()))
            y0 = int(h * (0.30 + 0.18 * sample_rng.random()))
            x1 = min(w - 1, x0 + int(w * 0.12))
            y1 = min(h - 1, y0 + int(h * 0.26))
            cv2.rectangle(mask, (x0, y0), (x1, y1), color=2, thickness=-1)

            # precious small targets (class 1).
            n_small = int(sample_rng.integers(1, 4))
            for _ in range(n_small):
                cx = int(w * (0.40 + 0.25 * sample_rng.random()))
                cy = int(h * (0.42 + 0.16 * sample_rng.random()))
                rad = int(max(2, image_size * (0.012 + 0.014 * sample_rng.random())))
                cv2.circle(mask, (cx, cy), radius=rad, color=1, thickness=-1)

            mask = _inject_boundary_label_noise(mask, sample_rng).astype(np.uint8)

            # Class mixing near boundaries makes decision regions less trivial.
            sigma = float(boundary_mix_sigma * sample_rng.uniform(0.5, 1.4))
            soft = _soft_labels(mask, sigma=max(0.1, sigma))

            sample_templates = _sample_variant_templates(patient_rng)
            cube = (sample_templates.T @ soft.reshape(num_classes, -1)).reshape(num_bands, h, w)
            cube = _add_artifacts(cube.astype(np.float32), sample_rng)

            hsi_path = fisher_dir / f"{sample_id}.npy"
            gt_path = fisher_dir / f"{sample_id}_gt.npy"
            np.save(hsi_path, cube.astype(np.float32))
            np.save(gt_path, mask.astype(np.int64))

            # Create RGB by random band mixing + gamma shift to mimic camera style variance.
            rgb_w = sample_rng.uniform(0.0, 1.0, size=(3, num_bands)).astype(np.float32)
            rgb_w = rgb_w / np.maximum(rgb_w.sum(axis=1, keepdims=True), 1e-6)
            rgb = (rgb_w @ cube.reshape(num_bands, -1)).reshape(3, h, w).transpose(1, 2, 0)
            gamma = float(sample_rng.uniform(0.8, 1.25))
            rgb = np.clip(rgb, 0.0, 1.0) ** gamma
            rgb = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
            cv2.imwrite(str(rgb_dir / f"{sample_id}_Merged_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    manifest = {
        "generator": "synthetic_hard_v2",
        "seed": seed,
        "num_subjects": num_subjects,
        "samples_per_subject": samples_per_subject,
        "image_size": image_size,
        "num_bands": num_bands,
        "domain_shift_strength": domain_shift_strength,
        "noise_std": noise_std,
        "boundary_mix_sigma": boundary_mix_sigma,
        "label_noise_prob": label_noise_prob,
    }
    with (root / "synthetic_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    tprint(
        "synthetic dataset generated at:\n"
        f"\t{root} (generator=synthetic_hard_v2, domain_shift_strength={domain_shift_strength},\n"
        f"\tnoise_std={noise_std}, boundary_mix_sigma={boundary_mix_sigma},\n"
        f"\tlabel_noise_prob={label_noise_prob})"
    )
    return {
        "hsi_dir": str(fisher_dir),
        "label_dir": str(fisher_dir),
        "rgb_dir": str(rgb_dir),
    }
