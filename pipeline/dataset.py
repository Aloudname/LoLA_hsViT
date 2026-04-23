from __future__ import annotations

from tqdm import tqdm
from munch import Munch
from pathlib import Path
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
from concurrent.futures import ThreadPoolExecutor, as_completed

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

    modes:
    - none: identity.
    - supervised_pca: pca on class-balanced train pixels.
    - lda_pca: pca first, then lda on pca features.
    - kernel_lda: random fourier feature projection + lda.
    - pca_nca: optional spectral normalization/derivative, then pca + neighborhood components analysis.
    """

    def __init__(
        self,
        mode: str,
        output_dim: int,
        max_fit_pixels: int,
        seed: int,
        preprocess_cfg: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.mode = mode.lower()
        self.output_dim = output_dim
        self.max_fit_pixels = max_fit_pixels
        self.seed = seed

        cfg = _as_munch(preprocess_cfg or {})
        self.enable_snv = bool(getattr(cfg, "snv", self.mode == "pca_nca"))
        self.derivative_order = max(0, int(getattr(cfg, "derivative_order", 1 if self.mode == "pca_nca" else 0)))
        self.savgol_window = max(3, int(getattr(cfg, "savgol_window", 7)))
        self.savgol_polyorder = max(1, int(getattr(cfg, "savgol_polyorder", 2)))
        self.enable_standardize = bool(getattr(cfg, "standardize", self.mode == "pca_nca"))
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
            # simple norm for none-mode
            x_fit, _ = self._collect_pixels(samples, num_classes, show_progress=show_progress)
            if x_fit.size == 0:
                raise RuntimeError("failed to collect fit pixels for none-mode normalization")
            _ = self._prepare_features(x_fit, fit=True)
            
            self.fitted = True
            self.reference_projection_name = "Raw spectra"
            self.reduced_projection_name = "Normalized spectra"
            return

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
        },
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


def _prepare_none_mode_cache(
    cfg: Munch,
    samples_by_split: Mapping[str, Sequence[SampleItem]],
    reducer: SpectralReducer,
    show_progress: bool,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    prepare cached none-mode normed cubes.
    use multi-threading to speed up.
    """
    cache_cfg = _as_munch(getattr(getattr(cfg.data, "preprocess", Munch()), "cache", Munch()))
    cache_enabled = bool(getattr(cache_cfg, "enabled", True))
    if not cache_enabled:
        return {}, {"enabled": False, "reason": "cache disabled"}

    cached_root = Path(getattr(cfg.path, "cached_dir", "./cached_hsi"))
    try:
        cached_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        tprint(
            "Warning: cached_dir is not writable\n"
            f"\trequested={cached_root}\n"
            f"\treason={exc}"
        )
        raise RuntimeError(f"cached_dir is not writable: {cached_root}") from exc

    manifest_path = cached_root / "cache_manifest.json"
    norm_root = cached_root / "normed"
    stats_root = cached_root / "stats"
    norm_root.mkdir(parents=True, exist_ok=True)
    stats_root.mkdir(parents=True, exist_ok=True)

    key_payload = {
        "seed": int(cfg.data.split.seed),
        "version_tag": str(getattr(cache_cfg, "version_tag", "v1")),
        "mode": str(cfg.data.preprocess.mode),
        "preprocess": dict(cfg.data.preprocess),
        "signature": _cache_signature(samples_by_split),
    }
    key_raw = json.dumps(key_payload, sort_keys=True)
    cache_key = hashlib.sha256(key_raw.encode("utf-8")).hexdigest()[:20]

    norm_dir = norm_root / cache_key
    stats_dir = stats_root / cache_key
    stats_dir.mkdir(parents=True, exist_ok=True)
    stats_path = stats_dir / "train_stats.npz"
    strict_manifest = bool(getattr(cache_cfg, "strict_manifest", True))

    existing_manifest = {}
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            existing_manifest = json.load(f)

    all_samples = [s for split in samples_by_split.values() for s in split]
    if strict_manifest:
        # check if cache valid
        matched = existing_manifest.get(cache_key, None)
        if matched is not None and norm_dir.exists() and stats_path.exists():
            all_ok = True
            for sample in all_samples:
                if not (norm_dir / f"{sample.sample_id}.npy").exists():
                    all_ok = False
                    break
            if all_ok:
                return (
                    {sample.sample_id: str(norm_dir / f"{sample.sample_id}.npy") for sample in all_samples},
                    {"enabled": True, "cache_key": cache_key, "cache_hit": True, "stats_path": str(stats_path)},
                )

    # collect train pixels for normed cube standardization
    if reducer.enable_standardize and reducer.scaler is None:
        train_samples = list(samples_by_split.get("train", []))
        x_fit, _ = reducer._collect_pixels(train_samples, num_classes=int(cfg.data.num_classes), show_progress=show_progress)
        if x_fit.size == 0:
            raise RuntimeError("failed to collect train pixels for none-mode cache standardization")
        reducer._prepare_features(x_fit, fit=True)

    reserve_cores = max(1, int(getattr(cache_cfg, "reserve_cores", 2)))
    user_workers = max(1, int(getattr(cache_cfg, "num_workers", 4)))
    cpu_total = os.cpu_count() or 4
    workers = max(1, min(user_workers, max(1, cpu_total - reserve_cores)))

    norm_dir.mkdir(parents=True, exist_ok=True)
    if reducer.scaler is not None:
        np.savez(
            stats_path,
            mean_=reducer.scaler.mean_.astype(np.float32),
            scale_=reducer.scaler.scale_.astype(np.float32),
        )

    def _process_one(sample: SampleItem) -> str:
        cube = np.load(sample.hsi_path).astype(np.float32)
        h, w, c = cube.shape
        flat = cube.reshape(-1, c)
        norm_flat = reducer._prepare_features(flat, fit=False)
        out_path = norm_dir / f"{sample.sample_id}.npy"
        np.save(out_path, norm_flat.reshape(h, w, c).astype(np.float32))
        return str(out_path)

    cache_paths: Dict[str, str] = {}
    try:
        # context manager for thread pool
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_one, sample): sample for sample in all_samples}
            # progress bar for threading
            iterator = tqdm(
                as_completed(futures),
                total=len(futures),
                desc="none-cache",
                dynamic_ncols=True,
                leave=False,
                disable=not show_progress,
            )
            for future in iterator:
                sample = futures[future]
                cache_paths[sample.sample_id] = future.result()
    except Exception as exc:
        tprint(f"Warning: threaded none-mode cache failed, fallback to single-thread. reason={exc}")
        cache_paths = {}
        for sample in tqdm(
            all_samples,
            desc="none-cache-fallback",
            dynamic_ncols=True,
            leave=False,
            disable=not show_progress,
        ):
            cache_paths[sample.sample_id] = _process_one(sample)

    # update manifest
    existing_manifest[cache_key] = {
        "key_payload": key_payload,
        "norm_dir": str(norm_dir),
        "stats_path": str(stats_path),
        "num_samples": len(all_samples),
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(existing_manifest, f, indent=2, ensure_ascii=False)
    return (
        cache_paths,
        {"enabled": True, "cache_key": cache_key, "cache_hit": False, "stats_path": str(stats_path), "workers": workers},
    )


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

    reducer = SpectralReducer(
        mode=str(cfg.data.preprocess.mode),
        output_dim=int(cfg.data.preprocess.output_dim),
        max_fit_pixels=int(cfg.data.preprocess.max_fit_pixels),
        seed=seed,
        preprocess_cfg=dict(cfg.data.preprocess),
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

    cached_hsi_paths: Dict[str, str] = {}
    none_cache_meta: Dict[str, Any] = {"enabled": False}
    if modality == "hsi" and reducer.mode == "none":
        cached_hsi_paths, none_cache_meta = _prepare_none_mode_cache(
            cfg=cfg,
            samples_by_split=samples_by_split,
            reducer=reducer,
            show_progress=show_progress,
        )
        tprint(
            "none-mode cache:\n"
            f"\tenabled={none_cache_meta.get('enabled', False)}, hit={none_cache_meta.get('cache_hit', False)}\n"
            f"\tkey={none_cache_meta.get('cache_key', 'n/a')}"
        )

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
            "cached_hsi_paths": cached_hsi_paths,
            "none_cache_meta": none_cache_meta,
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

        self._cube_cache: Dict[str, np.ndarray] = {}
        self._mask_cache: Dict[str, np.ndarray] = {}

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
        if sample.sample_id not in self._mask_cache:
            self._mask_cache[sample.sample_id] = np.load(sample.mask_path).astype(np.int64)

        if sample.sample_id not in self._cube_cache:
            if self.modality == "hsi":
                cached_paths = self.prepared.stats.get("cached_hsi_paths", {})
                cached_path = cached_paths.get(sample.sample_id, None)
                if cached_path is not None and Path(cached_path).exists():
                    cube = np.load(cached_path, mmap_mode="r")
                else:
                    cube = np.load(sample.hsi_path, mmap_mode="r")
            else:
                if sample.rgb_path is None:
                    raise FileNotFoundError(f"rgb file not found for sample: {sample.sample_id}")
                rgb = cv2.imread(str(sample.rgb_path), cv2.IMREAD_COLOR)
                if rgb is None:
                    raise FileNotFoundError(f"failed to read rgb image: {sample.rgb_path}")
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                cube = rgb.astype(np.float32) / 255.0
            self._cube_cache[sample.sample_id] = cube

        return self._cube_cache[sample.sample_id], self._mask_cache[sample.sample_id]

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

    runtime_device = str(getattr(cfg.runtime, "device", "auto")).lower()
    use_cuda = torch.cuda.is_available() and runtime_device != "cpu"

    loader_kwargs: Dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": bool(use_cuda),
        "persistent_workers": num_workers > 0,
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
