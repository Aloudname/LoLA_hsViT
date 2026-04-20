# dataset and preprocessing pipeline for hsi/rgb segmentation.
import cv2, json, torch, numpy as np

from munch import Munch
from pathlib import Path
from dataclasses import dataclass
from __future__ import annotations
from pipeline.monitor import tprint
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


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


@dataclass
class PreparedData:
    """prepared split and patch cache shared by train/eval/test datasets."""

    modality: str
    samples_by_split: Dict[str, List[SampleItem]]
    patches_by_split: Dict[str, List[PatchRecord]]
    reducer: "SpectralReducer"
    stats: Dict[str, Any]


def _as_munch(config: Mapping[str, Any]) -> Munch:
    """convert mapping to munch."""
    if isinstance(config, Munch):
        return config
    return Munch.fromDict(dict(config))


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

    for y in ys:
        for x in xs:
            yield y, x


def _pad_to_patch(image: np.ndarray, mask: np.ndarray, patch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """pad image and mask to at least patch_size."""
    h, w = mask.shape
    pad_h = max(0, patch_size - h)
    pad_w = max(0, patch_size - w)
    if pad_h == 0 and pad_w == 0:
        return image, mask

    image_pad = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    mask_pad = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    return image_pad, mask_pad


def _compute_hist(mask: np.ndarray, num_classes: int) -> np.ndarray:
    """compute class histogram from mask."""
    return np.bincount(mask.reshape(-1), minlength=num_classes).astype(np.int64)


def _should_keep_patch(mask_patch: np.ndarray, foreground_ratio_threshold: float) -> bool:
    """patch keep rule.

    keep when:
    - class-1 exists, or
    - foreground ratio >= threshold.
    """
    has_small_target = bool(np.any(mask_patch == 1))
    fg_ratio = float(np.mean(mask_patch > 0))
    return has_small_target or fg_ratio >= foreground_ratio_threshold


class SpectralReducer:
    """fit/transform spectral channels for method-b preprocessing.

    modes:
    - none: identity.
    - supervised_pca: pca on class-balanced train pixels.
    - lda_pca: lda first, then pca padding to target dim.
    """

    def __init__(self, mode: str, output_dim: int, max_fit_pixels: int, seed: int) -> None:
        self.mode = str(mode).lower()
        self.output_dim = int(output_dim)
        self.max_fit_pixels = int(max_fit_pixels)
        self.seed = int(seed)

        self.pca: Optional[PCA] = None
        self.lda: Optional[LinearDiscriminantAnalysis] = None
        self.pca_tail: Optional[PCA] = None
        self.fitted = False

    def fit(self, samples: Sequence[SampleItem], num_classes: int) -> None:
        """fit reducer from train split pixels.

        input:
            samples: training samples.
            num_classes: class count.
        """
        if self.mode == "none":
            self.fitted = True
            return

        x_fit, y_fit = self._collect_pixels(samples, num_classes)
        if x_fit.size == 0:
            raise RuntimeError("failed to collect fit pixels for spectral reducer")

        if self.mode == "supervised_pca":
            self.pca = PCA(n_components=min(self.output_dim, x_fit.shape[1]))
            self.pca.fit(x_fit)
            self.fitted = True
            return

        if self.mode == "lda_pca":
            lda_dim = int(min(max(1, len(np.unique(y_fit)) - 1), self.output_dim))
            try:
                self.lda = LinearDiscriminantAnalysis(n_components=lda_dim)
                self.lda.fit(x_fit, y_fit)
            except Exception:
                self.lda = None
                lda_dim = 0

            remain = self.output_dim - lda_dim
            if remain > 0:
                self.pca_tail = PCA(n_components=min(remain, x_fit.shape[1]))
                self.pca_tail.fit(x_fit)
            self.fitted = True
            return

        raise ValueError(f"unknown reducer mode: {self.mode}")

    def transform(self, image: np.ndarray) -> np.ndarray:
        """transform image channels.

        input:
            image: (c, h, w).
        output:
            transformed image: (c_new, h, w).
        """
        if self.mode == "none":
            return image
        if not self.fitted:
            raise RuntimeError("spectral reducer must be fitted before transform")

        c, h, w = image.shape
        x = image.transpose(1, 2, 0).reshape(-1, c)

        parts: List[np.ndarray] = []
        if self.lda is not None:
            parts.append(self.lda.transform(x).astype(np.float32))
        if self.pca is not None:
            parts.append(self.pca.transform(x).astype(np.float32))
        if self.pca_tail is not None:
            parts.append(self.pca_tail.transform(x).astype(np.float32))

        if not parts:
            return image

        y = np.concatenate(parts, axis=1)
        if y.shape[1] < self.output_dim:
            pad = np.zeros((y.shape[0], self.output_dim - y.shape[1]), dtype=np.float32)
            y = np.concatenate([y, pad], axis=1)
        if y.shape[1] > self.output_dim:
            y = y[:, : self.output_dim]

        return y.reshape(h, w, self.output_dim).transpose(2, 0, 1)

    def _collect_pixels(self, samples: Sequence[SampleItem], num_classes: int) -> Tuple[np.ndarray, np.ndarray]:
        """collect class-balanced train pixels for fitting."""
        rng = np.random.default_rng(self.seed)
        per_class_budget = max(256, self.max_fit_pixels // max(1, num_classes))
        x_by_class: List[List[np.ndarray]] = [[] for _ in range(num_classes)]

        for sample in samples:
            cube = np.load(sample.hsi_path).astype(np.float32)
            mask = np.load(sample.mask_path).astype(np.int64)
            cube, mask = _pad_to_patch(cube, mask, patch_size=1)
            h, w = mask.shape

            flat = cube.reshape(cube.shape[0], h * w).T
            y = mask.reshape(-1)

            for cls_idx in range(num_classes):
                idx = np.where(y == cls_idx)[0]
                if idx.size == 0:
                    continue
                take = min(idx.size, max(32, per_class_budget // max(1, len(samples))))
                chosen = rng.choice(idx, size=take, replace=False)
                x_by_class[cls_idx].append(flat[chosen])

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
) -> Tuple[List[PatchRecord], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """build patch index list and class histograms.

    output:
        records: kept patch indices.
        raw_hist: full-image pixel histogram.
        patch_hist: kept-patch pixel histogram.
        tracked_pre: tracked sample histogram before filtering.
        tracked_post: tracked sample histogram after filtering.
    """
    tracked_set = set(tracked_sample_ids)
    records: List[PatchRecord] = []

    raw_hist = np.zeros(num_classes, dtype=np.int64)
    patch_hist = np.zeros(num_classes, dtype=np.int64)
    tracked_pre = np.zeros(num_classes, dtype=np.int64)
    tracked_post = np.zeros(num_classes, dtype=np.int64)

    for sample_index, sample in enumerate(samples):
        mask = np.load(sample.mask_path).astype(np.int64)
        raw_hist += _compute_hist(mask, num_classes)

        cube = np.load(sample.hsi_path, mmap_mode="r")
        _, h, w = cube.shape

        if h < patch_size or w < patch_size:
            pad_h = max(0, patch_size - h)
            pad_w = max(0, patch_size - w)
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
            h, w = mask.shape

        sample_kept = 0
        for top, left in _iter_patch_positions(h, w, patch_size, stride):
            patch_mask = mask[top : top + patch_size, left : left + patch_size]
            if sample.sample_id in tracked_set:
                tracked_pre += _compute_hist(patch_mask, num_classes)

            keep = _should_keep_patch(patch_mask, foreground_ratio_threshold)
            if keep:
                records.append(PatchRecord(sample_index=sample_index, top=top, left=left))
                patch_hist += _compute_hist(patch_mask, num_classes)
                sample_kept += 1
                if sample.sample_id in tracked_set:
                    tracked_post += _compute_hist(patch_mask, num_classes)

        if sample_kept == 0:
            center_top = max(0, (h - patch_size) // 2)
            center_left = max(0, (w - patch_size) // 2)
            patch_mask = mask[center_top : center_top + patch_size, center_left : center_left + patch_size]
            records.append(PatchRecord(sample_index=sample_index, top=center_top, left=center_left))
            patch_hist += _compute_hist(patch_mask, num_classes)
            if sample.sample_id in tracked_set:
                tracked_post += _compute_hist(patch_mask, num_classes)

    return records, raw_hist, patch_hist, tracked_pre, tracked_post


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
        "preprocess": dict(config.data.preprocess),
    }
    return json.dumps(payload, sort_keys=True)


def prepare_data(config: Mapping[str, Any], modality: str) -> PreparedData:
    """prepare split, patch records, and reducer once."""
    cfg = _as_munch(config)
    modality = modality.lower()

    cache_key = _build_cache_key(cfg, modality)
    if cache_key in _PREPARED_CACHE:
        return _PREPARED_CACHE[cache_key]

    all_samples = _discover_hsi_samples(cfg)
    samples_by_split = _split_subjects(all_samples, cfg)

    patch_size = int(cfg.data.patch_size)
    stride = int(cfg.data.stride)
    threshold = float(cfg.data.foreground_ratio_threshold)
    num_classes = int(cfg.data.num_classes)
    seed = int(cfg.data.split.seed)
    track_ratio = float(cfg.data.track_sample_ratio)

    reducer = SpectralReducer(
        mode=str(cfg.data.preprocess.mode),
        output_dim=int(cfg.data.preprocess.output_dim),
        max_fit_pixels=int(cfg.data.preprocess.max_fit_pixels),
        seed=seed,
    )
    if modality == "hsi":
        reducer.fit(samples_by_split["train"], num_classes=num_classes)
    else:
        reducer.mode = "none"
        reducer.fitted = True

    patches_by_split: Dict[str, List[PatchRecord]] = {}
    split_stats: Dict[str, Any] = {}

    for split_idx, split_name in enumerate(["train", "eval", "test"]):
        split_samples = samples_by_split[split_name]
        tracked_ids = _pick_tracked_samples(split_samples, ratio=track_ratio, seed=seed + split_idx)

        records, raw_hist, patch_hist, tracked_pre, tracked_post = _build_patch_records(
            samples=split_samples,
            patch_size=patch_size,
            stride=stride,
            foreground_ratio_threshold=threshold,
            num_classes=num_classes,
            tracked_sample_ids=tracked_ids,
        )

        patches_by_split[split_name] = records
        split_stats[split_name] = {
            "num_samples": len(split_samples),
            "num_patches": len(records),
            "raw_pixel_hist": raw_hist.tolist(),
            "kept_patch_pixel_hist": patch_hist.tolist(),
            "tracked_sample_ids": tracked_ids,
            "tracked_pre_hist": tracked_pre.tolist(),
            "tracked_post_hist": tracked_post.tolist(),
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
        },
    )

    _PREPARED_CACHE[cache_key] = prepared
    tprint(
        f"prepared {modality} data: "
        f"train={len(patches_by_split['train'])} "
        f"eval={len(patches_by_split['eval'])} "
        f"test={len(patches_by_split['test'])} patches"
    )
    return prepared


class BasePatchDataset(Dataset):
    """base patch dataset with optional train-time augmentation."""

    def __init__(
        self,
        config: Mapping[str, Any],
        split: str,
        modality: str,
        prepared: Optional[PreparedData] = None,
        training: bool = False,
    ) -> None:
        self.config = _as_munch(config)
        self.split = split
        self.modality = modality
        self.training = bool(training)

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

        img_patch = image[:, rec.top : rec.top + self.patch_size, rec.left : rec.left + self.patch_size]
        mask_patch = mask[rec.top : rec.top + self.patch_size, rec.left : rec.left + self.patch_size]

        if self.training and self.enable_train_aug:
            img_patch, mask_patch = self._apply_augment(img_patch, mask_patch)

        if self.modality == "hsi":
            img_patch = self.prepared.reducer.transform(img_patch)

        return torch.from_numpy(img_patch.astype(np.float32)), torch.from_numpy(mask_patch.astype(np.int64))

    def _load_sample(self, sample: SampleItem) -> Tuple[np.ndarray, np.ndarray]:
        """load image and mask for one sample."""
        if sample.sample_id not in self._mask_cache:
            self._mask_cache[sample.sample_id] = np.load(sample.mask_path).astype(np.int64)

        if sample.sample_id not in self._cube_cache:
            if self.modality == "hsi":
                cube = np.load(sample.hsi_path).astype(np.float32)
            else:
                if sample.rgb_path is None:
                    raise FileNotFoundError(f"rgb file not found for sample: {sample.sample_id}")
                rgb = cv2.imread(str(sample.rgb_path), cv2.IMREAD_COLOR)
                if rgb is None:
                    raise FileNotFoundError(f"failed to read rgb image: {sample.rgb_path}")
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                cube = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
            self._cube_cache[sample.sample_id] = cube

        return self._cube_cache[sample.sample_id], self._mask_cache[sample.sample_id]

    def _apply_augment(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """apply configured train-time augmentation to one patch."""
        img = image.copy()
        msk = mask.copy()

        if self.rng.random() < self.flip_prob:
            if self.rng.random() < 0.5:
                img = img[:, :, ::-1]
                msk = msk[:, ::-1]
            else:
                img = img[:, ::-1, :]
                msk = msk[::-1, :]

        if self.rng.random() < self.rotate_prob:
            k = int(self.rng.integers(1, 4))
            img = np.rot90(img, k=k, axes=(1, 2)).copy()
            msk = np.rot90(msk, k=k).copy()

        if self.modality == "hsi":
            if self.rng.random() < self.noise_prob:
                img += self.rng.normal(0.0, self.noise_std, size=img.shape).astype(np.float32)

            if self.rng.random() < self.shift_prob:
                shift = self.rng.normal(0.0, self.shift_std, size=(img.shape[0], 1, 1)).astype(np.float32)
                img += shift

            if self.rng.random() < self.band_drop_prob:
                n_drop = int(self.rng.integers(1, 3))
                channels = self.rng.choice(img.shape[0], size=n_drop, replace=False)
                img[channels] = 0.0

        return img.astype(np.float32), msk.astype(np.int64)


class NpyHSIDataset(BasePatchDataset):
    """hsi dataset with strict subject split and patch filtering.

    input:
        config(mapping): runtime config tree.
        split(str): train/eval/test.
    output item:
        image tensor (c, h, w), mask tensor (h, w).
    """

    def __init__(self, config: Mapping[str, Any], split: str = "train", prepared: Optional[PreparedData] = None):
        super().__init__(
            config=config,
            split=split,
            modality="hsi",
            prepared=prepared,
            training=(split == "train"),
        )


class RGBDataset(BasePatchDataset):
    """rgb dataset using same masks and patch policy as hsi dataset."""

    def __init__(self, config: Mapping[str, Any], split: str = "train", prepared: Optional[PreparedData] = None):
        super().__init__(
            config=config,
            split=split,
            modality="rgb",
            prepared=prepared,
            training=(split == "train"),
        )


def build_dataloaders(config: Mapping[str, Any], modality: str) -> Tuple[Dict[str, DataLoader], PreparedData]:
    """build train/eval/test dataloaders for selected modality."""
    cfg = _as_munch(config)
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

    loaders = {
        "train": DataLoader(train_ds, batch_size=train_bs, shuffle=True, num_workers=num_workers, pin_memory=False),
        "eval": DataLoader(eval_ds, batch_size=eval_bs, shuffle=False, num_workers=num_workers, pin_memory=False),
        "test": DataLoader(test_ds, batch_size=eval_bs, shuffle=False, num_workers=num_workers, pin_memory=False),
    }
    return loaders, prepared


def generate_synthetic_dataset(
    root_dir: str,
    num_subjects: int = 12,
    samples_per_subject: int = 2,
    image_size: int = 128,
    num_bands: int = 96,
    seed: int = 3407,
) -> Dict[str, str]:
    """generate synthetic hsi/rgb paired dataset for end-to-end validation."""
    root = Path(root_dir)
    fisher_dir = root / "fisher"
    rgb_dir = root / "rgb"
    fisher_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    directions = ["LU", "RU", "LD", "RD", "CF", "AX"]

    # class-specific spectral templates.
    grid = np.linspace(0.0, 1.0, num_bands, dtype=np.float32)
    templates = np.stack(
        [
            0.15 + 0.05 * np.sin(grid * 2.0 * np.pi),
            0.45 + 0.10 * np.cos(grid * 3.0 * np.pi + 0.2),
            0.35 + 0.08 * np.sin(grid * 4.0 * np.pi + 0.8),
            0.55 + 0.12 * np.cos(grid * 2.5 * np.pi + 1.2),
        ],
        axis=0,
    ).astype(np.float32)

    for s_idx in range(num_subjects):
        patient_name = f"P{s_idx:03d}"
        for k in range(samples_per_subject):
            date = f"20260{1 + (s_idx % 6):02d}{1 + (k % 28):02d}"
            direction = directions[(s_idx + k) % len(directions)]
            sample_id = f"{patient_name}_{date}_{direction}"

            h, w = image_size, image_size
            mask = np.zeros((h, w), dtype=np.uint8)

            # thyroid region (class 3).
            center = (int(w * (0.35 + 0.25 * rng.random())), int(h * (0.40 + 0.20 * rng.random())))
            axes = (int(w * (0.22 + 0.08 * rng.random())), int(h * (0.18 + 0.07 * rng.random())))
            cv2.ellipse(mask, center, axes, 0.0, 0.0, 360.0, color=3, thickness=-1)

            # trachea region (class 2).
            x0 = int(w * (0.58 + 0.12 * rng.random()))
            y0 = int(h * (0.30 + 0.18 * rng.random()))
            x1 = min(w - 1, x0 + int(w * 0.12))
            y1 = min(h - 1, y0 + int(h * 0.26))
            cv2.rectangle(mask, (x0, y0), (x1, y1), color=2, thickness=-1)

            # parathyroid small targets (class 1).
            n_small = int(rng.integers(1, 3))
            for _ in range(n_small):
                cx = int(w * (0.40 + 0.25 * rng.random()))
                cy = int(h * (0.42 + 0.16 * rng.random()))
                rad = int(max(2, image_size * (0.015 + 0.01 * rng.random())))
                cv2.circle(mask, (cx, cy), radius=rad, color=1, thickness=-1)

            cube = np.zeros((num_bands, h, w), dtype=np.float32)
            noise = rng.normal(0.0, 0.02, size=(num_bands, h, w)).astype(np.float32)
            for cls_idx in range(4):
                cls_mask = mask == cls_idx
                if np.any(cls_mask):
                    cube[:, cls_mask] = templates[cls_idx][:, None]
            cube += noise
            cube = np.clip(cube, 0.0, 1.0)

            hsi_path = fisher_dir / f"{sample_id}.npy"
            gt_path = fisher_dir / f"{sample_id}_gt.npy"
            np.save(hsi_path, cube.astype(np.float32))
            np.save(gt_path, mask.astype(np.int64))

            # create rgb from selected bands.
            rgb = np.stack([cube[12], cube[42], cube[78]], axis=-1)
            rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
            cv2.imwrite(str(rgb_dir / f"{sample_id}_Merged_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    tprint(f"synthetic dataset generated at: {root}")
    return {
        "hsi_dir": str(fisher_dir),
        "label_dir": str(fisher_dir),
        "rgb_dir": str(rgb_dir),
    }
