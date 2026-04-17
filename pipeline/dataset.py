# dataset.py - Hyperspectral Dataset Handling.
# todo:
# - Implement class distribution analysis and visualization methods.
# - Try various seeds for a balanced label distribution in line 542 create_dataloader.
import os, re, time, json, hashlib, cv2, torch, numpy as np, warnings, tempfile
from contextlib import contextmanager
from scipy.ndimage import binary_dilation
from concurrent.futures import ThreadPoolExecutor

from munch import Munch
from pipeline.monitor import tprint
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional, List, Iterator
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset, Sampler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")


def _ensure_background_class(config: Munch) -> None:
    """
    Ensure an explicit background class BG at index 0.

    Existing label maps use 0 for BG. 
    To train/eval segmentation with BG included,
    reserve class-0 for BG, and shift FG count to ``len(targets) + 1``
    thus BG is not explicitly set for simplicity.
    """
    targets = list(getattr(config.clsf, 'targets', []))
    if targets and str(targets[0]).strip().upper() == 'BG':
        config.clsf.targets = targets
        config.clsf.num = len(targets)
        return

    config.clsf.targets = ['BG'] + targets
    config.clsf.num = len(config.clsf.targets)


def _split_cfg(config: Munch, key: str, default, legacy_key: Optional[str] = None):
    """Read split config with nested-path support and legacy fallback."""
    split_cfg = getattr(config, 'split', Munch())
    if '.' in key:
        cur = split_cfg
        for part in key.split('.'):
            if isinstance(cur, (dict, Munch)) and part in cur:
                cur = cur[part]
            else:
                cur = None
                break
        if cur is not None:
            return cur
    elif hasattr(split_cfg, key):
        return getattr(split_cfg, key)

    if legacy_key and hasattr(split_cfg, legacy_key):
        return getattr(split_cfg, legacy_key)
    return default


def _preprocess_cfg(config: Munch, key: str, default):
    """Read preprocess config with strict key usage."""
    pre_cfg = getattr(config, 'preprocess', Munch())
    return getattr(pre_cfg, key, default)


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    """Safely save numpy array with atomic replace semantics."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.stem}.",
        suffix='.npy',
    )
    os.close(fd)
    try:
        np.save(tmp_path, array)
        os.replace(tmp_path, str(path))
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Safely write json file with atomic replace semantics."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.stem}.",
        suffix='.json',
    )
    os.close(fd)
    try:
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=True)
        os.replace(tmp_path, str(path))
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


class HSPreprocessor(BaseEstimator, TransformerMixin):
    """
    Sklearn-API preprocessor for hyperspectral data.

    Pipeline:
    Z-score norm -> PCA dim reduction.
    Fits on FG pixels, then transforms on raw data per patient.

    Follows sklearn ``fit`` / ``transform`` API. Example::

        preprocessor = HSPreprocessor(pca_components=48)
        preprocessor.fit(foreground_pixels)         # get stats + fit.
        processed = preprocessor.transform(raw_data) # norm + transform.

    Attributes (NOTE: set after use ``fit``):
        global_mean_                  : np.ndarray, shape (n_bands,)
        global_std_                   : np.ndarray, shape (n_bands,)
        pca_                          : sklearn PCA object or None
        pca_components_               : np.ndarray (n_components, n_bands) or None
        pca_mean_                     : np.ndarray (n_bands,) or None
        pca_explained_variance_ratio_ : np.ndarray or None
        n_features_out_               : int, output channels after PCA
    """

    def __init__(self, pca_components=48, max_fit_samples=2_000_000,
                 max_pca_samples=500_000, random_state=350234):
        self.pca_components = pca_components
        self.max_fit_samples = max_fit_samples
        self.max_pca_samples = max_pca_samples
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        Fit norm and PCA stats based on raw data X.

        Args:
            X : np.ndarray, shape (n_pixels, n_bands) FG pixels.
            y : ignored (sklearn convention).

        Returns:
            self : sklearn API requires an instance return.
        """
        from sklearn.decomposition import PCA

        c = X.shape[1]

        # Subsample for stable mean/std (memory ~ max_fit * c * 4 bytes)
        if len(X) > self.max_fit_samples:
            rng_fit = np.random.RandomState(350235)
            fit_idx = rng_fit.choice(len(X), self.max_fit_samples, replace=False)
            fit_data = X[fit_idx]
        else:
            fit_data = X

        self.global_mean_ = fit_data.mean(axis=0).astype(np.float32)
        self.global_std_  = fit_data.std(axis=0).astype(np.float32) + 1e-8
        tprint(f"  norm-fit on {len(fit_data):,} pixels of {c} bands.")

        # PCA fit on Z-score normed pixels, capped at max_pca_samples
        # time complexity = O(n * C^2) where n = num_pixels and C = num_bands
        # 500k is sufficient for stable eigenvectors.
        if self.pca_components and self.pca_components < c:
            normalized_fit = (fit_data - self.global_mean_) / self.global_std_
            del fit_data

            if len(normalized_fit) > self.max_pca_samples:
                rng_pca = np.random.RandomState(42)
                pca_idx = rng_pca.choice(len(normalized_fit),
                                         self.max_pca_samples, replace=False)
                pca_fit = normalized_fit[pca_idx]
            else:
                pca_fit = normalized_fit

            self.pca_ = PCA(n_components=self.pca_components,
                            svd_solver='randomized',
                            random_state=self.random_state)
            self.pca_.fit(pca_fit)

            # explainable band ratio
            explained = sum(self.pca_.explained_variance_ratio_) * 100
            tprint(f"  pca: {c} -> {self.pca_components} channels, "
                   f"explained variance: {explained:.1f}%, "
                   f"fit on {len(pca_fit):,} pixels")

            # pre-compute fp32 projection matrix
            self.pca_components_ = self.pca_.components_.astype(np.float32)
            self.pca_mean_ = self.pca_.mean_.astype(np.float32)
            self.pca_explained_variance_ratio_ = \
                self.pca_.explained_variance_ratio_.astype(np.float32)
            self.n_features_out_ = self.pca_components

            del normalized_fit, pca_fit
        else:
            del fit_data
            self.pca_ = None
            self.pca_components_ = None
            self.pca_mean_ = None
            self.pca_explained_variance_ratio_ = None
            self.n_features_out_ = c

        return self

    def transform(self, X):
        """
        Apply fitted Z-score normalization + PCA projection.

        NOTE: manual fp32 matmul avoids sklearn's internal
        fp64 upcast, halving memory bandwidth.

        Args:
            X : np.ndarray, shape (n_pixels, n_bands) or (H, W, n_bands).

        Returns:
            np.ndarray: transformed data with reduced channels.
        """
        is_3d = (X.ndim == 3)
        if is_3d:
            h, w, c = X.shape
            flat = X.reshape(-1, c)
        else:
            flat = X

        # Z-score norm
        result = (flat - self.global_mean_) / self.global_std_

        # PCA projection using fp matmul.
        if self.pca_components_ is not None:
            result = (result - self.pca_mean_) @ self.pca_components_.T

        if is_3d:
            return result.reshape(h, w, self.n_features_out_)
        return result


class AbstractHSDataset(ABC, Dataset):
    """
    Abstract Base Class for different types of hyperspectral data.
    Inherited from ``torch.utils.data.Dataset``, ``torch.DataLoader`` compatible.

    NOTE: ``_load_data`` & ``_preprocess_data`` are abstract methods,
         which means re-implementation in inherit classes are necessary.
    """

    def __init__(self,
                 config: Munch = None,
                 transform: Optional[Any] = None,
                 **kwargs: Any) -> None:
        """
        Initialize the hyperspectral dataset.

        Args:
            config: Munch config object with dataset parameters.
            transform: Optional transform to apply to samples.
           **kwargs: Additional format-specific parameters.
        """

        _ensure_background_class(config)
        self.config = config  # store for get_dataset_info() etc.
        self.data_path = config.path.data
        self.label_path = config.path.label
        self.num = config.clsf.num
        self.targets = config.clsf.targets
        self.patch_size = int(_split_cfg(config, 'patch.size', 31, legacy_key='patch_size'))
        self.margin = (self.patch_size - 1) // 2
        self.transform = transform
        self.kwargs = kwargs
        self.test_rate = float(_split_cfg(config, 'ratio.legacy.test_rate', 0.2, legacy_key='test_rate'))

        # Core data structures
        self.raw_data: np.ndarray = None
        self.raw_labels: np.ndarray = None
        self.processed_data: np.ndarray = None
        self.patches: np.ndarray = None
        self.patch_labels: np.ndarray = None

        # Load and process data
        self._load_data()
        self._validate_raw_data()
        self._preprocess_data()
        self._create_patches()

    @abstractmethod
    def _load_data(self) -> None:
        """
        Load raw hyperspectral data and labels from files.
        
        This method must be implemented by subclasses to handle specific
        file formats. It should populate self.raw_data and self.raw_labels.
        """
        pass

    @abstractmethod
    def _preprocess_data(self) -> None:
        """
        Preprocess raw data (e.g., PCA, normalization, denoising).
        
        This method should return self.processed_data in
        shape (H, W, C) where C is the number of channels.
        """
        pass

    @staticmethod
    def _pad_with_zeros(x: np.ndarray, margin: int) -> np.ndarray:
        """
        Add zero padding to hyperspectral data.
        
        Args:
            x: Input data with shape (H, W, C)
            margin: Number of zeros to add on each side
            
        Returns:
            Padded data with shape (H + 2*margin, W + 2*margin, C)
        """
        if margin < 0:
            raise ValueError(f"Margin must be non-negative, got {margin}")
            
        padded = np.zeros(
            (x.shape[0] + 2 * margin, 
             x.shape[1] + 2 * margin, 
             x.shape[2]),
            dtype=x.dtype
        )
        padded[margin:-margin, margin:-margin, :] = x
        return padded

    def _validate_raw_data(self) -> None:
        """
        Validate the loaded raw data and labels.
        
        Ensures consistent shape alignment and valid label range.
        
        OPTIMIZATION: Chunked finite-value check avoids allocating a full
        boolean array the size of raw_data.
        """
        tprint("Validating raw data...")
        # Check data and label spatial dimensions match
        print("Validating spatial shape...")
        if self.raw_data.shape[:2] != self.raw_labels.shape[:2]:
            raise ValueError(
                f"Spatial dimensions mismatch: data {self.raw_data.shape[:2]}, "
                f"labels {self.raw_labels.shape[:2]}"
            )
        print("Shape validated!")
            
        # Check for non-finite values — chunked to avoid huge intermediate array
        print("Validating raw values...")
        flat = self.raw_data.reshape(-1)
        chunk = 1_000_000
        for i in range(0, flat.size, chunk):
            if not np.isfinite(flat[i:i + chunk]).all():
                raise ValueError("Raw data contains non-finite values (NaN/inf)")
        print("Value validated!")
            
        # Validate label range
        print("Validating raw label...")
        unique_labels = np.unique(self.raw_labels)
        if not np.all((unique_labels >= 0) & (unique_labels < self.num)):
            raise ValueError(
                f"Labels contain values outside valid range [0, {self.num - 1}]"
            )
        print("Raw label validated!")

    def _create_patches(self) -> None:
        """
        Extract spatial patches from preprocessed data.
        
        Patches are extracted with zero-padding at image boundaries.
        All center pixels are included, including background label 0.
        
        OPTIMIZATION: Padding is done once and cached to avoid re-padding for each sample.
        OPTIMIZATION: Vectorized with np.where instead of nested Python for-loop.
        """
        # Add zero padding around the image (ONCE at initialization)
        self.padded_data = self._pad_with_zeros(self.processed_data, self.margin)
        
        # Vectorized: include all center pixels (background included)
        rows, cols = np.indices(self.raw_labels.shape)
        rows = rows.reshape(-1)
        cols = cols.reshape(-1)
        
        # Convert to padded coordinates (offset by margin)
        self.patch_indices = np.stack(
            [rows + self.margin, cols + self.margin], axis=1
        ).astype(np.int32)
        
        # Labels are kept as-is: 0=background, 1..K=foreground classes.
        self.patch_labels = self.raw_labels[rows, cols].astype(np.int32)
        tprint(f"Created {len(self.patch_indices)} indices for patch.")
        print(f"Cached padded data shape: {self.padded_data.shape} (padding happens once at init)")

        # Pad labels for dense segmentation
        self.padded_labels = np.full(
            (self.raw_labels.shape[0] + 2 * self.margin,
             self.raw_labels.shape[1] + 2 * self.margin),
            fill_value=255, dtype=np.int32
        )
        # Keep background as a valid class; only padding remains ignore_index=255.
        label_region = self.raw_labels.copy().astype(np.int32)
        self.padded_labels[self.margin:self.margin + self.raw_labels.shape[0],
                           self.margin:self.margin + self.raw_labels.shape[1]] = label_region
        tprint(f"Created padded label map: {self.padded_labels.shape}")

    def _get_patch_(self, idx: int) -> np.ndarray:
        """
        Get a single patch from the cached padded data.
        
        OPTIMIZATION: Uses pre-computed padded_data instead of re-padding for each sample.
        This reduces per-sample overhead from O(H*W*C) to O(patch_size²*C).
        """
        r, c = self.patch_indices[idx]
        
        # Extract patch from pre-computed padded_data
        patch = self.padded_data[
            r - self.margin : r + self.margin + 1,
            c - self.margin : c + self.margin + 1,
            :
        ]
        return patch.copy()

    def _get_label_patch_(self, idx: int) -> np.ndarray:
        """
        Get the dense label patch for segmentation mode.
        Returns label map of shape (patch_size, patch_size) where
        background/padding pixels have value 255 (ignore_index).
        """
        r, c = self.patch_indices[idx]
        label_patch = self.padded_labels[
            r - self.margin : r + self.margin + 1,
            c - self.margin : c + self.margin + 1
        ]
        return label_patch.copy()

    # Magic methods.
    def __len__(self) -> int:
        """Return number of patches in dataset"""
        return len(self.patch_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            - patch: Tensor of shape (C, H, W)
            - label: Tensor of shape (H, W) with per-pixel class indices
                     (255 = ignore / background)
        """
        patch = self._get_patch_(idx)
        
        # Convert to (C, H, W) format for PyTorch
        patch = np.transpose(patch, (2, 0, 1))
        
        if self.transform:
            patch = self.transform(patch)
        
        # Always return dense per-pixel labels
        label_patch = self._get_label_patch_(idx)  # [H, W]
        return torch.FloatTensor(patch), torch.LongTensor(label_patch)

class NpyHSDataset(AbstractHSDataset):
    """
    Hyperspectral dataloader for .npy file format.
    
    There are a bunch of sample couple of patients in dir, in the form of:
    - data/processed_npy_64_norm/
        - Patient1_20250213_LU.npy
        - Patient1_20250213_LU_gt.npy
        - Patient1_20250213_RD.npy
        - Patient1_20250213_RD_gt.npy
        - Patient2_20250213_LU.npy
        - Patient2_20250213_LU_gt.npy
        - ...
        Each patient may conclude many directions, LU(left-upper), RD, RU, LD, RUD etc.
        Each direction of each patient has a pair of .npy files: one for hyperspectral data and one for labels.
        Read them in pairs, and create a dataset for each pair. Then concatenate all datasets together to form the final dataset.
    """
    def __init__(self, 
                 config: Munch = None,
                 transform: Optional[Any] = None,
                 limit_pairs: Optional[int] = None,
                 max_patches_per_patient: Optional[int] = None,
                 debug_mode: bool = False,
                  **kwargs: Any) -> None:
        """
        per-patient pipeline: 
        loads each patient independently,
        to avoid cross-patient patch contamination and zero-padding noise in pca.

        bypasses AbstractHSDataset.__init__ which concatenates all patients
        into one image, causing boundary artifacts and polluted statistics.
        """
        Dataset.__init__(self)

        _ensure_background_class(config)
        self.config = config
        self.data_path = config.path.data
        self.label_path = config.path.label
        self.num = config.clsf.num
        self.targets = config.clsf.targets
        self.patch_size = int(_split_cfg(config, 'patch.size', 31, legacy_key='patch_size'))
        self.margin = (self.patch_size - 1) // 2
        self.transform = transform
        self.kwargs = kwargs
        self.test_rate = float(_split_cfg(config, 'ratio.legacy.test_rate', 0.2, legacy_key='test_rate'))
        
        # Debug-friendly controls
        self.limit_pairs = limit_pairs  # only load first N pairs when set
        self.max_patches_per_patient = max_patches_per_patient  # cap patches per patient
        self.debug_mode = debug_mode

        # Cached stats for downstream analysis
        self.global_mean: Optional[np.ndarray] = None
        self.global_std: Optional[np.ndarray] = None
        self.pca_components: Optional[np.ndarray] = None
        self.pca_mean: Optional[np.ndarray] = None
        self.pca_explained_variance: Optional[np.ndarray] = None
        self.feature_dim: Optional[int] = None
        self._label_remap: Optional[np.ndarray] = None
        self.patch_pg_tg_boundary: Optional[np.ndarray] = None

        self._use_cached_split_pipeline = bool(
            getattr(self.config.preprocess, 'enable_split_cache_pipeline', True)
        )

        # try load cached data if enabled
        if self._use_cached_split_pipeline:
            with self._dataset_build_manager():
                self._build_or_load_split_cache()
        
        # otherwise, build from scratch
        else:
            # Legacy fallback pipeline
            with self._dataset_build_manager():  
                self._per_patient_pipeline()

    @contextmanager
    def _dataset_build_manager(self):
        """
        Context manager for dataset build lifecycle.
        Stream dataset processing and reports timing.
        """
        tic = time.perf_counter()
        tprint("[dataset_build_manager] start building dataset")
        try:
            yield
        finally:
            toc = time.perf_counter()
            n_patches = len(getattr(self, 'patch_indices', []))
            tprint(f"[dataset_build_manager] done in {toc - tic:.2f}s, "
                   f"patches: {n_patches:,}")
    
    @staticmethod
    def _extract_patient_id(filepath: str) -> str:
        """
        Extract patient ID from filename.
        
        Filename pattern: 'PatientName_YYYYMMDD_Direction.npy'
        Returns normalized (lowercase) patient name to handle inconsistent casing.
        """
        basename = os.path.basename(filepath)
        match = re.match(r'^(.+?)_(\d{8})_', basename)
        if match:
            return match.group(1).lower()
        # Fallback: use basename without extension
        return os.path.splitext(basename)[0].lower()

    def _augment_pixels(self,
                                    pixels: np.ndarray,
                                    rng: np.random.RandomState) -> np.ndarray:
        """Apply space spectral augmentation on pixels."""
        if pixels.size == 0:
            return pixels
        if not bool(_preprocess_cfg(self.config.augment, 'augment_train', True)):
            return pixels

        x = pixels.astype(np.float32, copy=True)
        n_bands = x.shape[1]

        scale_prob = float(_preprocess_cfg(self.config.augment.spectral, 'scale_prob', 0.2))
        noise_prob = float(_preprocess_cfg(self.config.augment.spectral, 'noise_prob', 0.2))
        band_jitter_prob = float(_preprocess_cfg(self.config.augment.spectral, 'band_jitter_prob', 0.2))
        noise_std = float(_preprocess_cfg(self.config.augment.spectral, 'noise_std', 0.01))

        if rng.rand() < scale_prob:
            gain = rng.uniform(0.92, 1.08, size=(1, n_bands)).astype(np.float32)
            x *= gain

        if rng.rand() < noise_prob and noise_std > 0:
            x += rng.normal(0.0, noise_std, size=x.shape).astype(np.float32)

        # Apply smooth jitter to contiguous bands.
        if rng.rand() < band_jitter_prob and n_bands >= 4:
            width = int(max(2, round(n_bands * float(_preprocess_cfg(self.config.augment.spectral, 'band_jitter_width', 0.15)))))
            width = min(width, n_bands)
            start = int(rng.randint(0, n_bands - width + 1))
            scale = float(rng.uniform(0.85, 1.15))
            x[:, start:start + width] *= scale

        return x

    def _augment_sample(self,
                                data: np.ndarray,
                                labels: np.ndarray,
                                rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
        """Apply spatial + spectral augmentation."""
        if data.size == 0:
            return data, labels
        if not bool(_preprocess_cfg(self.config.augment, 'augment_train', True)):
            return data, labels

        x = data.astype(np.float32, copy=True)
        y = labels.astype(np.int32, copy=True)

        # Spatial transforms are moved before PCA to keep label-data alignment.
        spatial_prob = float(_preprocess_cfg(self.config.augment.spatial, 'spatial_prob', 0.5))
        rotate_prob = float(_preprocess_cfg(self.config.augment.spatial, 'rotate_prob', 0.35))
        if rng.rand() < spatial_prob:
            if rng.rand() > 0.5:
                x = np.flip(x, axis=1).copy()
                y = np.flip(y, axis=1).copy()
            if rng.rand() > 0.5:
                x = np.flip(x, axis=0).copy()
                y = np.flip(y, axis=0).copy()
        if rng.rand() < rotate_prob:
            k = int(rng.choice([1, 2, 3]))
            x = np.rot90(x, k, axes=(0, 1)).copy()
            y = np.rot90(y, k, axes=(0, 1)).copy()

        # spectral transforms
        flat = x.reshape(-1, x.shape[2])
        flat = self._augment_pixels(flat, rng)
        x = flat.reshape(x.shape).astype(np.float32, copy=False)

        # spectral band drop
        band_drop_rate = float(_preprocess_cfg(self.config.augment.spectral, 'band_drop_rate', 0.02))
        if band_drop_rate > 0 and rng.rand() > 0.5:
            n_bands = x.shape[2]
            n_mod = max(1, int(n_bands * band_drop_rate))
            mod_idx = rng.choice(n_bands, n_mod, replace=False)
            scales = rng.uniform(0.7, 1.0, size=(1, 1, n_mod)).astype(np.float32)
            x[:, :, mod_idx] *= scales

        # spatial random cutout
        cutout_ratio = float(_preprocess_cfg(self.config.augment.spatial, 'cutout_ratio', 0.08))
        cutout_prob = float(_preprocess_cfg(self.config.augment.spatial, 'cutout_prob', 0.05))
        if cutout_ratio > 0 and rng.rand() < cutout_prob:
            h, w = x.shape[:2]
            ch = max(1, int(h * cutout_ratio))
            cw = max(1, int(w * cutout_ratio))
            y0 = int(rng.randint(0, h - ch + 1))
            x0 = int(rng.randint(0, w - cw + 1))
            x[y0:y0 + ch, x0:x0 + cw, :] = 0.0

        return x, y

    @staticmethod
    def _compute_pair_boundary_map(labels: np.ndarray,
                                   cls_a: int,
                                   cls_b: int) -> np.ndarray:
        """Compute pairwise boundary map between two classes using 8-neighbors."""
        a_mask = (labels == cls_a)
        b_mask = (labels == cls_b)
        boundary_map = np.zeros_like(labels, dtype=np.bool_)
        for dr, dc in [(-1, -1), (-1, 0), (-1, 1),
                       (0, -1),           (0, 1),
                       (1, -1),  (1, 0),  (1, 1)]:
            a_nb = np.roll(a_mask, shift=(dr, dc), axis=(0, 1))
            b_nb = np.roll(b_mask, shift=(dr, dc), axis=(0, 1))
            boundary_map |= (a_mask & b_nb) | (b_mask & a_nb)
        boundary_map[0, :] = False
        boundary_map[-1, :] = False
        boundary_map[:, 0] = False
        boundary_map[:, -1] = False
        return boundary_map

    @staticmethod
    def _compute_multiclass_boundary_map(labels: np.ndarray) -> np.ndarray:
        """Compute generic class boundaries for all classes using Sobel + finite-diff."""
        lbl = labels.astype(np.float32, copy=False)
        gx = cv2.Sobel(lbl, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(lbl, cv2.CV_32F, 0, 1, ksize=3)
        sobel_edge = (np.abs(gx) + np.abs(gy)) > 0

        diff_edge = np.zeros_like(labels, dtype=np.bool_)
        diff_edge[:, 1:] |= (labels[:, 1:] != labels[:, :-1])
        diff_edge[1:, :] |= (labels[1:, :] != labels[:-1, :])

        boundary_map = sobel_edge | diff_edge
        boundary_map[0, :] = False
        boundary_map[-1, :] = False
        boundary_map[:, 0] = False
        boundary_map[:, -1] = False
        return boundary_map

    def _get_cache_roots(self) -> Tuple[Path, Path]:
        data_root = Path(getattr(self.config.path, 'data', self.data_path))
        label_root = Path(getattr(self.config.path, 'label', self.label_path))
        return data_root, label_root

    def _config_fingerprint(self) -> str:
        """return a hash of current config.yaml for cache validation."""
        tracked = {
            'split_seed': int(getattr(self.config.split, 'split_seed', 350234)),
            'train_ratio': float(getattr(self.config.split, 'train_ratio', 0.8)),
            'val_ratio': float(getattr(self.config.split, 'val_ratio', 0.1)),
            'test_ratio': float(getattr(self.config.split, 'test_ratio', 0.1)),
            'patch_size': int(getattr(self.config.split, 'patch_size', 31)),
            'pca_components': int(getattr(self.config.preprocess, 'pca_components', 48)),
            'label_remap': list(getattr(self.config.clsf, 'label_remap', [])),
        }
        txt = json.dumps(tracked, sort_keys=True, ensure_ascii=True)
        return hashlib.md5(txt.encode('utf-8')).hexdigest()

    def _build_or_load_split_cache(self) -> None:
        cache_root_data, cache_root_label = self._get_cache_roots()
        cache_root_data.mkdir(parents=True, exist_ok=True)
        cache_root_label.mkdir(parents=True, exist_ok=True)

        for split_name in ('train', 'val', 'test'):
            (cache_root_data / split_name).mkdir(parents=True, exist_ok=True)
            (cache_root_label / split_name).mkdir(parents=True, exist_ok=True)

        force_rebuild = bool(getattr(self.config.preprocess, 'force_rebuild_split_cache', False))
        signature_file = cache_root_data / 'split_cache_signature.json'
        stats_file = cache_root_data / 'preprocess_stats.npz'
        manifest_files = {
            'train': cache_root_data / 'train' / 'manifest.json',
            'val': cache_root_data / 'val' / 'manifest.json',
            'test': cache_root_data / 'test' / 'manifest.json',
        }
        
        # _config_fingerprint() gets a hash from current config.yaml.
        cfg_hash = self._config_fingerprint()

        def _cache_ready() -> bool:
            """Check if the cached data is ok for use."""
            if not signature_file.exists() or not stats_file.exists():
                return False
            if not all(p.exists() for p in manifest_files.values()):
                return False
            try:
                sig = json.loads(signature_file.read_text(encoding='utf-8'))
                return sig.get('config_hash') == cfg_hash
            except Exception:
                return False

        # stream process per patient pipeline
        # implement of the context manager at line 1668
        # this manager includes patient split, preprocessor fit and apply.
        with _SplitPreprocessManager(self, cache_root_data, cache_root_label) as manager:
            if force_rebuild or (not _cache_ready()):
                tprint('Building split cache: patient split -> train-only fit -> split-wise preprocess save')
                manager.build()
                _atomic_write_json(signature_file, {'config_hash': cfg_hash})
            else:
                tprint('Using existing split cache under data root.')

        # load dataset stats for downstream analysis
        stats = np.load(stats_file)
        self.global_mean = stats['global_mean'].astype(np.float32)
        self.global_std = stats['global_std'].astype(np.float32)
        pca_components = stats['pca_components']
        self.pca_components = pca_components.astype(np.float32) if pca_components.size > 0 else None
        pca_mean = stats['pca_mean']
        self.pca_mean = pca_mean.astype(np.float32) if pca_mean.size > 0 else None
        pca_explained = stats['pca_explained']
        self.pca_explained_variance = pca_explained.astype(np.float32) if pca_explained.size > 0 else None
        self.feature_dim = int(stats['feature_dim'][0])

        self._split_manifests = {}
        self._split_class_counts = {}
        total_patches = 0
        train_patches = 0
        self._num_patients = 0

        # load split manifests to get class & patient stats for balanced sampling.
        for split_name, mfile in manifest_files.items():
            with open(mfile, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            self._split_manifests[split_name] = manifest
            cls_counts = np.asarray(manifest.get('class_counts', []), dtype=np.int64)
            if cls_counts.size == 0:
                cls_counts = np.zeros(self.num, dtype=np.int64)
            self._split_class_counts[split_name] = cls_counts
            split_patch_count = int(sum(int(item['height']) * int(item['width']) for item in manifest.get('samples', [])))
            total_patches += split_patch_count
            if split_name == 'train':
                train_patches = split_patch_count

            patients = {str(item.get('patient_id', '')) for item in manifest.get('samples', [])}
            self._num_patients += len([p for p in patients if p])

        self._num_patients = int(self._num_patients)
        self._cached_total_patches = int(total_patches)
        self._cached_train_patches = int(train_patches)

        # placeholders for trainer utility paths.
        self.patch_indices = np.empty((0, 3), dtype=np.int32)
        self.patch_labels = np.empty((0,), dtype=np.int32)
        self.patch_patient_groups = np.empty((0,), dtype=np.int32)

    def get_train_class_counts(self) -> np.ndarray:
        return np.asarray(self._split_class_counts.get('train', np.zeros(self.num, dtype=np.int64)), dtype=np.int64)

    def _build_label_remap(self) -> np.ndarray:
        """Build label remap table from config, defaulting to identity mapping.

                Supported config forms under clsf.label_remap:
                - list form: [0, 1, 2, 3] where index is raw label and value is mapped label.
                    If the list only covers foreground classes, e.g. [1, 2, 3], a BG=0
                    slot is prepended automatically.
                - dict form: {0: 0, 1: 1, 2: 2, 3: 3}

        If unset, defaults to identity for labels [0, clsf.num - 1].
        """
        remap_cfg = getattr(self.config.clsf, 'label_remap', None)

        if remap_cfg is None:
            return np.arange(max(int(self.num), 1), dtype=np.int32)

        if isinstance(remap_cfg, (dict, Munch)):
            remap_items = dict(remap_cfg)
            if not remap_items:
                raise ValueError("clsf.label_remap cannot be an empty dict")

            normalized: Dict[int, int] = {}
            for raw_k, mapped_v in remap_items.items():
                try:
                    k = int(raw_k)
                    v = int(mapped_v)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"clsf.label_remap dict must be int->int, got {raw_k}:{mapped_v}"
                    ) from None
                if k < 0 or v < 0:
                    raise ValueError(
                        f"clsf.label_remap entries must be non-negative, got {k}->{v}"
                    )
                normalized[k] = v

            max_key = max(normalized.keys())
            remap = np.zeros(max_key + 1, dtype=np.int32)
            remap[0] = 0
            for k, v in normalized.items():
                remap[k] = v
        else:
            try:
                remap = np.asarray(remap_cfg, dtype=np.int32).reshape(-1)
            except Exception as exc:
                raise ValueError(
                    "clsf.label_remap must be a 1D list/array or a dict"
                ) from exc

            if remap.size == 0:
                raise ValueError("clsf.label_remap cannot be empty")
            if np.any(remap < 0):
                raise ValueError("clsf.label_remap values must be non-negative")

            remap = remap.astype(np.int32, copy=False)
            if remap.size == max(int(self.num) - 1, 0):
                remap = np.concatenate([np.array([0], dtype=np.int32), remap])

        if np.max(remap) > self.num:
            raise ValueError(
                f"clsf.label_remap contains mapped label > clsf.num ({self.num}): "
                f"max mapped is {int(np.max(remap))}"
            )

        return remap

    def _apply_label_remap(self, labels_raw: np.ndarray,
                           remap: np.ndarray,
                           label_file: str) -> np.ndarray:
        """Apply configurable label remap with strict/soft unknown handling."""
        if labels_raw.ndim != 2:
            raise ValueError(
                f"label map must be 2D, got shape {labels_raw.shape} from {label_file}"
            )

        strict_remap = bool(getattr(self.config.clsf, 'strict_label_remap', True))
        unknown_policy = str(
            getattr(self.config.clsf, 'unknown_label_policy', 'error')
        ).strip().lower()
        if unknown_policy not in {'error', 'map_to_bg'}:
            raise ValueError(
                "clsf.unknown_label_policy must be 'error' or 'map_to_bg'"
            )

        min_raw = int(labels_raw.min())
        max_raw = int(labels_raw.max())
        if min_raw < 0:
            raise ValueError(
                f"{os.path.basename(label_file)} has negative label {min_raw}"
            )

        unknown_mask = labels_raw >= len(remap)
        unknown_count = int(unknown_mask.sum())
        if unknown_count > 0:
            unknown_values = np.unique(labels_raw[unknown_mask])
            preview = unknown_values[:10].tolist()
            suffix = "" if len(unknown_values) <= 10 else " ..."
            msg = (
                f"{os.path.basename(label_file)} contains {unknown_count:,} unknown label pixel(s), "
                f"values={preview}{suffix}, remap_max={len(remap)-1}"
            )
            if strict_remap or unknown_policy == 'error':
                raise ValueError(msg)
            print(f"\t\tWARNING: {msg}; mapping unknown labels to BG(0)")

        labels_clipped = np.clip(labels_raw, 0, len(remap) - 1)
        labels = remap[labels_clipped]

        if unknown_count > 0 and unknown_policy == 'map_to_bg':
            labels[unknown_mask] = 0

        if max_raw > len(remap) and not (unknown_count > 0 and unknown_policy == 'map_to_bg'):
            print(f"\t\tWARNING: {os.path.basename(label_file)} has max raw label {max_raw} "
                  f"over remap max {len(remap)-1}")

        return labels.astype(np.int32, copy=False)

    def _collect_pixels_for_preprocessor(self,
                                         data: np.ndarray,
                                         labels: np.ndarray,
                                         rng: np.random.RandomState) -> np.ndarray:
        """Collect pixels for norm/PCA fitting."""
        flat_data = data.reshape(-1, data.shape[2])
        flat_labels = labels.reshape(-1)

        fg_pixels = flat_data[flat_labels > 0]
        bg_pixels = flat_data[flat_labels == 0]
        if len(bg_pixels) == 0:
            return fg_pixels
        if len(fg_pixels) == 0:
            return bg_pixels

        max_bg = int(max(1, round(len(fg_pixels))))
        n_bg = min(len(bg_pixels), max_bg)
        if n_bg < len(bg_pixels):
            sel = rng.choice(len(bg_pixels), size=n_bg, replace=False)
            bg_pixels = bg_pixels[sel]

        return np.concatenate([fg_pixels, bg_pixels], axis=0)

    def _resolve_boundary_class_pair(self) -> Optional[Tuple[int, int]]:
        """
        Resolve boundary-pair class ids from class names in config.
        This helps with distiguishing hard-classification targets,
        which is defined in config.yaml: config.split.boundary_pair.
        """
        pair_cfg = getattr(self.config.split, 'boundary_pair', None)
        if pair_cfg is None:
            return None
        if not isinstance(pair_cfg, (list, tuple)) or len(pair_cfg) != 2:
            tprint("WARNING: split.boundary_pair must be a list of two class names; pair boost disabled")
            return None

        targets = list(self.targets)
        a_name = str(pair_cfg[0]).strip()
        b_name = str(pair_cfg[1]).strip()
        if a_name not in targets or b_name not in targets:
            tprint(f"WARNING: boundary_pair {pair_cfg} not found in targets {targets}; boundary boost disabled")
            return None

        a_idx = int(targets.index(a_name))
        b_idx = int(targets.index(b_name))
        if a_idx == b_idx:
            tprint("WARNING: boundary_pair contains duplicate classes; pair boost disabled")
            return None

        tprint(f"  boundary hard-mining pair: {a_name}({a_idx}) <-> {b_name}({b_idx})")
        return (a_idx, b_idx)

    def _cap_patient_centers(self,
                             rows: np.ndarray,
                             cols: np.ndarray,
                             labels: np.ndarray,
                             boundary_map: np.ndarray,
                             rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cap per-patient centers,
        while keeping minimum boundary/FG coverage.
        """
        if self.max_patches_per_patient is None:
            return rows, cols

        cap = int(self.max_patches_per_patient)
        n = int(rows.size)
        if cap <= 0 or n <= cap:
            return rows, cols

        flat_labels = labels.reshape(-1)
        flat_boundary = boundary_map.reshape(-1)

        min_boundary = int(max(0, getattr(self.config.split, 'min_boundary_centers_per_patient', 0)))
        min_fg = int(max(0, getattr(self.config.split, 'min_fg_centers_per_patient', 0)))

        selected_mask = np.zeros(n, dtype=np.bool_)
        selected = []

        boundary_pool = np.flatnonzero(flat_boundary)
        if boundary_pool.size > 0 and min_boundary > 0:
            take = int(min(min_boundary, cap, boundary_pool.size))
            b_ids = rng.choice(boundary_pool, size=take, replace=False)
            selected.append(b_ids)
            selected_mask[b_ids] = True

        remain = cap - int(selected_mask.sum())
        if remain > 0 and min_fg > 0:
            fg_pool = np.flatnonzero((flat_labels > 0) & (~selected_mask))
            if fg_pool.size > 0:
                take = int(min(min_fg, remain, fg_pool.size))
                fg_ids = rng.choice(fg_pool, size=take, replace=False)
                selected.append(fg_ids)
                selected_mask[fg_ids] = True
                remain = cap - int(selected_mask.sum())

        if remain > 0:
            rest_pool = np.flatnonzero(~selected_mask)
            rest_ids = rng.choice(rest_pool, size=remain, replace=False)
            selected.append(rest_ids)

        sel = np.concatenate(selected, axis=0) if selected else np.empty((0,), dtype=np.int64)
        if sel.size > 1:
            rng.shuffle(sel)
        return rows[sel], cols[sel]
    
    def _per_patient_pipeline(self) -> None:
        """
        load, normalize, pca, and extract patches per-patient independently.

        pipeline:
                    - load all patients, collect foreground/class-0 pixels for global stats
                    - fit global normalization + pca on selected training pixels
          - per-patient: normalize -> pca -> pad -> extract patches
          - concatenate patches from all patients
          
        2026.2.25 fixed two critical bugs in the old concatenation approach:
          1. patches at patient boundaries mixed tissues from different patients
          2. zero-padding from width alignment polluted normalization/pca stats
          
        2026.2.26 fixed:
          1. label remapping for merged classes (LN+Blood -> LQ)
          2. added robust validation for raw data and labels (shape, finite values, label range)
        """
        
        pairs = self._pair_data_and_labels_()
        if self.limit_pairs is not None:
            pairs = pairs[: self.limit_pairs]
            tprint(f"  DEBUG limit_pairs={self.limit_pairs}: using first {len(pairs)} pair(s)")
        if not pairs:
            raise RuntimeError("no data/label pairs found in " + self.data_path)

        # label remapping from config (soft-coded), defaulting to identity.
        _label_remap = self._build_label_remap()
        self._label_remap = _label_remap

        tprint("loading patients...")
        patient_records = []
        patient_fit_pixels: Dict[int, List[np.ndarray]] = {}
        _patient_id_map = {}
        rng_fit = np.random.RandomState(350236)
        boundary_pair = self._resolve_boundary_class_pair()
        split_seed = int(getattr(self.config.split, 'split_seed', 350234))
        test_rate = float(getattr(self.config.split, 'test_rate', 0.2))
        val_rate = float(getattr(self.config.split, 'val_rate', 0.1))
        targets = list(getattr(self.config.clsf, 'targets', []))
        pg_idx = targets.index('PG') if 'PG' in targets else 1
        tg_idx = targets.index('TG') if 'TG' in targets else max(1, self.num - 1)

        patient_fg_cnt: Dict[int, int] = {}
        patient_total_cnt: Dict[int, int] = {}
        patient_pg_cnt: Dict[int, int] = {}
        patient_tg_cnt: Dict[int, int] = {}
        unknown_pixels_total = 0
        unknown_pixels_all = 0

        for idx, (data_file, label_file) in enumerate(pairs):
            data = np.load(data_file).astype(np.float32)    # shape: (h_i, w_i, c)
            labels_raw = np.load(label_file).astype(np.int32)  # shape: (h_i, w_i), 1-based

            if data.shape[:2] != labels_raw.shape[:2]:
                raise ValueError(
                    f"shape mismatch: {data_file} {data.shape[:2]} "
                    f"vs {labels_raw.shape[:2]}")

            if not np.isfinite(data).all():
                raise ValueError(f"non-finite values in {data_file}")

            unknown_mask = labels_raw > len(_label_remap)
            unknown_pixels_total += int(unknown_mask.sum())
            unknown_pixels_all += int(labels_raw.size)
            labels = self._apply_label_remap(labels_raw, _label_remap, label_file)

            patient_id = self._extract_patient_id(data_file)
            if patient_id not in _patient_id_map:
                _patient_id_map[patient_id] = len(_patient_id_map)
            pid_idx = _patient_id_map[patient_id]

            patient_records.append((data, labels, pid_idx))

            # collect pixels for global stats; strategy can be FG-only or FG+sampled class-0
            fit_pixels = self._collect_pixels_for_preprocessor(data, labels, rng_fit)
            patient_fit_pixels.setdefault(pid_idx, []).append(fit_pixels)
            mask = labels.reshape(-1) > 0

            flat_labels = labels.reshape(-1)
            patient_fg_cnt[pid_idx] = int(patient_fg_cnt.get(pid_idx, 0) + int((flat_labels > 0).sum()))
            patient_total_cnt[pid_idx] = int(patient_total_cnt.get(pid_idx, 0) + int(flat_labels.size))
            patient_pg_cnt[pid_idx] = int(patient_pg_cnt.get(pid_idx, 0) + int((flat_labels == pg_idx).sum()))
            patient_tg_cnt[pid_idx] = int(patient_tg_cnt.get(pid_idx, 0) + int((flat_labels == tg_idx).sum()))

            tprint(f"  loaded {idx+1}/{len(pairs)}: "
                   f"{os.path.basename(data_file)} ({patient_id}), "
                   f"shape {data.shape}, possesses labels: {np.unique(labels)}, "
                   f"max label {labels.max()}, min label {labels.min()}"
                   )

        self._patient_names = {v: k for k, v in _patient_id_map.items()}
        self._num_patients = len(_patient_id_map)
        tprint(f"  {len(pairs)} pairs, {self._num_patients} unique patients")
        if unknown_pixels_total > 0:
            unknown_ratio = 100.0 * float(unknown_pixels_total) / float(max(unknown_pixels_all, 1))
            tprint(
                f"WARNING: unknown labels mapped to BG in build pipeline: "
                f"{unknown_pixels_total:,}/{unknown_pixels_all:,} ({unknown_ratio:.3f}%)"
            )

        # Build patient-level split before fitting preprocessor to avoid train/val/test leakage.
        patient_ids = np.asarray(sorted(self._patient_names.keys()), dtype=np.int64)
        train_patient_ids: np.ndarray
        val_patient_ids: np.ndarray
        test_patient_ids: np.ndarray

        if patient_ids.size < 3:
            train_patient_ids = patient_ids.copy()
            val_patient_ids = np.empty((0,), dtype=np.int64)
            test_patient_ids = np.empty((0,), dtype=np.int64)
            tprint("  WARNING: <3 patients; using all patients for train-only preprocessing fit")
        else:
            fg_ratio = np.asarray([
                float(patient_fg_cnt.get(int(pid), 0)) / float(max(patient_total_cnt.get(int(pid), 1), 1))
                for pid in patient_ids
            ], dtype=np.float64)
            pg_tg_ratio = np.asarray([
                float(patient_pg_cnt.get(int(pid), 0) + 1.0) / float(patient_tg_cnt.get(int(pid), 0) + 1.0)
                for pid in patient_ids
            ], dtype=np.float64)
            fg_bins = np.digitize(fg_ratio, np.quantile(fg_ratio, [0.33, 0.66]), right=False)
            pgtg_bins = np.digitize(pg_tg_ratio, np.quantile(pg_tg_ratio, [0.33, 0.66]), right=False)
            strata = fg_bins * 3 + pgtg_bins

            try:
                trainval_patients, test_patient_ids = train_test_split(
                    patient_ids,
                    test_size=test_rate,
                    random_state=split_seed,
                    stratify=strata,
                )
            except Exception:
                trainval_patients, test_patient_ids = train_test_split(
                    patient_ids,
                    test_size=test_rate,
                    random_state=split_seed,
                    stratify=None,
                )

            strata_map = {int(pid): int(s) for pid, s in zip(patient_ids, strata)}
            trainval_strata = np.asarray([strata_map[int(pid)] for pid in trainval_patients], dtype=np.int64)
            try:
                train_patient_ids, val_patient_ids = train_test_split(
                    trainval_patients,
                    test_size=val_rate,
                    random_state=split_seed,
                    stratify=trainval_strata,
                )
            except Exception:
                train_patient_ids, val_patient_ids = train_test_split(
                    trainval_patients,
                    test_size=val_rate,
                    random_state=split_seed,
                    stratify=None,
                )

        self._patient_split_plan = {
            'train': set(int(x) for x in np.asarray(train_patient_ids, dtype=np.int64).tolist()),
            'val': set(int(x) for x in np.asarray(val_patient_ids, dtype=np.int64).tolist()),
            'test': set(int(x) for x in np.asarray(test_patient_ids, dtype=np.int64).tolist()),
        }

        # norm + pca on selected training pixels.
        # select partial for fitting, speed up pca.
        tprint("fitting norm + pca on train set...")
        all_real_pixels = []
        for pid in self._patient_split_plan['train']:
            all_real_pixels.extend(patient_fit_pixels.get(int(pid), []))
        if not all_real_pixels:
            raise RuntimeError("No train pixels available for preprocessor fit")
        all_real = np.concatenate(all_real_pixels, axis=0)   # shape: (n_total_real, c)
        del all_real_pixels, patient_fit_pixels
        c = all_real.shape[1]

        rng = np.random.RandomState(split_seed + 17)
        all_real = self._augment_pixels(all_real, rng)

        n_components = self.config.preprocess.pca_components
        self.preprocessor = HSPreprocessor(
            pca_components=n_components,
            max_fit_samples=self.config.preprocess.max_fit_samples,
            max_pca_samples=self.config.preprocess.max_pca_samples,
            random_state=int(_split_cfg(self.config, 'seed', 350234, legacy_key='split_seed')),
        )
        self.preprocessor.fit(all_real)
        del all_real

        # Backward-compatible attribute aliases (point to fitted preprocessor)
        self.global_mean = self.preprocessor.global_mean_
        self.global_std = self.preprocessor.global_std_
        self.pca_components = self.preprocessor.pca_components_
        self.pca_mean = self.preprocessor.pca_mean_
        self.pca_explained_variance = self.preprocessor.pca_explained_variance_ratio_
        self.feature_dim = self.preprocessor.n_features_out_
        c_out = self.feature_dim

        # patient norm -> pca -> pad -> extract patches
        # patient padded independently, no cross-patient boundary bleeding
        tprint("patient patching...")
        self._patient_padded_data   = []  # list of padded arrays per patient
        self._patient_padded_labels = []  # list of padded label maps per patient

        all_patch_indices   = []  # each row: (patient_list_idx, row_padded, col_padded)
        all_patch_labels    = []  # 0-based center-pixel label
        all_patient_groups  = []  # patient group id for each patch
        all_pg_tg_boundary  = []  # bool flag: center is on PG<->TG boundary
        all_fg_roi_flags    = []  # bool flag: center within dilated foreground ROI

        for data, labels, pid_idx in patient_records:
            h, w, c_raw = data.shape                     # per-patient: (h_i, w_i, c)

            if int(pid_idx) in self._patient_split_plan['train']:
                aug_rng = np.random.RandomState(split_seed + int(pid_idx) * 9973 + h * 31 + w)
                data, labels = self._augment_sample(data, labels, aug_rng)

            # apply fitted preprocessor (Z-score + PCA) via sklearn transform API
            processed = self.preprocessor.transform(data) # shape: (h_i, w_i, c_out)

            padded_data = self._pad_with_zeros(processed, self.margin)
            # padded shape: (h_i + 2*margin, w_i + 2*margin, c_out)

            # pad labels: keep bg=0 as a valid class; only padding is 255
            # lbl shape: (h_i + 2*margin, w_i + 2*margin)
            padded_lbl = np.full(
                (h + 2 * self.margin, w + 2 * self.margin),
                fill_value=255, dtype=np.int32
                )       
            lbl_region = labels.copy().astype(np.int32)
            padded_lbl[self.margin:self.margin + h,
                       self.margin:self.margin + w] = lbl_region

            patient_list_idx = len(self._patient_padded_data)
            self._patient_padded_data.append(padded_data)
            self._patient_padded_labels.append(padded_lbl)

            # Multi-class boundary detection by default.
            boundary_map = self._compute_multiclass_boundary_map(labels)

            # Optional pair boost for especially hard class-pairs.
            if boundary_pair is not None:
                cls_a, cls_b = boundary_pair
                pair_boundary = self._compute_pair_boundary_map(labels, cls_a, cls_b)
                boundary_map |= pair_boundary

            include_fg_edge_fallback = bool(
                getattr(self.config.split, 'boundary_include_fg_edges', True)
            )
            if include_fg_edge_fallback:
                fg_mask = (labels > 0)
                fg_edge = np.zeros_like(labels, dtype=bool)
                for dr, dc in [(-1, -1), (-1, 0), (-1, 1),
                               (0, -1),           (0, 1),
                               (1, -1),  (1, 0),  (1, 1)]:
                    fg_nb = np.roll(fg_mask, shift=(dr, dc), axis=(0, 1))
                    fg_edge |= (fg_mask != fg_nb)
                fg_edge[0, :] = False
                fg_edge[-1, :] = False
                fg_edge[:, 0] = False
                fg_edge[:, -1] = False
                boundary_map |= fg_edge

            boundary_dilate = int(max(0, getattr(self.config.split, 'boundary_sampling_dilation', 0)))
            if boundary_dilate > 0 and boundary_map.any():
                boundary_map = binary_dilation(boundary_map, iterations=boundary_dilate)

            # ROI mask for foreground-focused sampling.
            fg_roi_map = binary_dilation(labels > 0, iterations=8)

            # extract patch centers: include background pixels
            rows, cols = np.indices(labels.shape)
            rows = rows.reshape(-1)
            cols = cols.reshape(-1)
            n_patches = len(rows)

            if self.max_patches_per_patient is not None and n_patches > self.max_patches_per_patient:
                rows, cols = self._cap_patient_centers(
                    rows=rows,
                    cols=cols,
                    labels=labels,
                    boundary_map=boundary_map,
                    rng=np.random.RandomState(123 + idx),
                )
                n_patches = len(rows)
                if self.debug_mode:
                    tprint(f"    capped patches for patient {pid_idx} to {n_patches}")

            # patch_indices: (patient_list_idx, row_in_padded, col_in_padded)
            indices = np.stack([
                np.full(n_patches, patient_list_idx, dtype=np.int32),
                (rows + self.margin).astype(np.int32),
                (cols + self.margin).astype(np.int32),
            ], axis=1)   # shape: (n_patches, 3)

            patch_labels = labels[rows, cols].astype(np.int32)   # shape: (n_patches,)
            patient_groups = np.full(n_patches, pid_idx, dtype=np.int32)
            boundary_flags = boundary_map[rows, cols].astype(np.bool_)
            fg_roi_flags = fg_roi_map[rows, cols].astype(np.bool_)

            all_patch_indices.append(indices)
            all_patch_labels.append(patch_labels)
            all_patient_groups.append(patient_groups)
            all_pg_tg_boundary.append(boundary_flags)
            all_fg_roi_flags.append(fg_roi_flags)
        # del inside the loop only removed loop-variable
            del data, labels, processed, padded_data, padded_lbl, boundary_map, fg_roi_map
        # patient_records still held references until here
        del patient_records, _patient_id_map

        # concatenate
        self.patch_indices = np.concatenate(all_patch_indices, axis=0)          # (N, 3)
        self.patch_labels = np.concatenate(all_patch_labels, axis=0)            # (N,)
        self.patch_patient_groups = np.concatenate(all_patient_groups, axis=0)  # (N,)
        self.patch_pg_tg_boundary = np.concatenate(all_pg_tg_boundary, axis=0)  # (N,), bool
        self.patch_fg_roi_mask = np.concatenate(all_fg_roi_flags, axis=0)       # (N,), bool

        unique_groups, group_counts = np.unique(
            self.patch_patient_groups, return_counts=True)
        tprint(f"  total patches: {len(self.patch_indices):,} from "
               f"{len(unique_groups)} patients")
        print(f"  patches per patient: min={group_counts.min()}, "
              f"max={group_counts.max()}, mean={group_counts.mean():.0f}")
        print(f"  patch shape: ({self.patch_size}, {self.patch_size}, {c_out}), "
              f"dtype: float32")

    def _get_patch_(self, idx: int) -> np.ndarray:
        """
        get a single patch from per-patient padded data.
        patch_indices[idx] = (patient_list_idx, row_padded, col_padded).
        returns: np.ndarray of shape (patch_size, patch_size, c_out).
        """
        p, r, c = self.patch_indices[idx]
        padded = self._patient_padded_data[p]
        return padded[
            r - self.margin : r + self.margin + 1,
            c - self.margin : c + self.margin + 1,
            :
        ].copy()

    def _get_label_patch_(self, idx: int) -> np.ndarray:
        """
        get dense label patch for segmentation.
        returns: np.ndarray of shape (patch_size, patch_size), 255 = ignore.
        """
        p, r, c = self.patch_indices[idx]
        padded = self._patient_padded_labels[p]
        return padded[
            r - self.margin : r + self.margin + 1,
            c - self.margin : c + self.margin + 1
        ].copy()

    def _pair_data_and_labels_(self) -> List[Tuple[str, str]]:
        """
        Pair data and label files based on naming convention.
        
        Returns:
            List of tuples containing (data_file, label_file) paths
        """
        
        # sorted() ensures deterministic file order across OS/filesystems
        all_files = sorted(os.listdir(self.data_path))

        # filter for .npy files and pair them
        data_files = [f for f in all_files if f.endswith('.npy') and not f.endswith('_gt.npy')]
        label_files = [f for f in all_files if f.endswith('_gt.npy')]
        
        pairs = []
        for data_file in data_files:
            base_name = data_file[:-4]  # remove .npy extension
            label_file = base_name + '_gt.npy'
            if label_file in label_files:
                pairs.append((os.path.join(self.data_path, data_file), os.path.join(self.data_path, label_file)))
            else:
                tprint(f"Warning: No label file found for {data_file}")
        return pairs
    
    def _load_data(self) -> None:
        """legacy stub — kept for abc compliance. see _per_patient_pipeline()."""
        pass
        
    def _preprocess_data(self) -> None:
        """legacy stub — kept for abc compliance. see _per_patient_pipeline()."""
        pass
    
    def _create_data_loader_(self, num_workers=4, batch_size=None, pin_memory=True, 
                           prefetch_factor=2, persistent_workers=False):
        """
        Create torch.utils.data.DataLoaders.
        Core function in dataset preprocessing.
        
        Pipeline:
            - retrieve patch indices, labels, and patient groups from the per-patient pipeline.
            - split patients into train/val/test with stratification on patient-level stats (FG ratio, PG/TG ratio).
            - 
        
        Uses StratifiedGroupKFold (single fold) for train/val+test split,
        to preserve distribution across splits while preventing patient-level leakage.
        Train set uses balanced sampling; val set for early stopping; test set held out.
        
        Args:
            num_workers: worker threads if parallel loading.
            batch_size: batch size for each iteration.
            pin_memory: whether to pin memory for GPU transfer.
            prefetch_factor: preload batches per worker.
            persistent_workers: bool, if keep workers from frequent recreation.
        
        Returns:
            Tuple of (`train_loader`, `val_loader`, `test_loader`).
        """

        if getattr(self, '_use_cached_split_pipeline', False):
            return self._create_cached_split_loaders(
                num_workers=num_workers,
                batch_size=batch_size,
                pin_memory=pin_memory,
                prefetch_factor=prefetch_factor,
                persistent_workers=persistent_workers,
            )

        total_indices = np.arange(len(self.patch_indices))
        all_classes = set(np.unique(self.patch_labels))

        split_seed = int(_split_cfg(self.config, 'seed', 350234, legacy_key='split_seed'))
        test_rate = float(_split_cfg(self.config, 'ratio.legacy.test_rate', 0.2, legacy_key='test_rate'))
        val_rate = float(_split_cfg(self.config, 'ratio.legacy.val_rate', 0.1, legacy_key='val_rate'))

        # Reuse split plan from preprocessor fitting when available to avoid mismatch.
        if hasattr(self, '_patient_split_plan') and isinstance(self._patient_split_plan, dict):
            train_patients = np.asarray(sorted(self._patient_split_plan.get('train', [])), dtype=np.int64)
            val_patients = np.asarray(sorted(self._patient_split_plan.get('val', [])), dtype=np.int64)
            test_patients = np.asarray(sorted(self._patient_split_plan.get('test', [])), dtype=np.int64)
        else:
            # Fallback for backward compatibility.
            patient_ids = np.unique(self.patch_patient_groups)
            targets = list(getattr(self.config.clsf, 'targets', []))
            pg_idx = targets.index('PG') if 'PG' in targets else 1
            tg_idx = targets.index('TG') if 'TG' in targets else max(1, self.num - 1)

            patient_fg_ratio = []
            patient_pg_tg_ratio = []
            for pid in patient_ids:
                p_mask = (self.patch_patient_groups == pid)
                p_labels = self.patch_labels[p_mask]
                fg_ratio = float((p_labels > 0).mean())
                pg_cnt = int((p_labels == pg_idx).sum())
                tg_cnt = int((p_labels == tg_idx).sum())
                pg_tg_ratio = float((pg_cnt + 1.0) / (tg_cnt + 1.0))
                patient_fg_ratio.append(fg_ratio)
                patient_pg_tg_ratio.append(pg_tg_ratio)

            patient_fg_ratio = np.asarray(patient_fg_ratio, dtype=np.float64)
            patient_pg_tg_ratio = np.asarray(patient_pg_tg_ratio, dtype=np.float64)
            fg_bins = np.digitize(patient_fg_ratio, np.quantile(patient_fg_ratio, [0.33, 0.66]), right=False)
            pgtg_bins = np.digitize(patient_pg_tg_ratio, np.quantile(patient_pg_tg_ratio, [0.33, 0.66]), right=False)
            patient_strata = fg_bins * 3 + pgtg_bins

            try:
                trainval_patients, test_patients = train_test_split(
                    patient_ids,
                    test_size=test_rate,
                    random_state=split_seed,
                    stratify=patient_strata,
                )
            except Exception:
                trainval_patients, test_patients = train_test_split(
                    patient_ids,
                    test_size=test_rate,
                    random_state=split_seed,
                    stratify=None,
                )

            # split train/val at patient level with stratification on the same strata code.
            strata_map = {int(pid): int(s) for pid, s in zip(patient_ids, patient_strata)}
            trainval_strata = np.array([strata_map[int(pid)] for pid in trainval_patients], dtype=np.int64)
            try:
                train_patients, val_patients = train_test_split(
                    trainval_patients,
                    test_size=val_rate,
                    random_state=split_seed,
                    stratify=trainval_strata,
                )
            except Exception:
                train_patients, val_patients = train_test_split(
                    trainval_patients,
                    test_size=val_rate,
                    random_state=split_seed,
                    stratify=None,
                )

        train_idx = total_indices[np.isin(self.patch_patient_groups, train_patients)]
        val_idx = total_indices[np.isin(self.patch_patient_groups, val_patients)]
        test_idx = total_indices[np.isin(self.patch_patient_groups, test_patients)]
        
        # verify no patient leakage.
        train_patients = set(self.patch_patient_groups[train_idx])
        val_patients = set(self.patch_patient_groups[val_idx])
        test_patients = set(self.patch_patient_groups[test_idx])
        assert len(train_patients & val_patients) == 0, "Leakage: train/val overlap!"
        assert len(train_patients & test_patients) == 0, "Leakage: train/test overlap!"
        assert len(val_patients & test_patients) == 0, "Leakage: val/test overlap!"
        tprint(f"Patient-level split(seed={split_seed}): {len(train_patients)} train, "
               f"{len(val_patients)} val, {len(test_patients)} test (0 overlap)")
        
        # check class coverage
        for name, idx in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
            split_classes = set(np.unique(self.patch_labels[idx]))
            if split_classes != all_classes:
                print(f"  WARNING: {name} set missing classes: {all_classes - split_classes}")
        
        # log class distribution for sanity check
        for name, idx in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
            dist = np.bincount(self.patch_labels[idx], minlength=self.num)
            pct = (dist / dist.sum() * 100).round(1)
            print(f"  {name} label distribution: {dict(zip(self.config.clsf.targets, pct))}")
        
        # create indexed subsets for augmentation
        train_subset = _IndexedSubset(self, train_idx)
        val_subset = _IndexedSubset(self, val_idx)
        test_subset = _IndexedSubset(self, test_idx)
        print("Training augmentation enabled")

        if batch_size is None:
            batch_size = getattr(self.config.split, 'batch_size',
                         getattr(self, 'batch_size', 32))
        actual_pin_memory = pin_memory and torch.cuda.is_available()

        # build tri-route pools on subset positions
        train_labels = self.patch_labels[train_idx]
        train_boundary = self.patch_pg_tg_boundary[train_idx] if self.patch_pg_tg_boundary is not None else np.zeros_like(train_labels, dtype=np.bool_)
        fg_positions = np.where(train_labels > 0)[0]
        boundary_positions = np.where(train_boundary)[0]
        background_positions = np.where(train_labels == 0)[0]
        all_positions = np.arange(train_labels.shape[0], dtype=np.int64)

        fg_class_positions = {}
        num_cls = int(getattr(self.config.clsf, 'num', 0))
        for cls_idx in range(1, max(num_cls, 1)):
            cls_pos = np.where(train_labels == cls_idx)[0]
            if cls_pos.size > 0:
                fg_class_positions[int(cls_idx)] = cls_pos

        empirical_fg_ratio = float((train_labels > 0).mean()) if train_labels.size > 0 else 0.0

        steps_per_epoch = max(1, len(train_idx) // max(int(batch_size), 1))
        sampler_mode = str(getattr(self.config.split, 'sampler_mode', 'fixed')).lower()
        ratio_fg = float(_split_cfg(self.config, 'sampler_mix_fg', 0.25))
        ratio_boundary = float(_split_cfg(self.config, 'sampler_mix_boundary', 0.15))
        ratio_background = float(_split_cfg(self.config, 'sampler_mix_bg', 0.35))
        ratio_random_cfg = _split_cfg(self.config, 'sampler_mix_random', None)
        ratio_random = None if ratio_random_cfg is None else float(ratio_random_cfg)
        train_batch_sampler = _FixedRatioBatchSampler(
            fg_positions=fg_positions,
            fg_class_positions=fg_class_positions,
            boundary_positions=boundary_positions,
            background_positions=background_positions,
            all_positions=all_positions,
            position_groups=self.patch_patient_groups[train_idx],
            batch_size=int(batch_size),
            ratio_fg=ratio_fg,
            ratio_boundary=ratio_boundary,
            ratio_background=ratio_background,
            ratio_random=ratio_random,
            sampler_mode=sampler_mode,
            target_fg_ratio=float(_split_cfg(self.config, 'sampler_target_mix_fg', ratio_fg)),
            target_boundary_ratio=float(_split_cfg(self.config, 'sampler_target_mix_boundary', ratio_boundary)),
            target_background_ratio=float(_split_cfg(self.config, 'sampler_target_mix_bg', ratio_background)),
            adapt_momentum=float(getattr(self.config.split, 'sampler_adapt_momentum', 0.9)),
            min_fg_ratio=float(_split_cfg(self.config, 'sampler_fg_ratio_min', 0.12)),
            max_fg_ratio=float(_split_cfg(self.config, 'sampler_fg_ratio_max', 0.28)),
            empirical_fg_ratio=empirical_fg_ratio,
            fg_inverse_pow=float(_split_cfg(self.config, 'sampler_fg_inverse_pow', 1.0)),
            fg_min_per_class=int(_split_cfg(self.config, 'sampler_fg_min_per_class', 0)),
            diversity_strength=float(getattr(self.config.split, 'sampler_diversity_strength', 0.75)),
            steps_per_epoch=steps_per_epoch,
            seed=split_seed,
        )
        cur_fg_ratio, cur_bd_ratio, cur_bg_ratio, cur_rd_ratio = train_batch_sampler.get_current_ratios()
        print(
            f"Train sampler pools (subset positions): FG={len(fg_positions)}, "
            f"Boundary={len(boundary_positions)}, BG={len(background_positions)}, All={len(all_positions)}"
        )
        print(
            f"Train sampler ratios(init): fg={cur_fg_ratio:.2f}, boundary={cur_bd_ratio:.2f}, "
            f"bg={cur_bg_ratio:.2f}, random={cur_rd_ratio:.2f}, "
            f"empirical_fg={empirical_fg_ratio:.2f} | mode={sampler_mode}"
        )
        print(
            f"Train sampler quotas/step(init): fg={train_batch_sampler.n_fg}, "
            f"boundary={train_batch_sampler.n_boundary}, bg={train_batch_sampler.n_background}, "
            f"random={train_batch_sampler.n_random}, batch_size={int(batch_size)}"
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_subset,
            batch_sampler=train_batch_sampler,
            num_workers=num_workers,
            pin_memory=actual_pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers and (num_workers > 0),
            timeout=60 if num_workers > 0 else 0
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=actual_pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers and (num_workers > 0),
            drop_last=False,
            timeout=60 if num_workers > 0 else 0
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=actual_pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers and (num_workers > 0),
            drop_last=False,
            timeout=60 if num_workers > 0 else 0
        )
        
        tprint('\n')
        print(f"Training set: {len(train_loader)} batches of {len(train_idx)} samples,\n\t with distribution: {np.bincount(self.patch_labels[train_idx], minlength=self.num)}\n")
        print(f"Validation set: {len(val_loader)} batches of {len(val_idx)} samples,\n\t with distribution: {np.bincount(self.patch_labels[val_idx], minlength=self.num)}\n")
        print(f"Test set: {len(test_loader)} batches of {len(test_idx)} samples,\n\t with distribution: {np.bincount(self.patch_labels[test_idx], minlength=self.num)}\n")
        print(f"DataLoader config: num_workers={num_workers}, batch_size={batch_size}, pin_memory={actual_pin_memory}")
        
        return train_loader, val_loader, test_loader

    def _create_cached_split_loaders(self,
                                     num_workers=4, prefetch_factor=2,
                                     batch_size=None, pin_memory=True,
                                     persistent_workers=False) -> Tuple[torch.utils.data.DataLoader,
                                            torch.utils.data.DataLoader,
                                            torch.utils.data.DataLoader]:
        """
        Pipeline:
            - per-patient norm + pca + pad + patch extraction.
            - cache extracted patches as sample-wise chunks of multiple patches.
            - train/val/test split at sample chunk level with balanced sampling on patch-level labels.

        Returns:
            Tuple of (`train_loader`, `val_loader`, `test_loader`).
        """
        
        samples_per_step = int(getattr(self.config.split, 'samples_per_step', 1))
        samples_per_step = max(1, samples_per_step)
        actual_pin_memory = pin_memory and torch.cuda.is_available()

        # Keep cached split pipeline aligned with trainer's intended global batch size.
        if batch_size is None:
            target_patches_per_step = int(getattr(self.config.split, 'target_patches_per_step',
                                         getattr(self.config.split, 'batch_size', 256)))
        else:
            target_patches_per_step = int(getattr(self.config.split, 'target_patches_per_step', batch_size))
        target_patches_per_step = max(128, target_patches_per_step)

        feature_dim = int(self.feature_dim or self.config.preprocess.pca_components)
        patch_area = int(self.patch_size) * int(self.patch_size)
        bytes_per_patch = (feature_dim * patch_area * 4) + (patch_area * 8)

        max_patches_per_chunk = int(getattr(self.config.split, 'max_patches_per_sample_chunk', 0))
        target_chunk_cap = max(128, target_patches_per_step // samples_per_step)
        if max_patches_per_chunk <= 0:
            budget_mb = int(getattr(self.config.memory, 'cached_loader_patch_budget_mb', 512))
            budget_mb = max(64, budget_mb)
            budget_bytes = budget_mb * 1024 * 1024
            # Each batch stacks multiple samples; reserve budget per sample chunk.
            per_sample_budget = max(1, budget_bytes // samples_per_step)
            budget_chunk_cap = max(128, per_sample_budget // max(bytes_per_patch, 1))
            max_patches_per_chunk = min(budget_chunk_cap, target_chunk_cap)
        else:
            # Never exceed the intended step-level global patch target unless user raises the target.
            max_patches_per_chunk = min(max_patches_per_chunk, target_chunk_cap)

        train_ds = _SamplePatchChunkDataset(
            samples=self._split_manifests['train'].get('samples', []),
            patch_size=self.patch_size,
            max_patches_per_chunk=max_patches_per_chunk,
            augment=False,
        )
        val_ds = _SamplePatchChunkDataset(
            samples=self._split_manifests['val'].get('samples', []),
            patch_size=self.patch_size,
            max_patches_per_chunk=max_patches_per_chunk,
            augment=False,
        )
        test_ds = _SamplePatchChunkDataset(
            samples=self._split_manifests['test'].get('samples', []),
            patch_size=self.patch_size,
            max_patches_per_chunk=max_patches_per_chunk,
            augment=False,
        )
        sampler_mode = str(getattr(self.config.split, 'sampler_mode', 'fixed')).lower()
        ratio_fg = float(_split_cfg(self.config, 'sampler_mix_fg', 0.20))
        ratio_boundary = float(_split_cfg(self.config, 'sampler_mix_boundary', 0.08))
        ratio_background = float(_split_cfg(self.config, 'sampler_mix_bg', 0.35))
        ratio_random_cfg = _split_cfg(self.config, 'sampler_mix_random', None)
        ratio_random = None if ratio_random_cfg is None else float(ratio_random_cfg)
        train_batch_sampler = _CachedSplitRatioBatchSampler(
            dataset=train_ds,
            samples_per_step=samples_per_step,
            ratio_fg=ratio_fg,
            ratio_boundary=ratio_boundary,
            ratio_background=ratio_background,
            ratio_random=ratio_random,
            sampler_mode=sampler_mode,
            target_fg_ratio=float(_split_cfg(self.config, 'sampler_target_mix_fg', ratio_fg)),
            target_boundary_ratio=float(_split_cfg(self.config, 'sampler_target_mix_boundary', ratio_boundary)),
            target_background_ratio=float(_split_cfg(self.config, 'sampler_target_mix_bg', ratio_background)),
            adapt_momentum=float(getattr(self.config.split, 'sampler_adapt_momentum', 0.9)),
            min_fg_ratio=float(_split_cfg(self.config, 'sampler_fg_ratio_min', 0.12)),
            max_fg_ratio=float(_split_cfg(self.config, 'sampler_fg_ratio_max', 0.28)),
            seed=int(getattr(self.config.split, 'split_seed', 350234)),
        )

        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_sampler=train_batch_sampler,
            num_workers=num_workers,
            pin_memory=actual_pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers and (num_workers > 0),
            collate_fn=_concat_sample_patch_batches,
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=samples_per_step,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=actual_pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers and (num_workers > 0),
            collate_fn=_concat_sample_patch_batches,
            drop_last=False,
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=samples_per_step,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=actual_pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers and (num_workers > 0),
            collate_fn=_concat_sample_patch_batches,
            drop_last=False,
        )

        tprint('Using cached split DataLoaders (sample-wise patch chunk stacking)')
        print(
            f"  sample chunks(train/val/test): {len(train_ds)}/{len(val_ds)}/{len(test_ds)}"
        )
        print(
            f"  cached train sampler mode={sampler_mode}, samples_per_step={samples_per_step}, "
            f"ratio_fg={ratio_fg:.2f}, ratio_boundary={ratio_boundary:.2f}, "
            f"ratio_bg={ratio_background:.2f}, ratio_random={ratio_random if ratio_random is not None else 'auto'}"
        )
        print(
            f"  cached train sampler quotas/step(init): fg={train_batch_sampler.n_fg}, "
            f"boundary={train_batch_sampler.n_boundary}, bg={train_batch_sampler.n_background}, "
            f"random={train_batch_sampler.n_random}, samples_per_step={samples_per_step}"
        )
        print(
            f"  samples_per_step={samples_per_step}, patch_size={self.patch_size}, channels={self.feature_dim}, "
            f"target_patches_per_step={target_patches_per_step}, max_patches_per_chunk={max_patches_per_chunk}"
        )
        est_step_mb = (samples_per_step * max_patches_per_chunk * bytes_per_patch) / (1024 * 1024)
        print(
            f"  estimated host batch tensor footprint ~= {est_step_mb:.1f} MB "
            f"(before DataLoader queues/pin buffers)"
        )
        return train_loader, val_loader, test_loader

    def _create_cv_data_loaders_(self, n_folds=5, num_workers=4, batch_size=None,
                               pin_memory=True, prefetch_factor=2,
                               persistent_workers=False):
        """
        Create K-fold cross-validation DataLoaders with patient-level grouping.

        Uses ``StratifiedGroupKFold`` to guarantee zero patient overlap between
        train and test sets while preserving class proportions across folds.

        Args:
            n_folds: Number of cross-validation folds.

        Returns:
            List of ``(train_loader, test_loader)`` tuples, one per fold.
        """

        total_indices = np.arange(len(self.patch_indices))
        all_classes = set(np.unique(self.patch_labels))

        split_seed = int(_split_cfg(self.config, 'seed', 350234, legacy_key='split_seed'))
        sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True,
                        random_state=split_seed)

        if batch_size is None:
            batch_size = self.batch_size if hasattr(self, 'batch_size') else 32
        actual_pin_memory = pin_memory and torch.cuda.is_available()

        fold_loaders = []

        tprint(f"\n[Cross-Validation Split] {n_folds} folds, "
               f"patient-level StratifiedGroupKFold (seed={split_seed})")

        for fold_idx, (train_idx, test_idx) in enumerate(
                sgkf.split(total_indices, self.patch_labels,
                           groups=self.patch_patient_groups)):

            # Verify no patient-level leakage
            train_patients = set(self.patch_patient_groups[train_idx])
            test_patients = set(self.patch_patient_groups[test_idx])
            assert len(train_patients & test_patients) == 0, \
                f"Fold {fold_idx+1}: Patient leakage detected!"

            # Check class coverage
            train_classes = set(np.unique(self.patch_labels[train_idx]))
            test_classes = set(np.unique(self.patch_labels[test_idx]))
            if train_classes != all_classes:
                tprint(f"[WARNING] Fold {fold_idx+1}: Training missing classes: "
                      f"{all_classes - train_classes}")
            if test_classes != all_classes:
                tprint(f"[WARNING] Fold {fold_idx+1}: Test missing classes: "
                      f"{all_classes - test_classes}")

            tprint(f"  Fold {fold_idx+1}/{n_folds}: "
                  f"train={len(train_idx)} ({len(train_patients)} patients), "
                  f"test={len(test_idx)} ({len(test_patients)} patients)")

            # subsets
            train_subset = _IndexedSubset(self, train_idx)
            test_subset = _IndexedSubset(self, test_idx)

            train_loader = torch.utils.data.DataLoader(
                train_subset,
                batch_size=batch_size,
                shuffle=True,  # no resampling — preserves original distribution
                num_workers=num_workers,
                pin_memory=actual_pin_memory,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
                persistent_workers=persistent_workers and (num_workers > 0),
                drop_last=True,
                timeout=60 if num_workers > 0 else 0
            )
            test_loader = torch.utils.data.DataLoader(
                test_subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=actual_pin_memory,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
                persistent_workers=persistent_workers and (num_workers > 0),
                drop_last=False,
                timeout=60 if num_workers > 0 else 0
            )

            fold_loaders.append((train_loader, test_loader))

        tprint(f"  Created {n_folds} fold DataLoader pairs\n")
        return fold_loaders

    # public interfaces
    def get_loaders(self, *args, **kwargs):
        return self._create_data_loader_(*args, **kwargs)

    def get_cv_loaders(self, *args, **kwargs):
        return self._create_cv_data_loaders_(*args, **kwargs)

    # convenience helpers
    def describe(self, top_k: int = 5) -> None:
        """Return and print basic dataset statistics for quick sanity check."""
        if getattr(self, '_use_cached_split_pipeline', False):
            print('[Dataset Summary] split-cache mode')
            print(f"  patients: {int(getattr(self, '_num_patients', 0))}")
            print(f"  estimated patches: {int(getattr(self, '_cached_total_patches', 0))}")
            for split_name in ('train', 'val', 'test'):
                manifest = self._split_manifests.get(split_name, {})
                counts = self._split_class_counts.get(split_name, np.zeros(self.num, dtype=np.int64))
                print(f"  {split_name}: samples={len(manifest.get('samples', []))}, class_counts={counts.tolist()}")
            return

        stats: Dict[str, Any] = {}
        stats["num_patients"] = int(getattr(self, "_num_patients", 0))
        stats["num_patches"] = int(len(self.patch_indices))
        stats["patch_shape"] = (self.patch_size, self.patch_size)
        cls_counts = np.bincount(self.patch_labels, minlength=self.num)
        stats["class_counts"] = cls_counts.tolist()
        stats["class_ratios"] = (cls_counts / np.maximum(cls_counts.sum(), 1)).round(4).tolist()

        # patient-level patch counts
        patient_counts = []
        if hasattr(self, "patch_patient_groups"):
            for pid in np.unique(self.patch_patient_groups):
                patient_counts.append((int(pid), int((self.patch_patient_groups == pid).sum())))
            stats["patient_patch_counts"] = patient_counts

        print("[Dataset Summary]")
        print(f"  patients: {stats['num_patients']}, patches: {stats['num_patches']}")
        print(f"  patch size: {stats['patch_shape']}, num_classes: {self.num}")
        print(f"  class counts: {cls_counts.tolist()}")
        print(f"  class ratios: {stats['class_ratios']}")
        if patient_counts:
            top = sorted(patient_counts, key=lambda x: x[1], reverse=True)[:top_k]
            print(f"  top-{top_k} patients by patch count: {top}")

    def __len__(self) -> int:
        if getattr(self, '_use_cached_split_pipeline', False):
            return int(getattr(self, '_cached_total_patches', 0))
        return super().__len__()


class _SplitPreprocessManager:
    """Build split-wise cached datasets with train-only fit and streaming transforms.

    Implements both iterator and context-manager interfaces as required.
    """

    def __init__(self, dataset: NpyHSDataset, cache_root_data: Path, cache_root_label: Path):
        self.dataset = dataset
        self.cache_root_data = cache_root_data
        self.cache_root_label = cache_root_label
        self.raw_pairs: List[Tuple[str, str]] = []

    def __enter__(self):
        self.raw_pairs = self._discover_raw_pairs()
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        return iter(self.raw_pairs)

    def _discover_raw_pairs(self) -> List[Tuple[str, str]]:
        data_root = Path(self.dataset.data_path)
        label_root = Path(self.dataset.label_path)
        
        data_files = sorted([p for p in data_root.glob('*.npy') if p.is_file() and not p.name.endswith('_gt.npy')])
        label_files = sorted([p for p in label_root.glob('*.npy') if p.is_file() and p.name.endswith('_gt.npy')])
        label_map = {p.name: p for p in label_files}

        pairs: List[Tuple[str, str]] = []
        for data_file in data_files:
            label_name = data_file.stem + '_gt.npy'
            if label_name in label_map:
                pairs.append((str(data_file), str(label_map[label_name])))
        return pairs

    def _split_pairs(self,
                     pairs: List[Tuple[str, str]],
                     label_remap: np.ndarray) -> Dict[str, List[Tuple[str, str]]]:
        """
        Counts label stats and split in patient level.
        Returns dict of split name to list of (data_file, label_file) pairs.
        """

        split_seed = int(getattr(self.dataset.config.split, 'split_seed', 350234))
        train_ratio = float(getattr(self.dataset.config.split, 'train_ratio', 0.8))
        val_ratio = float(getattr(self.dataset.config.split, 'val_ratio', 0.1))
        test_ratio = float(getattr(self.dataset.config.split, 'test_ratio', 0.1))

        if train_ratio <= 0 or val_ratio < 0 or test_ratio < 0:
            raise ValueError('split ratios must be >= 0 and train_ratio be > 0')
        ratio_sum = train_ratio + val_ratio + test_ratio
        if ratio_sum <= 0:
            raise ValueError('split ratios sum must be > 0')

        train_ratio /= ratio_sum
        val_ratio /= ratio_sum
        test_ratio /= ratio_sum

        patient_to_pairs: Dict[str, List[Tuple[str, str]]] = {}
        for data_file, label_file in pairs:
            pid = self.dataset._extract_patient_id(data_file)
            patient_to_pairs.setdefault(pid, []).append((data_file, label_file))

        patient_ids = np.asarray(sorted(patient_to_pairs.keys()))
        if patient_ids.size < 3:
            raise RuntimeError('Too few patients for train/val/test split')

        targets = list(getattr(self.dataset.config.clsf, 'targets', []))
        pg_idx = targets.index('PG') if 'PG' in targets else 1
        tg_idx = targets.index('TG') if 'TG' in targets else max(1, len(targets) - 1)

        patient_fg_ratio = []
        patient_pg_tg_ratio = []
        for pid in patient_ids:
            fg_cnt = 0
            total_cnt = 0
            pg_cnt = 0
            tg_cnt = 0
            for data_file, label_file in patient_to_pairs[pid]:
                labels_raw = np.load(label_file).astype(np.int32)
                labels = self.dataset._apply_label_remap(labels_raw, label_remap, label_file)
                
                # counts FG vs BG, and PG vs TG for stratification
                fg_cnt += int((labels > 0).sum())
                total_cnt += int(labels.size)
                pg_cnt += int((labels == pg_idx).sum())
                tg_cnt += int((labels == tg_idx).sum())

            # tracking total stats.
            # add 1 to avoid ZeroDivisionError.
            fg_ratio = float(fg_cnt / max(total_cnt, 1))
            pg_tg_ratio = float((pg_cnt + 1.0) / (tg_cnt + 1.0))
            patient_fg_ratio.append(fg_ratio)
            patient_pg_tg_ratio.append(pg_tg_ratio)

        patient_fg_ratio = np.asarray(patient_fg_ratio, dtype=np.float64)
        patient_pg_tg_ratio = np.asarray(patient_pg_tg_ratio, dtype=np.float64)

        # bin patients into 3 groups based on FG and PG/TG ratio,
        # total 9 strata in 3 splits for stratified splitting.
        # this clusters samples with same distribution, splits different ones.
        fg_bins = np.digitize(patient_fg_ratio, np.quantile(patient_fg_ratio, [0.33, 0.66]), right=False)
        pgtg_bins = np.digitize(patient_pg_tg_ratio, np.quantile(patient_pg_tg_ratio, [0.33, 0.66]), right=False)
        patient_strata = fg_bins * 3 + pgtg_bins

        trainval_ratio = max(1e-8, train_ratio + val_ratio)
        test_size = float(np.clip(test_ratio, 1e-8, 0.9))
        
        # try patient level split.
        # phase 1 split test from train + val.
        try:
            trainval_patients, test_patients = train_test_split(
                patient_ids,
                test_size=test_size,
                random_state=split_seed,
                stratify=patient_strata,
            )
        except Exception:
            trainval_patients, test_patients = train_test_split(
                patient_ids,
                test_size=test_size,
                random_state=split_seed,
                stratify=None,
            )

        # phase 2 split train from val.
        patient_to_strata = {pid: int(s) for pid, s in zip(patient_ids.tolist(), patient_strata.tolist())}
        trainval_strata = np.array([patient_to_strata[str(pid)] for pid in trainval_patients], dtype=np.int64)
        val_share = float(np.clip(val_ratio / trainval_ratio, 1e-8, 0.9))
        try:
            train_patients, val_patients = train_test_split(
                trainval_patients,
                test_size=val_share,
                random_state=split_seed,
                stratify=trainval_strata,
            )
        except Exception:
            train_patients, val_patients = train_test_split(
                trainval_patients,
                test_size=val_share,
                random_state=split_seed,
                stratify=None,
            )

        train_p = set([str(x) for x in train_patients.tolist()])
        val_p = set([str(x) for x in val_patients.tolist()])
        test_p = set([str(x) for x in test_patients.tolist()])

        # fallback to random split if any split is empty
        if not train_p or not val_p or not test_p:
            rng = np.random.RandomState(split_seed)
            perm = patient_ids[rng.permutation(patient_ids.size)]
            n_total = len(perm)
            n_test = max(1, int(round(n_total * test_ratio)))
            n_val = max(1, int(round(n_total * val_ratio)))
            n_train = max(1, n_total - n_val - n_test)
            if n_train + n_val + n_test > n_total:
                n_test = max(1, n_total - n_train - n_val)
            train_p = set([str(x) for x in perm[:n_train].tolist()])
            val_p = set([str(x) for x in perm[n_train:n_train + n_val].tolist()])
            test_p = set([str(x) for x in perm[n_train + n_val:n_train + n_val + n_test].tolist()])

        # finalize split lists.
        out = {'train': [], 'val': [], 'test': []}
        for pid, file_pairs in patient_to_pairs.items():
            if pid in train_p:
                out['train'].extend(file_pairs)
            elif pid in val_p:
                out['val'].extend(file_pairs)
            elif pid in test_p:
                out['test'].extend(file_pairs)
        return out

    def build(self) -> None:
        """
        Splitting the data into train + val + test.
        
        Pipeline:
        1) Pair raw .npy data/label and patient level,
        2) Cluster based on label stats, split patient level.
        3) Fit a global preprocessor based on train, save stats to cache.
        4) For each split, stream-process per sample with the preprocessor.
        5) Build split manifests with sample metadata and class counts for further DataLoader.
        """
        
        # pair + cluster + split
        pairs = list(self.raw_pairs)
        if not pairs:
            raise RuntimeError('No raw .npy pairs found for split cache build')

        label_remap = self.dataset._build_label_remap()
        split_pairs = self._split_pairs(pairs, label_remap)

        # fit preprocessor based on train.
        rng_fit = np.random.RandomState(350236)
        fit_pixels: List[np.ndarray] = []
        for data_file, label_file in split_pairs['train']:
            data = np.load(data_file).astype(np.float32)
            labels_raw = np.load(label_file).astype(np.int32)
            labels = self.dataset._apply_label_remap(labels_raw, label_remap, label_file)
            data_fit, labels_fit = self.dataset._augment_sample(data, labels, rng_fit)
            fit_pixels.append(self.dataset._collect_pixels_for_preprocessor(data_fit, labels_fit, rng_fit))

        if not fit_pixels:
            raise RuntimeError('No train pixels available to fit preprocessor')

        # this preprocessor inherits from sklearn PCA and StandardScaler,
        # with additional logic to handle large data and PCA dim reduction.
        # use `fit` to compute global mean/std and PCA components,
        # then use `transform` to apply the fit to all splits.
        # implementation see line 42, Z-score norm + PCA.
        # actual transform see line 165 + line 169.
        all_fit = np.concatenate(fit_pixels, axis=0)
        
        # init preprocessor and get the fit
        preprocessor = HSPreprocessor(
            pca_components=self.dataset.config.preprocess.pca_components,
            max_fit_samples=int(getattr(self.dataset.config.preprocess, 'max_fit_samples', 2_000_000)),
            max_pca_samples=int(getattr(self.dataset.config.preprocess, 'max_pca_samples', 500_000)),
            random_state=350234)
        preprocessor.fit(all_fit)

        # Save preprocessor stats atomically.
        stats_target = self.cache_root_data / 'preprocess_stats.npz'
        fd, tmp_stats = tempfile.mkstemp(
            dir=str(self.cache_root_data),
            prefix='.preprocess_stats.',
            suffix='.npz',
        )
        os.close(fd)
        try:
            np.savez_compressed(
                tmp_stats,
                global_mean=preprocessor.global_mean_,
                global_std=preprocessor.global_std_,
                pca_components=preprocessor.pca_components_ if preprocessor.pca_components_ is not None else np.empty((0, 0), dtype=np.float32),
                pca_mean=preprocessor.pca_mean_ if preprocessor.pca_mean_ is not None else np.empty((0,), dtype=np.float32),
                pca_explained=preprocessor.pca_explained_variance_ratio_ if preprocessor.pca_explained_variance_ratio_ is not None else np.empty((0,), dtype=np.float32),
                feature_dim=np.asarray([preprocessor.n_features_out_], dtype=np.int32),
            )
            os.replace(tmp_stats, str(stats_target))
        finally:
            if os.path.exists(tmp_stats):
                try:
                    os.remove(tmp_stats)
                except OSError:
                    pass

        worker_count = int(getattr(self.dataset.config.preprocess, 'split_preprocess_workers', 1))
        worker_count = max(1, worker_count)

        # resolve boundary class pair,
        # which is used for especially distiguishing hard-classification pairs.
        boundary_pair = self.dataset._resolve_boundary_class_pair()

        # process 3 splits and save to cache with metadata manifest.
        for split_name, pair_list in split_pairs.items():
            # dirs like root_dir/{train,val,test}.
            split_data_dir = self.cache_root_data / split_name
            split_label_dir = self.cache_root_label / split_name
            split_data_dir.mkdir(parents=True, exist_ok=True)
            split_label_dir.mkdir(parents=True, exist_ok=True)

            samples = []
            class_counts = np.zeros(self.dataset.num, dtype=np.int64)
            split_unknown_pixels = 0
            split_total_pixels = 0

            def _process_pair(data_file: str, label_file: str):
                """
                use `transform` per sample.
                Save processed data along with stats.
                """
                data = np.load(data_file).astype(np.float32)
                labels_raw = np.load(label_file).astype(np.int32)
                unknown_mask = labels_raw >= len(label_remap)
                unknown_count = int(unknown_mask.sum())
                labels = self.dataset._apply_label_remap(labels_raw, label_remap, label_file)

                if split_name == 'train' and bool(_preprocess_cfg(self.dataset.config.augment, 'augment_train', True)):
                    file_seed = int(hashlib.md5(data_file.encode('utf-8')).hexdigest()[:8], 16)
                    aug_rng = np.random.RandomState(
                        int(getattr(self.dataset.config.split, 'split_seed', 350234))
                        + int(file_seed % 100000)
                    )
                    data, labels = self.dataset._augment_sample(data, labels, aug_rng)
                processed = preprocessor.transform(data).astype(np.float32)

                boundary_map = self.dataset._compute_multiclass_boundary_map(labels)
                if boundary_pair is not None:
                    cls_a, cls_b = boundary_pair
                    boundary_map |= self.dataset._compute_pair_boundary_map(labels, cls_a, cls_b)
                boundary_ratio = float(boundary_map.mean())

                data_name = os.path.basename(data_file)
                label_name = os.path.basename(label_file)
                out_data = split_data_dir / data_name
                out_label = split_label_dir / label_name

                _atomic_save_npy(out_data, processed)
                _atomic_save_npy(out_label, labels.astype(np.int32))

                binc = np.bincount(labels.reshape(-1), minlength=self.dataset.num)
                binc = binc.astype(np.int64)

                pid = self.dataset._extract_patient_id(data_file)
                h, w = labels.shape[:2]
                sample_item = {
                    'data': str(out_data),
                    'label': str(out_label),
                    'name': data_name,
                    'patient_id': pid,
                    'height': int(h),
                    'width': int(w),
                    'fg_ratio': float((labels > 0).mean()),
                    'boundary_ratio': float(boundary_ratio),
                    'class_hist': binc.astype(np.int64).tolist(),
                }
                return sample_item, binc, unknown_count, int(labels_raw.size)

            if worker_count > 1 and len(pair_list) > 1:
                with ThreadPoolExecutor(max_workers=worker_count) as ex:
                    futures = [ex.submit(_process_pair, d, l) for d, l in pair_list]
                    for fu in futures:
                        sample_item, binc, unknown_count, total_pixels = fu.result()
                        samples.append(sample_item)
                        class_counts += binc
                        split_unknown_pixels += int(unknown_count)
                        split_total_pixels += int(total_pixels)
            else:
                for data_file, label_file in pair_list:
                    sample_item, binc, unknown_count, total_pixels = _process_pair(data_file, label_file)
                    samples.append(sample_item)
                    class_counts += binc
                    split_unknown_pixels += int(unknown_count)
                    split_total_pixels += int(total_pixels)

            if split_unknown_pixels > 0:
                ratio = 100.0 * float(split_unknown_pixels) / float(max(split_total_pixels, 1))
                tprint(
                    f"WARNING: split={split_name} unknown-label pixels mapped to BG: "
                    f"{split_unknown_pixels:,}/{split_total_pixels:,} ({ratio:.3f}%)"
                )

            manifest = {
                'split': split_name,
                'num_samples': len(samples),
                'class_counts': class_counts.astype(np.int64).tolist(),
                'samples': samples,
            }
            _atomic_write_json(split_data_dir / 'manifest.json', manifest)


class _SamplePatchChunkDataset(Dataset):
    """Dataset that yields chunked patches from one sample per item.

    This avoids materializing all H*W patches of a sample at once, which can
    easily exceed worker memory for large HSI tiles.
    """

    def __init__(self, samples: List[Dict[str, Any]], patch_size: int,
                 max_patches_per_chunk: int, augment: bool = False):
        self.samples = list(samples)
        self.patch_size = int(patch_size)
        self.margin = (self.patch_size - 1) // 2
        self.max_patches_per_chunk = max(1, int(max_patches_per_chunk))
        self.augment = bool(augment)

        d = np.arange(-self.margin, self.margin + 1, dtype=np.int64)
        self._row_offsets, self._col_offsets = np.meshgrid(d, d, indexing='ij')

        self.chunks: List[Tuple[int, int, int, int]] = []
        self._chunks_per_sample = [0 for _ in self.samples]
        self._sample_chunk_ranges: List[Tuple[int, int]] = []
        for sample_idx, item in enumerate(self.samples):
            h = int(item['height'])
            w = int(item['width'])
            total = h * w
            begin = len(self.chunks)
            for start in range(0, total, self.max_patches_per_chunk):
                end = min(total, start + self.max_patches_per_chunk)
                self.chunks.append((sample_idx, start, end, w))
                self._chunks_per_sample[sample_idx] += 1
            finish = len(self.chunks)
            self._sample_chunk_ranges.append((begin, finish))

        oversized = [
            i for i, item in enumerate(self.samples)
            if int(item['height']) * int(item['width']) > self.max_patches_per_chunk
        ]
        if oversized and all(self._chunks_per_sample[i] == 1 for i in oversized):
            raise RuntimeError(
                "Chunk build anomaly: oversized samples produced only one chunk each. "
                "Please verify chunk construction logic."
            )

        # Worker-local one-sample cache; avoids repeated mmap + pad per chunk.
        self._cache_sample_idx: Optional[int] = None
        self._cache_padded_data: Optional[np.ndarray] = None
        self._cache_padded_labels: Optional[np.ndarray] = None

    def __len__(self):
        return len(self.chunks)

    def _extract_chunk_patches(self, padded_data: np.ndarray, padded_labels: np.ndarray,
                               start: int, end: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
        centers = np.arange(start, end, dtype=np.int64)
        rows = (centers // width) + self.margin
        cols = (centers % width) + self.margin

        rr = rows[:, None, None] + self._row_offsets[None, :, :]
        cc = cols[:, None, None] + self._col_offsets[None, :, :]

        patches = padded_data[rr, cc, :].astype(np.float32, copy=False)
        label_patches = padded_labels[rr, cc].astype(np.int32, copy=False)

        if self.augment and np.random.rand() > 0.5:
            patches = np.flip(patches, axis=2).copy()
            label_patches = np.flip(label_patches, axis=2).copy()

        patches = np.transpose(patches, (0, 3, 1, 2))
        return np.ascontiguousarray(patches), np.ascontiguousarray(label_patches)

    def _load_padded_sample(self, sample_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._cache_sample_idx == sample_idx and \
                self._cache_padded_data is not None and \
                self._cache_padded_labels is not None:
            return self._cache_padded_data, self._cache_padded_labels

        item = self.samples[sample_idx]
        data = np.load(item['data'], mmap_mode='r')
        labels = np.load(item['label'], mmap_mode='r')
        padded_data = np.pad(
            data,
            ((self.margin, self.margin), (self.margin, self.margin), (0, 0)),
            mode='constant',
            constant_values=0.0,
        )
        padded_labels = np.pad(
            labels,
            ((self.margin, self.margin), (self.margin, self.margin)),
            mode='constant',
            constant_values=255,
        )

        self._cache_sample_idx = sample_idx
        self._cache_padded_data = padded_data
        self._cache_padded_labels = padded_labels
        return padded_data, padded_labels

    def __getitem__(self, idx: int):
        sample_idx, start, end, width = self.chunks[idx]
        padded_data, padded_labels = self._load_padded_sample(sample_idx)
        patches, label_patches = self._extract_chunk_patches(padded_data, padded_labels, start, end, width)
        chunk_meta = {
            'sample_idx': int(sample_idx),
            'start': int(start),
            'count': int(end - start),
            'width': int(width),
        }
        return torch.from_numpy(patches), torch.from_numpy(label_patches), chunk_meta


class _CachedSplitRatioBatchSampler(Sampler[List[int]]):
    """
    Ratio-aware batch sampler over cached sample chunks.

    It balances chunk selection by sample-level FG and boundary statistics
    while preserving chunk-wise cache locality.
    """

    def __init__(self,
                 dataset: _SamplePatchChunkDataset,
                 samples_per_step: int,
                 ratio_fg: float = 0.20,
                 ratio_boundary: float = 0.08,
                 ratio_background: float = 0.35,
                 ratio_random: Optional[float] = None,
                 sampler_mode: str = 'fixed',
                 target_fg_ratio: float = 0.20,
                 target_boundary_ratio: float = 0.08,
                 target_background_ratio: float = 0.35,
                 adapt_momentum: float = 0.9,
                 min_fg_ratio: float = 0.12,
                 max_fg_ratio: float = 0.28,
                 seed: int = 350234):
        self.dataset = dataset
        self.samples_per_step = max(1, int(samples_per_step))
        self.seed = int(seed)
        self.sampler_mode = str(sampler_mode).lower()

        self.target_fg_ratio = float(np.clip(target_fg_ratio, 0.0, 1.0))
        self.target_boundary_ratio = float(np.clip(target_boundary_ratio, 0.0, 1.0))
        self.target_background_ratio = float(np.clip(target_background_ratio, 0.0, 1.0))
        self.adapt_momentum = float(np.clip(adapt_momentum, 0.0, 0.999))
        self.min_fg_ratio = float(np.clip(min_fg_ratio, 0.0, 1.0))
        self.max_fg_ratio = float(np.clip(max_fg_ratio, self.min_fg_ratio, 1.0))

        self._ratio_fg = float(np.clip(ratio_fg, 0.0, 1.0))
        self._ratio_boundary = float(np.clip(ratio_boundary, 0.0, 1.0))
        self._ratio_background = float(np.clip(ratio_background, 0.0, 1.0))
        self._auto_random_ratio = ratio_random is None
        self._ratio_random = float(np.clip(0.0 if ratio_random is None else ratio_random, 0.0, 1.0))

        self.sample_fg_ratio = np.asarray([
            float(item.get('fg_ratio', 0.0)) for item in self.dataset.samples
        ], dtype=np.float64)
        self.sample_boundary_ratio = np.asarray([
            float(item.get('boundary_ratio', 0.0)) for item in self.dataset.samples
        ], dtype=np.float64)
        self.sample_class_hist = [
            np.asarray(item.get('class_hist', []), dtype=np.int64)
            for item in self.dataset.samples
        ]
        self.sample_patient_ids = np.asarray([
            str(item.get('patient_id', item.get('name', f'sample_{i:03d}')))
            for i, item in enumerate(self.dataset.samples)
        ], dtype=object)

        fg_thr = float(np.percentile(self.sample_fg_ratio, 65)) if self.sample_fg_ratio.size > 0 else 0.0
        bd_thr = float(np.percentile(self.sample_boundary_ratio, 65)) if self.sample_boundary_ratio.size > 0 else 0.0

        all_ids = np.arange(len(self.dataset.samples), dtype=np.int64)
        self.fg_sample_ids = all_ids[self.sample_fg_ratio >= max(fg_thr, 1e-6)]
        self.boundary_sample_ids = all_ids[self.sample_boundary_ratio >= max(bd_thr, 1e-6)]
        bg_thr = float(np.percentile(self.sample_fg_ratio, 35)) if self.sample_fg_ratio.size > 0 else 0.0
        self.bg_sample_ids = all_ids[self.sample_fg_ratio <= bg_thr]

        if self.fg_sample_ids.size == 0:
            # Fallback: keep sampler functional even when FG pool is empty in tiny subsets.
            self.fg_sample_ids = all_ids
        if self.boundary_sample_ids.size == 0:
            self.boundary_sample_ids = all_ids
        if self.bg_sample_ids.size == 0:
            self.bg_sample_ids = all_ids

        self.fg_class_to_sample_ids: Dict[int, np.ndarray] = {}
        if self.sample_class_hist:
            max_cls = max((arr.size for arr in self.sample_class_hist), default=0)
            for cls_idx in range(1, max_cls):
                cls_samples = []
                for sid, hist in enumerate(self.sample_class_hist):
                    if hist.size > cls_idx and int(hist[cls_idx]) > 0:
                        cls_samples.append(sid)
                if cls_samples:
                    self.fg_class_to_sample_ids[int(cls_idx)] = np.asarray(cls_samples, dtype=np.int64)

            # HSI-friendly difficulty score: foreground ratio + boundary ratio + rare-class presence.
            self.sample_hsi_score = self._build_hsi_sample_scores()
            self.diversity_strength = 0.8

        self.steps_per_epoch = max(1, len(self.dataset.chunks) // self.samples_per_step)
        self._epoch = 0
        self._refresh_quota()

    def __len__(self) -> int:
        return self.steps_per_epoch

    def _refresh_quota(self) -> None:
        ratio_random = self._ratio_random
        if self._auto_random_ratio:
            ratio_random = max(0.0, 1.0 - self._ratio_fg - self._ratio_boundary - self._ratio_background)

        shares = np.asarray([
            self._ratio_fg,
            self._ratio_boundary,
            self._ratio_background,
            ratio_random,
        ], dtype=np.float64)
        if float(shares.sum()) <= 1e-12:
            shares = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        shares = shares / shares.sum()

        raw = shares * float(self.samples_per_step)
        quotas = np.floor(raw).astype(np.int64)
        rem = int(self.samples_per_step - int(quotas.sum()))
        if rem > 0:
            frac = raw - quotas.astype(np.float64)
            order = np.argsort(frac)[::-1]
            for i in order[:rem]:
                quotas[int(i)] += 1

        self.n_fg = int(quotas[0])
        self.n_boundary = int(quotas[1])
        self.n_background = int(quotas[2])
        self.n_random = int(quotas[3])

        # Guard against ratio quantization collapse on tiny samples_per_step.
        # Example: samples_per_step=2 with fg_ratio=0.10 can produce n_fg=0 after floor/round.
        self._enforce_minimum_quota()

    def _borrow_one_from_other_routes(self, candidates: List[str]) -> bool:
        for name in candidates:
            cur = int(getattr(self, name, 0))
            if cur > 0:
                setattr(self, name, cur - 1)
                return True
        return False

    def _enforce_minimum_quota(self) -> None:
        # Keep at least one FG sample when FG route is enabled and pool exists.
        if self.samples_per_step >= 2 and self.fg_sample_ids.size > 0 and self._ratio_fg > 0.0 and self.n_fg <= 0:
            if self._borrow_one_from_other_routes(['n_random', 'n_background', 'n_boundary']):
                self.n_fg = 1

        # Keep at least one boundary sample when boundary route is enabled and pool exists.
        if self.samples_per_step >= 2 and self.boundary_sample_ids.size > 0 and self._ratio_boundary > 0.0 and self.n_boundary <= 0:
            if self._borrow_one_from_other_routes(['n_random', 'n_background', 'n_fg']):
                self.n_boundary = 1

    def _adapt_ratios(self) -> None:
        if self.sampler_mode not in {'adaptive', 'hsi_adaptive'}:
            return

        empirical_fg = float(np.mean(self.sample_fg_ratio > 0.0)) if self.sample_fg_ratio.size > 0 else 0.0
        desired_fg = 0.5 * empirical_fg + 0.5 * self.target_fg_ratio
        desired_fg = float(np.clip(desired_fg, self.min_fg_ratio, self.max_fg_ratio))
        desired_bd = float(np.clip(self.target_boundary_ratio, 0.0, 1.0))
        desired_bg = float(np.clip(self.target_background_ratio, 0.0, 1.0))

        m = self.adapt_momentum
        self._ratio_fg = float(np.clip(m * self._ratio_fg + (1.0 - m) * desired_fg,
                                       self.min_fg_ratio, self.max_fg_ratio))
        self._ratio_boundary = float(np.clip(m * self._ratio_boundary + (1.0 - m) * desired_bd,
                                             0.0, 1.0))
        self._ratio_background = float(np.clip(m * self._ratio_background + (1.0 - m) * desired_bg,
                               0.0, 1.0))
        self._refresh_quota()

    @staticmethod
    def _safe_norm(x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        mn = float(np.min(x))
        mx = float(np.max(x))
        if mx - mn < 1e-12:
            return np.zeros_like(x, dtype=np.float64)
        return (x - mn) / (mx - mn)

    def _build_hsi_sample_scores(self) -> np.ndarray:
        n = len(self.dataset.samples)
        if n == 0:
            return np.empty((0,), dtype=np.float64)

        fg_norm = self._safe_norm(self.sample_fg_ratio.astype(np.float64))
        bd_norm = self._safe_norm(self.sample_boundary_ratio.astype(np.float64))

        # Rare-class score from per-sample class histogram (class-0 excluded).
        rare_scores = np.zeros(n, dtype=np.float64)
        if self.sample_class_hist:
            max_cls = max((h.size for h in self.sample_class_hist), default=0)
            if max_cls > 1:
                global_hist = np.zeros(max_cls, dtype=np.float64)
                for h in self.sample_class_hist:
                    if h.size:
                        global_hist[:h.size] += h.astype(np.float64)
                inv = np.zeros_like(global_hist, dtype=np.float64)
                valid = global_hist > 0
                inv[valid] = 1.0 / np.sqrt(global_hist[valid])
                inv[0] = 0.0
                for i, h in enumerate(self.sample_class_hist):
                    if h.size <= 1:
                        continue
                    w = inv[:h.size]
                    cnt = h.astype(np.float64)
                    denom = float(np.sum(cnt[1:]))
                    if denom > 0:
                        rare_scores[i] = float(np.sum(cnt[1:] * w[1:]) / denom)
        rare_norm = self._safe_norm(rare_scores)

        score = 0.45 * fg_norm + 0.30 * bd_norm + 0.25 * rare_norm
        return np.clip(score, 1e-6, None)

    def _draw_hsi_adaptive(self,
                           pool: np.ndarray,
                           n: int,
                           rng: np.random.RandomState,
                           pid_counts: Optional[Dict[str, int]] = None) -> np.ndarray:
        if n <= 0 or pool.size == 0:
            return np.empty(0, dtype=np.int64)

        if pid_counts is None:
            pid_counts = {}

        replace = bool(pool.size < n)
        selected: List[int] = []
        available = pool.copy().astype(np.int64, copy=False)
        local_counts = dict(pid_counts)

        for _ in range(n):
            if available.size == 0:
                if not replace:
                    break
                available = pool.copy().astype(np.int64, copy=False)

            base_w = self.sample_hsi_score[available]
            pids = self.sample_patient_ids[available]
            penalties = np.asarray([
                1.0 / (1.0 + self.diversity_strength * float(local_counts.get(str(pid), 0)))
                for pid in pids
            ], dtype=np.float64)
            probs = base_w * penalties
            probs_sum = float(probs.sum())
            if probs_sum <= 0:
                probs = np.ones_like(probs) / max(len(probs), 1)
            else:
                probs = probs / probs_sum

            pick_idx = int(rng.choice(len(available), size=1, replace=False, p=probs)[0])
            sid = int(available[pick_idx])
            selected.append(sid)
            pid = str(self.sample_patient_ids[sid])
            local_counts[pid] = int(local_counts.get(pid, 0) + 1)

            if not replace:
                available = np.delete(available, pick_idx)

        for k, v in local_counts.items():
            pid_counts[k] = int(v)
        return np.asarray(selected, dtype=np.int64)

    @staticmethod
    def _draw(pool: np.ndarray, n: int, rng: np.random.RandomState) -> np.ndarray:
        if n <= 0 or pool.size == 0:
            return np.empty(0, dtype=np.int64)
        replace = pool.size < n
        return rng.choice(pool, size=n, replace=replace).astype(np.int64, copy=False)

    def _draw_fg_stratified(self, n: int, rng: np.random.RandomState) -> np.ndarray:
        if n <= 0:
            return np.empty(0, dtype=np.int64)
        if not self.fg_class_to_sample_ids:
            return self._draw(self.fg_sample_ids, n, rng)

        cls_ids = sorted(self.fg_class_to_sample_ids.keys())
        pools = [self.fg_class_to_sample_ids[c] for c in cls_ids]
        freqs = np.asarray([max(1, p.size) for p in pools], dtype=np.float64)
        probs = 1.0 / np.sqrt(freqs)
        probs = probs / max(probs.sum(), 1e-8)

        quotas = np.floor(probs * n).astype(np.int64)
        rem = int(n - quotas.sum())
        if rem > 0:
            extra = rng.choice(len(quotas), size=rem, replace=True, p=probs)
            for i in extra:
                quotas[int(i)] += 1

        chunks = []
        for pool, q in zip(pools, quotas):
            if q <= 0:
                continue
            draw = self._draw(pool, int(q), rng)
            if draw.size > 0:
                chunks.append(draw)
        if not chunks:
            return self._draw(self.fg_sample_ids, n, rng)

        out = np.concatenate(chunks)
        if out.size < n:
            extra = self._draw(self.fg_sample_ids, n - out.size, rng)
            if extra.size > 0:
                out = np.concatenate([out, extra])
        if out.size > n:
            out = rng.choice(out, size=n, replace=False)
        return out.astype(np.int64, copy=False)

    def _make_sample_chunk_orders(self, rng: np.random.RandomState):
        orders = []
        cursors = []
        for sid in range(len(self.dataset.samples)):
            b, e = self.dataset._sample_chunk_ranges[sid]
            arr = np.arange(b, e, dtype=np.int64)
            if arr.size > 1:
                rng.shuffle(arr)
            orders.append(arr)
            cursors.append(0)
        return orders, cursors

    @staticmethod
    def _next_chunk_idx(sample_id: int,
                        orders: List[np.ndarray],
                        cursors: List[int],
                        rng: np.random.RandomState) -> int:
        arr = orders[int(sample_id)]
        if arr.size == 0:
            return -1
        cur = int(cursors[int(sample_id)])
        if cur >= arr.size:
            if arr.size > 1:
                rng.shuffle(arr)
            cur = 0
        idx = int(arr[cur])
        cursors[int(sample_id)] = cur + 1
        return idx

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.RandomState(self.seed + self._epoch)
        self._epoch += 1
        self._adapt_ratios()
        orders, cursors = self._make_sample_chunk_orders(rng)

        all_samples = np.arange(len(self.dataset.samples), dtype=np.int64)
        for _ in range(self.steps_per_epoch):
            ids = []
            if self.sampler_mode == 'hsi_adaptive':
                pid_counts: Dict[str, int] = {}
                fg = self._draw_hsi_adaptive(self.fg_sample_ids, self.n_fg, rng, pid_counts=pid_counts)
                bd = self._draw_hsi_adaptive(self.boundary_sample_ids, self.n_boundary, rng, pid_counts=pid_counts)
                bg = self._draw_hsi_adaptive(self.bg_sample_ids, self.n_background, rng, pid_counts=pid_counts)
                rnd = self._draw_hsi_adaptive(all_samples, self.n_random, rng, pid_counts=pid_counts)
            else:
                fg = self._draw_fg_stratified(self.n_fg, rng)
                bd = self._draw(self.boundary_sample_ids, self.n_boundary, rng)
                bg = self._draw(self.bg_sample_ids, self.n_background, rng)
                rnd = self._draw(all_samples, self.n_random, rng)
            if fg.size:
                ids.append(fg)
            if bd.size:
                ids.append(bd)
            if bg.size:
                ids.append(bg)
            if rnd.size:
                ids.append(rnd)

            if ids:
                sample_batch = np.concatenate(ids)
            else:
                sample_batch = self._draw(all_samples, self.samples_per_step, rng)

            if sample_batch.size < self.samples_per_step:
                extra = self._draw(all_samples, self.samples_per_step - sample_batch.size, rng)
                if extra.size > 0:
                    sample_batch = np.concatenate([sample_batch, extra])

            rng.shuffle(sample_batch)
            chunk_batch = []
            for sid in sample_batch[:self.samples_per_step].tolist():
                cidx = self._next_chunk_idx(int(sid), orders, cursors, rng)
                if cidx >= 0:
                    chunk_batch.append(cidx)

            if not chunk_batch:
                continue
            yield chunk_batch


def _concat_sample_patch_batches(batch):
    patches = [x[0] for x in batch]
    labels = [x[1] for x in batch]
    chunk_meta = [x[2] for x in batch]
    return torch.cat(patches, dim=0).float(), torch.cat(labels, dim=0).long(), chunk_meta


class RGBDataset(AbstractHSDataset):
    """Patch-wise RGB segmentation dataset compatible with current training pipeline."""

    def __init__(self,
                 config: Munch = None,
                 transform: Optional[Any] = None,
                 limit_pairs: Optional[int] = None,
                 max_patches_per_patient: Optional[int] = None,
                 **kwargs: Any) -> None:
        Dataset.__init__(self)

        _ensure_background_class(config)
        self.config = config
        self.data_path = config.path.data
        self.label_path = config.path.label
        self.num = config.clsf.num
        self.targets = config.clsf.targets
        self.patch_size = int(_split_cfg(config, 'patch.size', 31, legacy_key='patch_size'))
        self.margin = (self.patch_size - 1) // 2
        self.transform = transform
        self.kwargs = kwargs
        self.test_rate = float(_split_cfg(config, 'ratio.legacy.test_rate', 0.2, legacy_key='test_rate'))

        self.limit_pairs = limit_pairs
        self.max_patches_per_patient = max_patches_per_patient
        self._label_remap: Optional[np.ndarray] = None

        self._num_patients = 0
        self._patient_names: Dict[int, str] = {}
        self.patch_pg_tg_boundary: Optional[np.ndarray] = None
        self.patch_fg_roi_mask: Optional[np.ndarray] = None

        self._build_rgb_pipeline()

    @staticmethod
    def _extract_patient_id(filepath: str) -> str:
        basename = os.path.basename(filepath)
        match = re.match(r'^(.+?)_(\d{8})_', basename)
        if match:
            return match.group(1).lower()
        return os.path.splitext(basename)[0].lower()

    @staticmethod
    def _base_name_from_rgb_file(filename: str) -> str:
        if filename.endswith('_Merged_rgb.png'):
            return filename[:-len('_Merged_rgb.png')]
        stem = os.path.splitext(filename)[0]
        if stem.endswith('_rgb'):
            return stem[:-len('_rgb')]
        return stem

    def _pair_data_and_labels_(self) -> List[Tuple[str, str]]:
        data_files = sorted([f for f in os.listdir(self.data_path) if f.lower().endswith('.png')])
        label_files = sorted([f for f in os.listdir(self.label_path) if f.lower().endswith('_gt.npy')])
        label_set = set(label_files)

        pairs: List[Tuple[str, str]] = []
        for data_file in data_files:
            base = self._base_name_from_rgb_file(data_file)
            label_file = f"{base}_gt.npy"
            if label_file in label_set:
                pairs.append((
                    os.path.join(self.data_path, data_file),
                    os.path.join(self.label_path, label_file),
                ))
        return pairs

    def _build_label_remap(self) -> np.ndarray:
        remap_cfg = getattr(self.config.clsf, 'label_remap', None)
        if remap_cfg is None:
            return np.arange(max(int(self.num), 1), dtype=np.int32)

        if isinstance(remap_cfg, (dict, Munch)):
            remap_items = dict(remap_cfg)
            if not remap_items:
                raise ValueError('clsf.label_remap cannot be an empty dict')
            normalized: Dict[int, int] = {}
            for raw_k, mapped_v in remap_items.items():
                normalized[int(raw_k)] = int(mapped_v)
            max_key = max(normalized.keys())
            remap = np.zeros(max_key + 1, dtype=np.int32)
            for k, v in normalized.items():
                if k < 0 or v < 0:
                    raise ValueError(f'Invalid label remap entry: {k}->{v}')
                remap[k] = v
        else:
            remap = np.asarray(remap_cfg, dtype=np.int32).reshape(-1)
            if remap.size == 0:
                raise ValueError('clsf.label_remap cannot be empty')
            if np.any(remap < 0):
                raise ValueError('clsf.label_remap values must be non-negative')
            if remap.size == max(int(self.num) - 1, 0):
                remap = np.concatenate([np.array([0], dtype=np.int32), remap])

        if np.max(remap) >= self.num:
            raise ValueError(f"clsf.label_remap contains mapped label >= clsf.num ({self.num})")
        return remap.astype(np.int32, copy=False)

    def _apply_label_remap(self, labels_raw: np.ndarray,
                           remap: np.ndarray,
                           label_file: str) -> np.ndarray:
        if labels_raw.ndim == 3 and labels_raw.shape[-1] == 1:
            labels_raw = labels_raw[..., 0]
        if labels_raw.ndim != 2:
            raise ValueError(f'label map must be 2D, got {labels_raw.shape} from {label_file}')

        strict_remap = bool(getattr(self.config.clsf, 'strict_label_remap', True))
        unknown_policy = str(getattr(self.config.clsf, 'unknown_label_policy', 'error')).strip().lower()
        if unknown_policy not in {'error', 'map_to_bg'}:
            raise ValueError("clsf.unknown_label_policy must be 'error' or 'map_to_bg'")

        if int(labels_raw.min()) < 0:
            raise ValueError(f"{os.path.basename(label_file)} has negative labels")

        unknown_mask = labels_raw >= len(remap)
        unknown_count = int(unknown_mask.sum())
        if unknown_count > 0 and (strict_remap or unknown_policy == 'error'):
            unknown_values = np.unique(labels_raw[unknown_mask])[:10].tolist()
            raise ValueError(
                f"{os.path.basename(label_file)} has unknown labels {unknown_values}, remap_max={len(remap)-1}"
            )
        if unknown_count > 0 and unknown_policy == 'map_to_bg':
            unknown_values = np.unique(labels_raw[unknown_mask])[:10].tolist()
            print(
                f"\t\tWARNING: {os.path.basename(label_file)} maps unknown labels "
                f"{unknown_values} to BG(0), pixels={unknown_count:,}"
            )

        labels_clipped = np.clip(labels_raw, 0, len(remap) - 1)
        labels = remap[labels_clipped]
        if unknown_count > 0 and unknown_policy == 'map_to_bg':
            labels[unknown_mask] = 0
        return labels.astype(np.int32, copy=False)

    @staticmethod
    def _build_boundary_map(labels: np.ndarray) -> np.ndarray:
        edge = np.zeros_like(labels, dtype=np.bool_)
        edge[:, 1:] |= (labels[:, 1:] != labels[:, :-1])
        edge[1:, :] |= (labels[1:, :] != labels[:-1, :])
        return edge

    def _cap_patient_centers(self,
                             rows: np.ndarray,
                             cols: np.ndarray,
                             rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
        if self.max_patches_per_patient is None:
            return rows, cols
        cap = int(self.max_patches_per_patient)
        n = int(rows.size)
        if cap <= 0 or n <= cap:
            return rows, cols
        sel = rng.choice(n, size=cap, replace=False)
        return rows[sel], cols[sel]

    def _build_rgb_pipeline(self) -> None:
        pairs = self._pair_data_and_labels_()
        if self.limit_pairs is not None:
            pairs = pairs[: self.limit_pairs]
        if not pairs:
            raise RuntimeError(
                f'No RGB/mask pairs found. data={self.data_path}, label={self.label_path}'
            )

        self._label_remap = self._build_label_remap()

        rng = np.random.RandomState(int(getattr(self.config.split, 'split_seed', 350234)))
        self._patient_padded_data = []
        self._patient_padded_labels = []
        patient_map: Dict[str, int] = {}

        all_patch_indices = []
        all_patch_labels = []
        all_patient_groups = []
        all_boundary_flags = []
        all_fg_roi_flags = []

        for data_file, label_file in pairs:
            img_bgr = cv2.imread(data_file, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError(f'Cannot read RGB image: {data_file}')
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            labels_raw = np.load(label_file).astype(np.int32)
            labels = self._apply_label_remap(labels_raw, self._label_remap, label_file)
            if img.shape[:2] != labels.shape[:2]:
                raise ValueError(
                    f'Spatial mismatch for {os.path.basename(data_file)}: '
                    f'img={img.shape[:2]} vs label={labels.shape[:2]}'
                )

            patient_id = self._extract_patient_id(data_file)
            if patient_id not in patient_map:
                patient_map[patient_id] = len(patient_map)
            pid = patient_map[patient_id]

            padded_data = self._pad_with_zeros(img, self.margin)
            padded_labels = np.full(
                (labels.shape[0] + 2 * self.margin, labels.shape[1] + 2 * self.margin),
                fill_value=255,
                dtype=np.int32,
            )
            padded_labels[
                self.margin:self.margin + labels.shape[0],
                self.margin:self.margin + labels.shape[1],
            ] = labels

            self._patient_padded_data.append(padded_data)
            self._patient_padded_labels.append(padded_labels)

            rows, cols = np.indices(labels.shape)
            rows = rows.reshape(-1)
            cols = cols.reshape(-1)
            rows, cols = self._cap_patient_centers(rows, cols, rng)

            boundary_map = self._build_boundary_map(labels)
            fg_map = labels > 0
            dilation = int(max(0, getattr(self.config.split, 'boundary_sampling_dilation', 1)))
            fg_roi = binary_dilation(fg_map, iterations=dilation) if dilation > 0 else fg_map

            centers = np.stack([rows + self.margin, cols + self.margin], axis=1).astype(np.int32)
            patch_indices = np.concatenate([
                np.full((centers.shape[0], 1), pid, dtype=np.int32),
                centers,
            ], axis=1)

            all_patch_indices.append(patch_indices)
            all_patch_labels.append(labels[rows, cols].astype(np.int32))
            all_patient_groups.append(np.full(rows.shape[0], pid, dtype=np.int32))
            all_boundary_flags.append(boundary_map[rows, cols].astype(np.bool_))
            all_fg_roi_flags.append(fg_roi[rows, cols].astype(np.bool_))

        self.patch_indices = np.concatenate(all_patch_indices, axis=0)
        self.patch_labels = np.concatenate(all_patch_labels, axis=0)
        self.patch_patient_groups = np.concatenate(all_patient_groups, axis=0)
        self.patch_pg_tg_boundary = np.concatenate(all_boundary_flags, axis=0)
        self.patch_fg_roi_mask = np.concatenate(all_fg_roi_flags, axis=0)

        self._patient_names = {v: k for k, v in patient_map.items()}
        self._num_patients = len(self._patient_names)
        tprint(
            f"RGBDataset loaded {len(pairs)} pairs, {self._num_patients} patients, "
            f"{len(self.patch_indices):,} patches"
        )

    def _get_patch_(self, idx: int) -> np.ndarray:
        p, r, c = self.patch_indices[idx]
        padded = self._patient_padded_data[p]
        return padded[
            r - self.margin:r + self.margin + 1,
            c - self.margin:c + self.margin + 1,
            :
        ].copy()

    def _get_label_patch_(self, idx: int) -> np.ndarray:
        p, r, c = self.patch_indices[idx]
        padded = self._patient_padded_labels[p]
        return padded[
            r - self.margin:r + self.margin + 1,
            c - self.margin:c + self.margin + 1,
        ].copy()

    def _load_data(self) -> None:
        pass

    def _preprocess_data(self) -> None:
        pass

    def _patient_level_split(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        total_indices = np.arange(len(self.patch_indices), dtype=np.int64)
        patient_ids = np.unique(self.patch_patient_groups)
        split_seed = int(getattr(self.config.split, 'split_seed', 350234))

        if patient_ids.size < 3:
            train_idx, test_idx = train_test_split(
                total_indices,
                test_size=float(getattr(self.config.split, 'test_rate', 0.1)),
                random_state=split_seed,
                stratify=self.patch_labels,
            )
            train_idx, val_idx = train_test_split(
                train_idx,
                test_size=float(getattr(self.config.split, 'val_rate', 0.1)),
                random_state=split_seed,
                stratify=self.patch_labels[train_idx],
            )
            return train_idx, val_idx, test_idx

        train_ratio = float(getattr(self.config.split, 'train_ratio', 0.8))
        val_ratio = float(getattr(self.config.split, 'val_ratio', 0.1))
        test_ratio = float(getattr(self.config.split, 'test_ratio', 0.1))
        ratio_sum = max(train_ratio + val_ratio + test_ratio, 1e-8)
        train_ratio /= ratio_sum
        val_ratio /= ratio_sum
        test_ratio /= ratio_sum

        fg_ratio = []
        for pid in patient_ids:
            m = self.patch_patient_groups == pid
            fg_ratio.append(float((self.patch_labels[m] > 0).mean()))
        fg_ratio = np.asarray(fg_ratio, dtype=np.float64)

        strata = None
        if patient_ids.size >= 6:
            bins = np.quantile(fg_ratio, [0.33, 0.66])
            if np.unique(bins).size > 0:
                strata = np.digitize(fg_ratio, bins, right=False)

        try:
            trainval_patients, test_patients = train_test_split(
                patient_ids,
                test_size=test_ratio,
                random_state=split_seed,
                stratify=strata,
            )
        except Exception:
            trainval_patients, test_patients = train_test_split(
                patient_ids,
                test_size=test_ratio,
                random_state=split_seed,
                stratify=None,
            )

        if len(trainval_patients) < 2:
            trainval_patients = patient_ids
            test_patients = np.asarray([], dtype=patient_ids.dtype)

        val_rel = val_ratio / max(train_ratio + val_ratio, 1e-8)
        val_rel = float(np.clip(val_rel, 0.05, 0.5))
        try:
            train_patients, val_patients = train_test_split(
                trainval_patients,
                test_size=val_rel,
                random_state=split_seed,
                stratify=None,
            )
        except Exception:
            train_patients = trainval_patients
            val_patients = np.asarray([], dtype=patient_ids.dtype)

        train_idx = total_indices[np.isin(self.patch_patient_groups, train_patients)]
        val_idx = total_indices[np.isin(self.patch_patient_groups, val_patients)]
        test_idx = total_indices[np.isin(self.patch_patient_groups, test_patients)]

        if val_idx.size == 0 and train_idx.size > 2:
            train_idx, val_idx = train_test_split(
                train_idx,
                test_size=min(0.15, max(0.05, val_ratio)),
                random_state=split_seed,
                stratify=self.patch_labels[train_idx],
            )
        if test_idx.size == 0 and train_idx.size > 2:
            train_idx, test_idx = train_test_split(
                train_idx,
                test_size=min(0.15, max(0.05, test_ratio)),
                random_state=split_seed,
                stratify=self.patch_labels[train_idx],
            )

        return train_idx, val_idx, test_idx

    def _create_data_loader_(self, num_workers=4, batch_size=None, pin_memory=True,
                             prefetch_factor=2, persistent_workers=False):
        train_idx, val_idx, test_idx = self._patient_level_split()

        train_subset = _IndexedSubset(self, train_idx)
        val_subset = _IndexedSubset(self, val_idx)
        test_subset = _IndexedSubset(self, test_idx)

        if batch_size is None:
            batch_size = int(getattr(self.config.split, 'batch_size', 64))
        actual_pin_memory = pin_memory and torch.cuda.is_available()

        train_loader = torch.utils.data.DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=actual_pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers and (num_workers > 0),
            drop_last=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=actual_pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers and (num_workers > 0),
            drop_last=False,
        )
        test_loader = torch.utils.data.DataLoader(
            test_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=actual_pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers and (num_workers > 0),
            drop_last=False,
        )

        print(f"RGB train/val/test samples: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
        print(f"RGB class distribution(train): {np.bincount(self.patch_labels[train_idx], minlength=self.num)}")
        return train_loader, val_loader, test_loader

    def _create_cv_data_loaders_(self, n_folds=5, num_workers=4, batch_size=None,
                                 pin_memory=True, prefetch_factor=2,
                                 persistent_workers=False):
        total_indices = np.arange(len(self.patch_indices))
        split_seed = int(getattr(self.config.split, 'split_seed', 350234))
        sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=split_seed)

        if batch_size is None:
            batch_size = int(getattr(self.config.split, 'batch_size', 64))
        actual_pin_memory = pin_memory and torch.cuda.is_available()

        fold_loaders = []
        for train_idx, test_idx in sgkf.split(total_indices, self.patch_labels, groups=self.patch_patient_groups):
            train_subset = _IndexedSubset(self, train_idx)
            test_subset = _IndexedSubset(self, test_idx)

            train_loader = torch.utils.data.DataLoader(
                train_subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=actual_pin_memory,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
                persistent_workers=persistent_workers and (num_workers > 0),
                drop_last=True,
            )
            test_loader = torch.utils.data.DataLoader(
                test_subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=actual_pin_memory,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
                persistent_workers=persistent_workers and (num_workers > 0),
                drop_last=False,
            )
            fold_loaders.append((train_loader, test_loader))

        return fold_loaders

    def get_loaders(self, *args, **kwargs):
        return self._create_data_loader_(*args, **kwargs)

    def get_cv_loaders(self, *args, **kwargs):
        return self._create_cv_data_loaders_(*args, **kwargs)

    def describe(self, top_k: int = 5) -> None:
        cls_counts = np.bincount(self.patch_labels, minlength=self.num)
        print('[RGBDataset Summary]')
        print(f"  patients: {self._num_patients}, patches: {len(self.patch_indices)}")
        print(f"  patch size: ({self.patch_size}, {self.patch_size}), num_classes: {self.num}")
        print(f"  class counts: {cls_counts.tolist()}")
        uniq, cnt = np.unique(self.patch_patient_groups, return_counts=True)
        pairs = sorted(list(zip(uniq.tolist(), cnt.tolist())), key=lambda x: x[1], reverse=True)
        print(f"  top-{top_k} patients by patches: {pairs[:top_k]}")

class MatHSDataset(AbstractHSDataset):
    """
    Hyperspectral dataset loader for .mat file format.
    
    Handles datasets stored in MATLAB .mat files, commonly used for
    hyperspectral image datasets.
    """
    
    def __init__(self, 
                 config: Munch = None,
                 transform: Optional[Any] = None,
                  **kwargs: Any) -> None:
        """
        Initialize a MATLAB format hyperspectral dataset.
        
        Args:
            data_key: Key in .mat file for hyperspectral data
            label_key: Key in .mat file for label data
            pca_components: Number of PCA components for dimensionality reduction
            See parent class for other parameters
        """
        
        self.data_key = config.key.data
        self.label_key = config.key.label
        self.batch_size = int(_split_cfg(config, 'batch.train_batch_size', 64, legacy_key='batch_size'))
        self.pin_memory = config.memory.pin_memory
        self.pca_components = config.preprocess.pca_components
        super().__init__(config, transform, **kwargs)

    def _load_data(self) -> None:
        """Load data from .mat files using scipy.io"""
        import scipy.io as sio
        
        try:
            # Load data and labels from .mat files
            data_mat = sio.loadmat(self.data_path)
            label_mat = sio.loadmat(self.label_path)
            
            # Extract data using specified keys
            self.raw_data = data_mat[self.data_key]
            self.raw_labels = label_mat[self.label_key]

            # Execute a clip from min 1% to max 99%.
            self.raw_data = np.nan_to_num(self.raw_data, nan=0.0, posinf=0.0, neginf=0.0)
            self.raw_data = np.clip(self.raw_data, a_min=np.percentile(self.raw_data, 1), a_max=np.percentile(self.raw_data, 99))
            
            # Handle possible singleton dimensions
            if self.raw_labels.ndim == 3 and self.raw_labels.shape[-1] == 1:
                self.raw_labels = self.raw_labels.squeeze(-1)
                
            tprint(f"Loaded .mat data: {self.raw_data.shape}, labels: {self.raw_labels.shape}")
            
        except KeyError as e:
            raise ValueError(f"Missing key in .mat file: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading .mat files: {e}")

    def _preprocess_data(self):
        """
        Preprocess data with PCA dimensionality reduction and normalization.
        
        OPTIMIZATION: Randomized SVD for fast PCA when n_components << n_features.
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        # Reshape data for PCA (flatten spatial dimensions)
        h, w, c = self.raw_data.shape
        flat_data = self.raw_data.reshape(-1, c)
        
        # Handle outliers and normalization
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(flat_data).astype(np.float32)
        
        # Apply PCA with randomized SVD — much faster when pca_components << c
        pca = PCA(n_components=self.pca_components,
                  svd_solver='randomized', random_state=42)
        pca.fit(scaled_data)
        
        # Manual projection in float32 to avoid internal float64 upcast
        components = pca.components_.astype(np.float32)
        mean = pca.mean_.astype(np.float32)
        pca_data = (scaled_data - mean) @ components.T
        
        # Reshape back to original spatial dimensions
        self.processed_data = pca_data.reshape(h, w, self.pca_components)
        return self.processed_data, pca
    
    def create_data_loader(self, num_workers=0, batch_size=None, pin_memory=True,
                           prefetch_factor=2, persistent_workers=False):
        """
        Create PyTorch DataLoaders using the parent class's patch-based pipeline.
        
        MatHSDataset reuses the same _create_patches / _get_patch approach as
        NpyHSDataset (patches and labels are already built by AbstractHSDataset.__init__).
        Because MatHSDataset has no patient-level grouping, a stratified random
        split is used instead.
        
        Returns:
            Tuple of (train_loader, test_loader)
        """
        total_indices = np.arange(len(self.patch_indices))
        
        train_idx, test_idx = train_test_split(
            total_indices,
            test_size=self.test_rate,
            random_state=350234,
            stratify=self.patch_labels
        )
        
        # check class coverage
        train_classes = set(np.unique(self.patch_labels[train_idx]))
        test_classes = set(np.unique(self.patch_labels[test_idx]))
        all_classes = set(np.unique(self.patch_labels))
        if train_classes != all_classes:
            print(f"  WARNING: Training set missing classes: {all_classes - train_classes}")
        if test_classes != all_classes:
            print(f"  WARNING: Test set missing classes: {all_classes - test_classes}")
        
        train_subset = _IndexedSubset(self, train_idx)
        test_subset = _IndexedSubset(self, test_idx)
        
        if batch_size is None:
            batch_size = self.batch_size if hasattr(self, 'batch_size') else 32
        actual_pin_memory = pin_memory and torch.cuda.is_available()
        
        train_loader = torch.utils.data.DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=actual_pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers and (num_workers > 0),
            drop_last=True,
        )
        test_loader = torch.utils.data.DataLoader(
            test_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=actual_pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers and (num_workers > 0),
            drop_last=False,
        )
        
        print(f"Training set: {len(train_loader)} batches ({len(train_idx)} samples)")
        print(f"Test set: {len(test_loader)} batches ({len(test_idx)} samples)")
        return train_loader, test_loader

class _IndexedSubset(Dataset):
    """
    indexed subset wrapper class.
    """
    def __init__(self, dataset: AbstractHSDataset, indices: np.ndarray):
        self.dataset = dataset
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx: int):
        original_idx = self.indices[idx]
        return self.dataset[original_idx]


class _FixedRatioBatchSampler(Sampler[List[int]]):
    """Four-route mixed sampler over subset positions.

    Each batch mixes:
    - foreground centers (rare-class aware)
    - boundary hard examples
    - explicit background centers
    - global random centers from all positions
    """

    def __init__(self,
                 fg_positions: np.ndarray,
                 fg_class_positions: Optional[Dict[int, np.ndarray]],
                 boundary_positions: np.ndarray,
                 background_positions: np.ndarray,
                 all_positions: np.ndarray,
                 position_groups: Optional[np.ndarray],
                 batch_size: int,
                 ratio_fg: float = 0.25,
                 ratio_boundary: float = 0.15,
                 ratio_background: float = 0.35,
                 ratio_random: Optional[float] = None,
                 sampler_mode: str = 'fixed',
                 target_fg_ratio: float = 0.20,
                 target_boundary_ratio: float = 0.08,
                 target_background_ratio: float = 0.35,
                 adapt_momentum: float = 0.9,
                 min_fg_ratio: float = 0.12,
                 max_fg_ratio: float = 0.28,
                 empirical_fg_ratio: float = 0.18,
                 fg_inverse_pow: float = 1.0,
                 fg_min_per_class: int = 0,
                 diversity_strength: float = 0.75,
                 steps_per_epoch: Optional[int] = None,
                 seed: int = 350234):
        self.fg_positions = np.asarray(fg_positions, dtype=np.int64)
        self.fg_class_positions = {
            int(k): np.asarray(v, dtype=np.int64)
            for k, v in (fg_class_positions or {}).items()
            if np.asarray(v).size > 0
        }
        self.boundary_positions = np.asarray(boundary_positions, dtype=np.int64)
        self.background_positions = np.asarray(background_positions, dtype=np.int64)
        self.all_positions = np.asarray(all_positions, dtype=np.int64)
        if self.all_positions.size == 0:
            self.all_positions = np.unique(
                np.concatenate([
                    self.fg_positions,
                    self.boundary_positions,
                    self.background_positions,
                ])
            ) if (self.fg_positions.size + self.boundary_positions.size + self.background_positions.size) > 0 else np.empty((0,), dtype=np.int64)

        self.position_groups = None if position_groups is None else np.asarray(position_groups, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.sampler_mode = str(sampler_mode).lower()
        self.target_fg_ratio = float(np.clip(target_fg_ratio, 0.0, 1.0))
        self.target_boundary_ratio = float(np.clip(target_boundary_ratio, 0.0, 1.0))
        self.target_background_ratio = float(np.clip(target_background_ratio, 0.0, 1.0))
        self.adapt_momentum = float(np.clip(adapt_momentum, 0.0, 0.999))
        self.min_fg_ratio = float(np.clip(min_fg_ratio, 0.0, 1.0))
        self.max_fg_ratio = float(np.clip(max_fg_ratio, self.min_fg_ratio, 1.0))
        self.empirical_fg_ratio = float(np.clip(empirical_fg_ratio, 0.0, 1.0))
        self.fg_inverse_pow = float(max(0.0, fg_inverse_pow))
        self.fg_min_per_class = int(max(0, fg_min_per_class))
        self.diversity_strength = float(max(0.0, diversity_strength))
        self._epoch = 0

        if self.position_groups is not None:
            max_pos = int(self.all_positions.max()) if self.all_positions.size > 0 else -1
            if max_pos >= self.position_groups.shape[0]:
                raise ValueError("position_groups must align with subset positions used by sampler")

        if self.all_positions.size == 0:
            raise ValueError("Sampler received empty pools")

        self._ratio_fg = float(np.clip(ratio_fg, 0.0, 1.0))
        self._ratio_boundary = float(np.clip(ratio_boundary, 0.0, 1.0))
        self._ratio_background = float(np.clip(ratio_background, 0.0, 1.0))
        self._auto_random_ratio = ratio_random is None
        self._ratio_random = float(np.clip(0.0 if ratio_random is None else ratio_random, 0.0, 1.0))
        self._refresh_quota()

        if steps_per_epoch is None or steps_per_epoch <= 0:
            max_pool = max(len(self.all_positions), 1)
            self.steps_per_epoch = max(1, max_pool // max(self.batch_size, 1))
        else:
            self.steps_per_epoch = int(steps_per_epoch)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def get_current_ratios(self) -> Tuple[float, float, float, float]:
        return (
            float(self._ratio_fg),
            float(self._ratio_boundary),
            float(self._ratio_background),
            float(self._ratio_random if not self._auto_random_ratio else max(0.0, 1.0 - self._ratio_fg - self._ratio_boundary - self._ratio_background)),
        )

    def _refresh_quota(self) -> None:
        ratio_random = self._ratio_random
        if self._auto_random_ratio:
            ratio_random = max(0.0, 1.0 - self._ratio_fg - self._ratio_boundary - self._ratio_background)

        shares = np.asarray([
            self._ratio_fg,
            self._ratio_boundary,
            self._ratio_background,
            ratio_random,
        ], dtype=np.float64)

        if float(shares.sum()) <= 1e-12:
            shares = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        shares = shares / shares.sum()

        raw = shares * float(self.batch_size)
        quotas = np.floor(raw).astype(np.int64)
        rem = int(self.batch_size - int(quotas.sum()))
        if rem > 0:
            frac = raw - quotas.astype(np.float64)
            order = np.argsort(frac)[::-1]
            for i in order[:rem]:
                quotas[int(i)] += 1

        self.n_fg = int(quotas[0])
        self.n_boundary = int(quotas[1])
        self.n_background = int(quotas[2])
        self.n_random = int(quotas[3])

        # Guard against ratio quantization collapse on small batch sizes.
        self._enforce_minimum_quota()

    def _borrow_one_from_other_routes(self, candidates: List[str]) -> bool:
        for name in candidates:
            cur = int(getattr(self, name, 0))
            if cur > 0:
                setattr(self, name, cur - 1)
                return True
        return False

    def _enforce_minimum_quota(self) -> None:
        # Keep at least one FG sample when FG route is enabled and pool exists.
        if self.batch_size >= 2 and self.fg_positions.size > 0 and self._ratio_fg > 0.0 and self.n_fg <= 0:
            if self._borrow_one_from_other_routes(['n_random', 'n_background', 'n_boundary']):
                self.n_fg = 1

        # Keep at least one boundary sample when boundary route is enabled and pool exists.
        if self.batch_size >= 2 and self.boundary_positions.size > 0 and self._ratio_boundary > 0.0 and self.n_boundary <= 0:
            if self._borrow_one_from_other_routes(['n_random', 'n_background', 'n_fg']):
                self.n_boundary = 1

    def _adapt_ratios(self) -> None:
        if self.sampler_mode not in {'adaptive', 'hsi_adaptive'}:
            return

        desired_fg = 0.5 * self.empirical_fg_ratio + 0.5 * self.target_fg_ratio
        desired_fg = float(np.clip(desired_fg, self.min_fg_ratio, self.max_fg_ratio))
        desired_bd = float(np.clip(self.target_boundary_ratio, 0.0, 1.0))
        desired_bg = float(np.clip(self.target_background_ratio, 0.0, 1.0))

        m = self.adapt_momentum
        self._ratio_fg = float(np.clip(m * self._ratio_fg + (1.0 - m) * desired_fg,
                                       self.min_fg_ratio, self.max_fg_ratio))
        self._ratio_boundary = float(np.clip(m * self._ratio_boundary + (1.0 - m) * desired_bd,
                                             0.0, 1.0))
        self._ratio_background = float(np.clip(m * self._ratio_background + (1.0 - m) * desired_bg,
                                               0.0, 1.0))
        self._refresh_quota()

    @staticmethod
    def _draw(pool: np.ndarray, n: int, rng: np.random.RandomState) -> np.ndarray:
        if n <= 0:
            return np.empty(0, dtype=np.int64)
        if pool.size == 0:
            return np.empty(0, dtype=np.int64)
        if pool.size >= n:
            return rng.choice(pool, size=n, replace=False)
        return rng.choice(pool, size=n, replace=True)

    def _draw_group_diverse(self, pool: np.ndarray, n: int,
                            rng: np.random.RandomState,
                            group_counts: Optional[Dict[int, int]] = None) -> np.ndarray:
        if n <= 0:
            return np.empty(0, dtype=np.int64)
        if pool.size == 0 or self.position_groups is None:
            return self._draw(pool, n, rng)

        if group_counts is None:
            group_counts = {}
        replace = bool(pool.size < n)
        selected = []
        available = pool.copy().astype(np.int64, copy=False)

        for _ in range(n):
            if available.size == 0:
                if not replace:
                    break
                available = pool.copy().astype(np.int64, copy=False)

            groups = self.position_groups[available]
            uniq, cnt = np.unique(groups, return_counts=True)
            freq_map = {int(g): int(c) for g, c in zip(uniq, cnt)}
            base = np.asarray([
                1.0 / np.sqrt(float(freq_map.get(int(g), 1))) for g in groups
            ], dtype=np.float64)
            penalty = np.asarray([
                1.0 / (1.0 + self.diversity_strength * float(group_counts.get(int(g), 0)))
                for g in groups
            ], dtype=np.float64)
            probs = base * penalty
            s = float(probs.sum())
            if s <= 0:
                probs = np.ones_like(probs) / max(len(probs), 1)
            else:
                probs = probs / s

            pick_idx = int(rng.choice(len(available), size=1, replace=False, p=probs)[0])
            pos = int(available[pick_idx])
            selected.append(pos)
            gid = int(self.position_groups[pos])
            group_counts[gid] = int(group_counts.get(gid, 0) + 1)

            if not replace:
                available = np.delete(available, pick_idx)

        return np.asarray(selected, dtype=np.int64)

    def _draw_fg_stratified(self, n: int, rng: np.random.RandomState) -> np.ndarray:
        if n <= 0:
            return np.empty(0, dtype=np.int64)
        if not self.fg_class_positions:
            return self._draw(self.fg_positions, n, rng)

        cls_ids = sorted(self.fg_class_positions.keys())
        pools = [self.fg_class_positions[c] for c in cls_ids if self.fg_class_positions[c].size > 0]
        if not pools:
            return self._draw(self.fg_positions, n, rng)

        freqs = np.asarray([max(1, p.size) for p in pools], dtype=np.float64)
        probs = 1.0 / np.power(freqs, max(self.fg_inverse_pow, 1e-8))
        probs = probs / max(probs.sum(), 1e-8)

        quotas = np.zeros(len(pools), dtype=np.int64)
        if self.fg_min_per_class > 0:
            base = int(min(self.fg_min_per_class, n // len(pools)))
            if base > 0:
                quotas += base

        rem = int(max(0, n - quotas.sum()))
        if rem > 0:
            extra_ids = rng.choice(len(quotas), size=rem, replace=True, p=probs)
            for i in extra_ids:
                quotas[int(i)] += 1

        chunks = []
        for pool, q in zip(pools, quotas):
            if q <= 0:
                continue
            draw = self._draw(pool, int(q), rng)
            if draw.size > 0:
                chunks.append(draw)

        if not chunks:
            return self._draw(self.fg_positions, n, rng)

        fg = np.concatenate(chunks)
        if fg.size < n:
            extra = self._draw(self.fg_positions, int(n - fg.size), rng)
            if extra.size > 0:
                fg = np.concatenate([fg, extra])
        if fg.size > n:
            fg = rng.choice(fg, size=n, replace=False)
        return fg

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.RandomState(self.seed + self._epoch)
        self._adapt_ratios()
        self._epoch += 1
        for _ in range(self.steps_per_epoch):
            ids = []
            if self.sampler_mode == 'hsi_adaptive':
                group_counts: Dict[int, int] = {}
                fg = self._draw_fg_stratified(self.n_fg, rng)
                if fg.size > 0 and self.position_groups is not None:
                    fg = self._draw_group_diverse(fg, int(fg.size), rng, group_counts=group_counts)
                bd = self._draw_group_diverse(self.boundary_positions, self.n_boundary, rng, group_counts=group_counts)
                bg = self._draw_group_diverse(self.background_positions, self.n_background, rng, group_counts=group_counts)
                rnd = self._draw_group_diverse(self.all_positions, self.n_random, rng, group_counts=group_counts)
            else:
                fg = self._draw_fg_stratified(self.n_fg, rng)
                bd = self._draw(self.boundary_positions, self.n_boundary, rng)
                bg = self._draw(self.background_positions, self.n_background, rng)
                rnd = self._draw(self.all_positions, self.n_random, rng)

            if fg.size:
                ids.append(fg)
            if bd.size:
                ids.append(bd)
            if bg.size:
                ids.append(bg)
            if rnd.size:
                ids.append(rnd)

            if not ids:
                batch = self._draw(self.all_positions, self.batch_size, rng)
                if batch.size == 0:
                    continue
                yield batch[:self.batch_size].tolist()
                continue

            batch = np.concatenate(ids)
            if batch.size < self.batch_size:
                extra = self._draw(self.all_positions, self.batch_size - batch.size, rng)
                if extra.size:
                    batch = np.concatenate([batch, extra])

            rng.shuffle(batch)
            yield batch[:self.batch_size].tolist()


