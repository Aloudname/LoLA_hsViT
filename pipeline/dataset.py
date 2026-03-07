# dataset.py - Hyperspectral Dataset Handling.
# todo:
# - Implement class distribution analysis and visualization methods.
# - Try various seeds for a balanced label distribution in line 542 create_dataloader.
import os, re, torch, numpy as np, warnings

from munch import Munch
from pipeline.monitor import tprint
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional, List
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset, WeightedRandomSampler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")


class HSPreprocessor(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible preprocessor for hyperspectral data.

    Pipeline: Z-score normalization -> PCA dimensionality reduction.
    Fits on non-background pixels, then transforms raw per-patient data.

    Follows sklearn's ``fit`` / ``transform`` API::

        preprocessor = HSPreprocessor(pca_components=48)
        preprocessor.fit(non_bg_pixels)             # compute stats + fit PCA
        processed = preprocessor.transform(raw_data) # normalize + project

    Attributes (set after ``fit``):
        global_mean_                  : np.ndarray, shape (n_bands,)
        global_std_                   : np.ndarray, shape (n_bands,)
        pca_                          : sklearn PCA object or None
        pca_components_               : np.ndarray (n_components, n_bands) or None
        pca_mean_                     : np.ndarray (n_bands,) or None
        pca_explained_variance_ratio_ : np.ndarray or None
        n_features_out_               : int, output channel count
    """

    def __init__(self, pca_components=48, max_fit_samples=2_000_000,
                 max_pca_samples=500_000, random_state=350234):
        self.pca_components = pca_components
        self.max_fit_samples = max_fit_samples
        self.max_pca_samples = max_pca_samples
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        Fit normalization statistics and PCA on raw pixel data.

        Args:
            X : np.ndarray, shape (n_pixels, n_bands) -- non-background pixels.
            y : ignored (sklearn convention).

        Returns:
            self
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
        tprint(f"  normalization fit on {len(fit_data):,} non-bg pixels")

        # PCA fit on Z-score normalized pixels, capped at max_pca_samples
        # O(n * p^2); 500K is sufficient for stable eigenvectors
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

            explained = sum(self.pca_.explained_variance_ratio_) * 100
            tprint(f"  pca: {c} -> {self.pca_components} channels, "
                   f"explained variance: {explained:.1f}%, "
                   f"fit on {len(pca_fit):,} pixels")

            # Pre-compute float32 projection matrix for fast transform
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

        OPTIMIZATION: manual float32 matmul avoids sklearn's internal
        float64 upcast, halving memory bandwidth.

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

        # Z-score normalization
        result = (flat - self.global_mean_) / self.global_std_

        # PCA projection (float32 matmul, no upcast)
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

        self.config = config  # store for get_dataset_info() etc.
        self.data_path = config.path.data
        self.label_path = config.path.label
        self.num = config.clsf.num
        self.targets = config.clsf.targets
        self.patch_size = config.split.patch_size
        self.margin = (config.split.patch_size - 1) // 2
        self.transform = transform
        self.kwargs = kwargs
        self.test_rate = config.split.test_rate
        self.split_seed = int(getattr(config.split, 'split_seed', 350234))

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
        if not np.all((unique_labels >= 0) & (unique_labels <= self.num)):
            raise ValueError(
                f"Labels contain values outside valid range [0, {self.num}]"
            )
        print("Raw label validated!")

    def _create_patches(self) -> None:
        """
        Extract spatial patches from preprocessed data.
        
        Patches are extracted with zero-padding at image boundaries. Only patches
        with non-zero labels are included (background/zero labels are excluded).
        
        OPTIMIZATION: Padding is done once and cached to avoid re-padding for each sample.
        OPTIMIZATION: Vectorized with np.where instead of nested Python for-loop.
        """
        # Add zero padding around the image (ONCE at initialization)
        self.padded_data = self._pad_with_zeros(self.processed_data, self.margin)
        
        # Vectorized: find all non-background pixel positions at once
        rows, cols = np.where(self.raw_labels > 0)
        
        # Convert to padded coordinates (offset by margin)
        self.patch_indices = np.stack(
            [rows + self.margin, cols + self.margin], axis=1
        ).astype(np.int32)
        
        # Convert labels to 0-based index
        self.patch_labels = (self.raw_labels[rows, cols] - 1).astype(np.int32)
        tprint(f"Created {len(self.patch_indices)} indices for patch.")
        print(f"Cached padded data shape: {self.padded_data.shape} (padding happens once at init)")

        # Pad labels for dense segmentation
        self.padded_labels = np.full(
            (self.raw_labels.shape[0] + 2 * self.margin,
             self.raw_labels.shape[1] + 2 * self.margin),
            fill_value=255, dtype=np.int32
        )
        # Convert to 0-based class indices; background (label==0) -> ignore (255)
        label_region = self.raw_labels.copy().astype(np.int32)
        label_region[label_region > 0] -= 1
        label_region[self.raw_labels == 0] = 255
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

    def _sampler_balance(self, indices: np.ndarray) -> WeightedRandomSampler:
        """Build a weighted sampler with class balance + PG hard-example boosts.

        Base weights are inverse class frequency. Optional boosts:
        - PG center boost: increase PG patch sampling frequency.
        - PG-TG boundary boost: focus on boundary-confusion hard samples.
        """
        labels = self.patch_labels[indices]
        class_counts = np.bincount(labels, minlength=self.num).astype(np.float64)
        class_counts = np.maximum(class_counts, 1)
        sample_weight = 1.0 / class_counts
        weights = sample_weight[labels].astype(np.float64)

        # Optional PG-focused weighting knobs from config
        pg_center_boost = float(getattr(self.config.split, 'pg_center_boost', 1.0))
        pg_tg_boundary_boost = float(getattr(self.config.split, 'pg_tg_boundary_boost', 1.0))

        if pg_center_boost > 1.0:
            targets = list(getattr(self.config.clsf, 'targets', []))
            pg_idx = targets.index('PG') if 'PG' in targets else 0
            weights[labels == pg_idx] *= pg_center_boost

        if pg_tg_boundary_boost > 1.0 and self.patch_pg_tg_boundary is not None:
            boundary_flags = self.patch_pg_tg_boundary[indices]
            weights[boundary_flags] *= pg_tg_boundary_boost

        return WeightedRandomSampler(
            weights=torch.from_numpy(weights).double(),
            num_samples=len(indices),
            replacement=True,
        )

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

        self.config = config
        self.data_path = config.path.data
        self.label_path = config.path.label
        self.num = config.clsf.num
        self.targets = config.clsf.targets
        self.patch_size = config.split.patch_size
        self.margin = (config.split.patch_size - 1) // 2
        self.transform = transform
        self.kwargs = kwargs
        self.test_rate = config.split.test_rate
        
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

        # per-patient pipeline (_load_data + _preprocess_data + _create_patches)
        self._per_patient_pipeline()
    
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
    
    def _per_patient_pipeline(self) -> None:
        """
        load, normalize, pca, and extract patches per-patient independently.

        pipeline:
          - load all patients, collect real (non-bg) pixels for global stats
          - fit global normalization + pca on non-bg pixels only
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

        # raw:    0=bg, 1=PG, 2=FAT, 3=TG, 4=LN, 5=MS, 6=Blood, 7=Tra, 8=ES
        # mapped: identity — all 8 tissue classes kept separate
        # LN and Blood are spectrally distinct; merging them caused AUC<0.5
        _label_remap = np.array([
            0,  # 0: background -> background
            1,  # 1: PG         -> PG    (1)
            0,  # 2: FAT        -> FAT   (2)
            2,  # 3: TG         -> TG    (3)
            0,  # 4: LN         -> LN    (4)
            0,  # 5: MS         -> MS    (5)
            0,  # 6: Blood      -> Blood (6)
            3,  # 7: Tra        -> Tra   (7)
            0,  # 8: ES         -> ES    (8)
        ], dtype=np.int32)
        self._label_remap = _label_remap

        tprint("loading patients...")
        patient_records = []
        all_real_pixels = []
        _patient_id_map = {}

        for idx, (data_file, label_file) in enumerate(pairs):
            data = np.load(data_file).astype(np.float32)    # shape: (h_i, w_i, c)
            labels_raw = np.load(label_file).astype(np.int32)  # shape: (h_i, w_i), 1-based

            if data.shape[:2] != labels_raw.shape[:2]:
                raise ValueError(
                    f"shape mismatch: {data_file} {data.shape[:2]} "
                    f"vs {labels_raw.shape[:2]}")

            if not np.isfinite(data).all():
                raise ValueError(f"non-finite values in {data_file}")

            # apply label remap — validate range then clip
            # raw labels are 1-based: 0=bg, 1-8=classes
            # clip to _label_remap's valid index range (0 to len-1), NOT to config.clsf.num
            min_raw = int(labels_raw.min())
            max_raw = int(labels_raw.max())
            if min_raw < 0:
                print(f"\t\tWARNING: {os.path.basename(label_file)} has negative "
                       f"label {min_raw}, will be clipped to 0")
            if max_raw >= len(_label_remap):
                print(f"\t\tWARNING: {os.path.basename(label_file)} has label value "
                       f"{max_raw} exceeding remap table max "
                       f"{len(_label_remap)-1}, will be clipped")
            # CRITICAL: clip to len(_label_remap)-1 (=8), NOT config.clsf.num (=7).
            # clipping to 7 would map ES(8) -> _label_remap[7]=Tra, silently corrupting ES labels.
            labels_clipped = np.clip(labels_raw, 0, len(_label_remap) - 1)
            labels = _label_remap[labels_clipped]  # shape: (h_i, w_i), now 1-based merged

            patient_id = self._extract_patient_id(data_file)
            if patient_id not in _patient_id_map:
                _patient_id_map[patient_id] = len(_patient_id_map)
            pid_idx = _patient_id_map[patient_id]

            patient_records.append((data, labels, pid_idx))

            # collect non-bg pixels for global stats (labels > 0)
            mask = labels.reshape(-1) > 0
            real_px = data.reshape(-1, data.shape[2])[mask]
            all_real_pixels.append(real_px)

            tprint(f"  loaded {idx+1}/{len(pairs)}: "
                   f"{os.path.basename(data_file)} ({patient_id}), "
                   f"shape {data.shape}, non-bg pixels: {mask.sum():,}, "
                   f"possesses non_bg labels: {np.unique(labels)}, "
                   f"max label {labels.max()}, min label {labels.min()}"
                   )

        self._patient_names = {v: k for k, v in _patient_id_map.items()}
        self._num_patients = len(_patient_id_map)
        tprint(f"  {len(pairs)} pairs, {self._num_patients} unique patients")

        # norm + pca on non-bg pixels (via sklearn-compatible HSPreprocessor)
        # excludes background and zero-padding -> cleaner statistics
        tprint("fitting norm + pca...")
        all_real = np.concatenate(all_real_pixels, axis=0)   # shape: (n_total_real, c)
        del all_real_pixels
        c = all_real.shape[1]

        n_components = self.config.preprocess.pca_components
        self.preprocessor = HSPreprocessor(
            pca_components=n_components,
            max_fit_samples=2_000_000,
            max_pca_samples=500_000,
            random_state=350234,
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

        for data, labels, pid_idx in patient_records:
            h, w, c_raw = data.shape                     # per-patient: (h_i, w_i, c)

            # apply fitted preprocessor (Z-score + PCA) via sklearn transform API
            processed = self.preprocessor.transform(data) # shape: (h_i, w_i, c_out)

            padded_data = self._pad_with_zeros(processed, self.margin)
            # padded shape: (h_i + 2*margin, w_i + 2*margin, c_out)

            # pad labels: bg / padding -> 255 (ignore_index)
            # lbl shape: (h_i + 2*margin, w_i + 2*margin)
            padded_lbl = np.full(
                (h + 2 * self.margin, w + 2 * self.margin),
                fill_value=255, dtype=np.int32
                )       
            lbl_region = labels.copy().astype(np.int32)
            lbl_region[lbl_region > 0] -= 1    # 1-based -> 0-based class index
            lbl_region[labels == 0] = 255      # background -> ignore
            padded_lbl[self.margin:self.margin + h,
                       self.margin:self.margin + w] = lbl_region

            patient_list_idx = len(self._patient_padded_data)
            self._patient_padded_data.append(padded_data)
            self._patient_padded_labels.append(padded_lbl)

            # pre-compute PG<->TG boundary map for center-pixel hard-example mining
            # labels are 1-based here: 1=PG, 2=TG, 3=Tra, 0=bg
            pg_mask = (labels == 1)
            tg_mask = (labels == 2)
            boundary_map = np.zeros_like(labels, dtype=bool)
            for dr, dc in [(-1, -1), (-1, 0), (-1, 1),
                           (0, -1),           (0, 1),
                           (1, -1),  (1, 0),  (1, 1)]:
                pg_nb = np.roll(pg_mask, shift=(dr, dc), axis=(0, 1))
                tg_nb = np.roll(tg_mask, shift=(dr, dc), axis=(0, 1))
                boundary_map |= (pg_mask & tg_nb) | (tg_mask & pg_nb)

            # remove wrapped edges introduced by np.roll
            boundary_map[0, :] = False
            boundary_map[-1, :] = False
            boundary_map[:, 0] = False
            boundary_map[:, -1] = False

            # extract patch centers: only non-background pixels
            rows, cols = np.where(labels > 0)
            n_patches = len(rows)

            if self.max_patches_per_patient is not None and n_patches > self.max_patches_per_patient:
                sel = np.random.RandomState(123 + idx).choice(n_patches, self.max_patches_per_patient, replace=False)
                rows = rows[sel]
                cols = cols[sel]
                n_patches = len(rows)
                if self.debug_mode:
                    tprint(f"    capped patches for patient {pid_idx} to {n_patches}")

            # patch_indices: (patient_list_idx, row_in_padded, col_in_padded)
            indices = np.stack([
                np.full(n_patches, patient_list_idx, dtype=np.int32),
                (rows + self.margin).astype(np.int32),
                (cols + self.margin).astype(np.int32),
            ], axis=1)   # shape: (n_patches, 3)

            patch_labels = (labels[rows, cols] - 1).astype(np.int32)   # shape: (n_patches,)
            patient_groups = np.full(n_patches, pid_idx, dtype=np.int32)
            boundary_flags = boundary_map[rows, cols].astype(np.bool_)

            all_patch_indices.append(indices)
            all_patch_labels.append(patch_labels)
            all_patient_groups.append(patient_groups)
            all_pg_tg_boundary.append(boundary_flags)
        # del inside the loop only removed loop-variable
            del data, labels, processed, padded_data, padded_lbl, boundary_map
        # patient_records still held references until here
        del patient_records, _patient_id_map

        # concatenate
        self.patch_indices = np.concatenate(all_patch_indices, axis=0)          # (N, 3)
        self.patch_labels = np.concatenate(all_patch_labels, axis=0)            # (N,)
        self.patch_patient_groups = np.concatenate(all_patient_groups, axis=0)  # (N,)
        self.patch_pg_tg_boundary = np.concatenate(all_pg_tg_boundary, axis=0)  # (N,), bool

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

        # Filter for .npy files and pair them
        data_files = [f for f in all_files if f.endswith('.npy') and not f.endswith('_gt.npy')]
        label_files = [f for f in all_files if f.endswith('_gt.npy')]
        
        pairs = []
        for data_file in data_files:
            base_name = data_file[:-4]  # Remove .npy extension
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
        Create PyTorch DataLoaders with optimized performance settings.
        
        Uses StratifiedGroupKFold (single fold) for train/val+test split to preserve
        class distribution across splits while preventing patient-level leakage.
        Train set uses balanced sampling; val set for early stopping; test set held out.
        
        Args:
            num_workers: Number of worker threads for parallel data loading.
            batch_size: Batch size for each iteration. If None, uses config value or 32.
            pin_memory: Whether to pin memory for GPU transfer.
            prefetch_factor: preload batches per worker.
            persistent_workers: keep workers from frequent recreation.
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """

        from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
        
        total_indices = np.arange(len(self.patch_indices))
        all_classes = set(np.unique(self.patch_labels))
        
        # stratified patient-level split into train+val vs test
        # StratifiedGroupKFold preserves class proportions while ensuring patient isolation.
        # use 5-fold split: take 1 fold as test (20%), 4 folds as train+val pool.
        split_seed = int(getattr(self.config.split, 'split_seed', self.split_seed))
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=split_seed)
        trainval_idx, test_idx = next(sgkf.split(
            total_indices, self.patch_labels,
            groups=self.patch_patient_groups
        ))
        
        # split train+val pool into train vs val (patient-level)
        # GroupShuffleSplit on the trainval pool: 10% val.
        val_rate = self.config.split.val_rate
        gss = GroupShuffleSplit(n_splits=1, test_size=val_rate, random_state=split_seed)
        train_rel, val_rel = next(gss.split(
            np.arange(len(trainval_idx)),
            self.patch_labels[trainval_idx],
            groups=self.patch_patient_groups[trainval_idx]
        ))
        train_idx = trainval_idx[train_rel]
        val_idx = trainval_idx[val_rel]
        
        # verify no patient leakage
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
            tprint(f"  {name} class dist: {dict(zip(self.config.clsf.targets, pct))}")
        
        # create indexed subsets
        # augmentation for train only, no resampling to preserve original distribution
        train_subset = _AugmentedSubset(
            self, train_idx,
            noise_std=0.02,
            band_drop_rate=0.05,
            cutout_ratio=0.15
        )
        val_subset = _IndexedSubset(self, val_idx)
        test_subset = _IndexedSubset(self, test_idx)
        print(f"Training augmentations enabled: flip, rotate, spectral noise, band dropout, cutout")

        if batch_size is None:
            batch_size = getattr(self.config.split, 'batch_size',
                         getattr(self, 'batch_size', 32))
        actual_pin_memory = pin_memory and torch.cuda.is_available()
        
        train_loader = torch.utils.data.DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,  # plain shuffle, no resampling — preserves original distribution
            num_workers=num_workers,
            pin_memory=actual_pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers and (num_workers > 0),
            drop_last=True,
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

        split_seed = int(getattr(self.config.split, 'split_seed', self.split_seed))
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
            train_subset = _AugmentedSubset(
                self, train_idx,
                noise_std=0.02,
                band_drop_rate=0.05,
                cutout_ratio=0.15
            )
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

    # public interfaces.
    def get_loaders(self, *args, **kwargs):
        return self._create_data_loader_(*args, **kwargs)

    def get_cv_loaders(self, *args, **kwargs):
        return self._create_cv_data_loaders_(*args, **kwargs)

    # convenience helpers
    def describe(self, top_k: int = 5) -> None:
        """Return and print basic dataset statistics for quick sanity check."""
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

    def sample_batch(self, batch_size: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a tiny batch (no augmentation) for debugging shapes."""
        idx = np.random.choice(len(self.patch_indices), size=min(batch_size, len(self.patch_indices)), replace=False)
        subset = _IndexedSubset(self, idx)
        loader = torch.utils.data.DataLoader(subset, batch_size=len(idx))
        return next(iter(loader))

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
        self.batch_size = config.split.batch_size
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


class _AugmentedSubset(Dataset):
    """
    Training-only subset with on-the-fly spatial & spectral augmentations.
    
    Augmentations applied (all random, each with 50% probability):
      - Spatial: horizontal flip, vertical flip, 90/180/270 deg. rotation
      - Spectral: Gaussian noise, band-wise dropout (zero out random bands)
      - Mixup-style: CutOut (mask random spatial region)
    
    Returns:
    Augmented patch and label shapes of (C, H, W) and (H, W).
    """
    def __init__(self, dataset: AbstractHSDataset, indices: np.ndarray,
                 noise_std: float = 0.02,
                 band_drop_rate: float = 0.05,
                 cutout_ratio: float = 0.15,
                 minority_threshold: float = 0.3,
                 minority_blend_prob: float = 0.5):
        """Augment subset with optional rare-class copy-paste blending.

        Args:
            minority_threshold: classes with < threshold * max_count as rare.
            minority_blend_prob: prob to apply rare-class blend on rare-class samples.
        """
        self.dataset = dataset
        self.indices = indices
        self.noise_std = noise_std
        self.band_drop_rate = band_drop_rate
        self.cutout_ratio = cutout_ratio
        self.minority_threshold = minority_threshold
        self.minority_blend_prob = minority_blend_prob

        # pre-compute rare classes over subset
        labels = dataset.patch_labels[indices]
        class_counts = np.bincount(labels, minlength=dataset.num)
        self.class_counts = class_counts
        self.max_count = max(1, class_counts.max())
        rare_mask = class_counts < (self.max_count * minority_threshold)
        self.minority_classes = set(np.where(rare_mask)[0].tolist())

        # map class
        # make indices for fast sampling
        self.class_to_indices: Dict[int, np.ndarray] = {}
        for cls_idx in np.where(class_counts > 0)[0]:
            self.class_to_indices[int(cls_idx)] = indices[labels == cls_idx]
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        original_idx = self.indices[idx]
        
        # Get raw numpy patch (H, W, C) and label (H, W)
        patch = self.dataset._get_patch_(original_idx)      # (H, W, C)
        label = self.dataset._get_label_patch_(original_idx) # (H, W)
        
        # horizontal flip
        if np.random.rand() > 0.5:
            patch = np.flip(patch, axis=1).copy()
            label = np.flip(label, axis=1).copy()
        
        # vertical flip
        if np.random.rand() > 0.5:
            patch = np.flip(patch, axis=0).copy()
            label = np.flip(label, axis=0).copy()
        
        # random 90 rotation
        if np.random.rand() > 0.5:
            k = np.random.choice([1, 2, 3])
            patch = np.rot90(patch, k, axes=(0, 1)).copy()
            label = np.rot90(label, k, axes=(0, 1)).copy()
        
        # gaussian noise
        if self.noise_std > 0 and np.random.rand() > 0.5:
            noise = np.random.normal(0, self.noise_std, patch.shape).astype(patch.dtype)
            patch = patch + noise
        
        # random band dropout (zero out a few spectral bands)
        if self.band_drop_rate > 0 and np.random.rand() > 0.5:
            n_bands = patch.shape[2]
            n_drop = max(1, int(n_bands * self.band_drop_rate))
            drop_idx = np.random.choice(n_bands, n_drop, replace=False)
            patch[:, :, drop_idx] = 0.0
        
        # spatial cutout — reduced from 70% to 30% to preserve more supervision signal
        if self.cutout_ratio > 0 and np.random.rand() > 0.7:
            h, w = patch.shape[:2]
            ch = max(1, int(h * self.cutout_ratio))
            cw = max(1, int(w * self.cutout_ratio))
            y0 = np.random.randint(0, h - ch + 1)
            x0 = np.random.randint(0, w - cw + 1)
            patch[y0:y0+ch, x0:x0+cw, :] = 0.0
            # Mark cutout region as ignore in labels
            label[y0:y0+ch, x0:x0+cw] = 255

        # rare-class copy-paste: enrich minority classes with a same-class donor patch
        center_cls = int(self.dataset.patch_labels[original_idx])
        if (
            self.minority_classes
            and center_cls in self.minority_classes
            and np.random.rand() < self.minority_blend_prob
        ):
            donor_pool = self.class_to_indices.get(center_cls)
            if donor_pool is not None and len(donor_pool) > 1:
                donor_idx = int(np.random.choice(donor_pool))
                if donor_idx != original_idx:
                    donor_patch = self.dataset._get_patch_(donor_idx).copy()
                    donor_label = self.dataset._get_label_patch_(donor_idx).copy()

                    h, w = patch.shape[:2]
                    # Paste a moderate region (20%~50% side length) to avoid overpowering
                    rh = np.random.randint(max(1, int(0.2 * h)), max(2, int(0.5 * h)))
                    rw = np.random.randint(max(1, int(0.2 * w)), max(2, int(0.5 * w)))
                    y0 = np.random.randint(0, h - rh + 1)
                    x0 = np.random.randint(0, w - rw + 1)

                    patch[y0:y0+rh, x0:x0+rw, :] = donor_patch[y0:y0+rh, x0:x0+rw, :]
                    label[y0:y0+rh, x0:x0+rw] = donor_label[y0:y0+rh, x0:x0+rw]
        
        # convert to (C, H, W) tensor
        patch = np.transpose(patch, (2, 0, 1))
        
        if self.dataset.transform:
            patch = self.dataset.transform(patch)
        
        return torch.FloatTensor(patch), torch.LongTensor(label)
