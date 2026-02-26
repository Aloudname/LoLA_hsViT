# dataset.py - Hyperspectral Dataset Handling.
# todo:
# - Implement class distribution analysis and visualization methods.
# - Try various seeds for a balanced label distribution in line 542 create_dataloader.
import os, re, torch, numpy as np, warnings

from munch import Munch
from pipeline.monitor import tprint
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional, List
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset, WeightedRandomSampler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")


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

    def _get_patch(self, idx: int) -> np.ndarray:
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

    def _get_label_patch(self, idx: int) -> np.ndarray:
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
        """Build a weighted sampler using sqrt-inverse class frequency.

        Sqrt-inverse provides moderate oversampling for rare classes without
        extreme repetition that pure inverse-frequency would cause.
        """
        labels = self.patch_labels[indices]
        class_counts = np.bincount(labels, minlength=self.num).astype(np.float64)
        class_counts = np.maximum(class_counts, 1)
        sample_weight = 1.0 / np.sqrt(class_counts)
        weights = sample_weight[labels]
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
        patch = self._get_patch(idx)
        
        # Convert to (C, H, W) format for PyTorch
        patch = np.transpose(patch, (2, 0, 1))
        
        if self.transform:
            patch = self.transform(patch)
        
        # Always return dense per-pixel labels
        label_patch = self._get_label_patch(idx)  # [H, W]
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
                  **kwargs: Any) -> None:
        """
        per-patient pipeline: loads each patient independently to avoid
        cross-patient patch contamination and zero-padding noise in pca.

        bypasses AbstractHSDataset.__init__ which concatenates all patients
        into one image, causing boundary artifacts and polluted statistics.
        """
        # initialize torch Dataset directly (skip abstract concat pipeline)
        Dataset.__init__(self)

        # replicate base attributes from AbstractHSDataset.__init__
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

        # per-patient pipeline (replaces _load_data + _preprocess_data + _create_patches)
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
        
        from sklearn.decomposition import PCA
        pairs = self._pair_data_and_labels()
        if not pairs:
            raise RuntimeError("no data/label pairs found in " + self.data_path)

        # raw:    0=bg, 1=PG, 2=FAT, 3=TG, 4=LN, 5=MS, 6=Blood, 7=Tra, 8=ES
        # mapped: identity — all 8 tissue classes kept separate
        # LN and Blood are spectrally distinct; merging them caused AUC<0.5
        _label_remap = np.array([
            0,  # 0: background -> background
            1,  # 1: PG         -> PG    (1)
            2,  # 2: FAT        -> FAT   (2)
            3,  # 3: TG         -> TG    (3)
            4,  # 4: LN         -> LN    (4)
            5,  # 5: MS         -> MS    (5)
            6,  # 6: Blood      -> Blood (6)
            7,  # 7: Tra        -> Tra   (7)
            8,  # 8: ES         -> ES    (8)
        ], dtype=np.int32)

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

            # collect non-bg pixels for global stats (unchanged: labels > 0)
            mask = labels.reshape(-1) > 0
            real_px = data.reshape(-1, data.shape[2])[mask]
            all_real_pixels.append(real_px)

            tprint(f"  loaded {idx+1}/{len(pairs)}: "
                   f"{os.path.basename(data_file)} ({patient_id}), "
                   f"shape {data.shape}, non-bg pixels: {mask.sum():,}, "
                   f"max label {labels.max()}, min label {labels.min()}"
                   )

        self._patient_names = {v: k for k, v in _patient_id_map.items()}
        self._num_patients = len(_patient_id_map)
        tprint(f"  {len(pairs)} pairs, {self._num_patients} unique patients")

        # norm + pca on non-bg pixels
        # excludes background and zero-padding -> cleaner statistics
        tprint("fitting norm + pca...")
        all_real = np.concatenate(all_real_pixels, axis=0)   # shape: (n_total_real, c)
        del all_real_pixels
        c = all_real.shape[1]

        # subsample for normalization stats — use up to 2M pixels for stable mean/std
        # (larger sample = more accurate global stats; memory cost ~2M * c * 4 bytes)
        max_fit = 2_000_000
        if len(all_real) > max_fit:
            rng_fit = np.random.RandomState(350235)
            fit_idx = rng_fit.choice(len(all_real), max_fit, replace=False)
            fit_data = all_real[fit_idx]   # shape: (max_fit, c) — independent copy
            del all_real                   # free large array immediately
        else:
            fit_data = all_real
            all_real = None               # release reference; fit_data keeps the data

        # global normalization stats — fit on non-bg pixels only
        global_mean = fit_data.mean(axis=0).astype(np.float32)       # shape: (c,)
        global_std  = fit_data.std(axis=0).astype(np.float32) + 1e-8 # shape: (c,)
        tprint(f"  normalization fit on {len(fit_data):,} non-bg pixels")

        # pca — fit on normalized non-bg pixels, capped at 500K
        # O(n * p^2); 500K is sufficient for stable eigenvectors)
        n_components = self.config.preprocess.pca_components
        pca_obj = None
        
        # global mean: shape (c,), global std: shape (c,),
        # Z-score normalization on fit_data before PCA fitting,
        # to remove possible sensor bias and ensure PCA captures meaningful variance directions.
        if n_components and n_components < c:
            normalized_fit = (fit_data - global_mean) / global_std  # shape: (n_fit, c)
            max_pca = 500_000
            if len(normalized_fit) > max_pca:
                rng_pca = np.random.RandomState(42)
                pca_idx = rng_pca.choice(len(normalized_fit), max_pca, replace=False)
                pca_fit = normalized_fit[pca_idx]   # shape: (max_pca, c)
            else:
                pca_fit = normalized_fit            # already within budget

            pca_obj = PCA(n_components=n_components, svd_solver='randomized',
                          random_state=350234)
            pca_obj.fit(pca_fit)
            explained = sum(pca_obj.explained_variance_ratio_) * 100
            tprint(f"  pca: {c} -> {n_components} channels, "
                   f"explained variance: {explained:.1f}%, "
                   f"fit on {len(pca_fit):,} pixels")
            del normalized_fit, pca_fit

        del fit_data   # all_real already freed above

        # pre-compute pca projection matrix in float32
        if pca_obj is not None:
            pca_components = pca_obj.components_.astype(np.float32)  # shape: (n_components, c)
            pca_mean = pca_obj.mean_.astype(np.float32)              # shape: (c,)
            c_out = n_components
        else:
            pca_components = None
            pca_mean = None
            c_out = c

        # patient norm -> pca -> pad -> extract patches
        # patient padded independently, no cross-patient boundary bleeding
        tprint("patient patching...")
        self._patient_padded_data   = []  # list of padded arrays per patient
        self._patient_padded_labels = []  # list of padded label maps per patient

        all_patch_indices   = []  # each row: (patient_list_idx, row_padded, col_padded)
        all_patch_labels    = []  # 0-based center-pixel label
        all_patient_groups  = []  # patient group id for each patch

        for data, labels, pid_idx in patient_records:
            h, w, c_raw = data.shape                     # per-patient: (h_i, w_i, c)
            flat = data.reshape(-1, c_raw)               # shape: (h_i * w_i, c)

            # normalize using global stats (fit on non-bg pixels only)
            flat = (flat - global_mean) / global_std

            # pca projection
            # pca_mean is the mean of normalized training pixels (≈0 but not exactly)
            # must subtract before projection to match sklearn's PCA.transform() behavior.
            if pca_components is not None:
                flat = (flat - pca_mean) @ pca_components.T  # shape: (h_i*w_i, n_components)

            processed = flat.reshape(h, w, c_out)        # shape: (h_i, w_i, c_out)

            # pad this patient independently — no cross-patient bleeding
            padded_data = self._pad_with_zeros(processed, self.margin)
            # padded shape: (h_i + 2*margin, w_i + 2*margin, c_out)

            # pad labels: background / padding -> 255 (ignore_index)
            padded_lbl = np.full(
                (h + 2 * self.margin, w + 2 * self.margin),
                fill_value=255, dtype=np.int32
            )  # shape: (h_i + 2*margin, w_i + 2*margin)
            lbl_region = labels.copy().astype(np.int32)
            lbl_region[lbl_region > 0] -= 1    # 1-based -> 0-based class index
            lbl_region[labels == 0] = 255      # background -> ignore
            padded_lbl[self.margin:self.margin + h,
                       self.margin:self.margin + w] = lbl_region

            patient_list_idx = len(self._patient_padded_data)
            self._patient_padded_data.append(padded_data)
            self._patient_padded_labels.append(padded_lbl)

            # extract patch centers: only non-background pixels
            rows, cols = np.where(labels > 0)
            n_patches = len(rows)

            # patch_indices: (patient_list_idx, row_in_padded, col_in_padded)
            indices = np.stack([
                np.full(n_patches, patient_list_idx, dtype=np.int32),
                (rows + self.margin).astype(np.int32),
                (cols + self.margin).astype(np.int32),
            ], axis=1)   # shape: (n_patches, 3)

            patch_labels = (labels[rows, cols] - 1).astype(np.int32)   # shape: (n_patches,)
            patient_groups = np.full(n_patches, pid_idx, dtype=np.int32)

            all_patch_indices.append(indices)
            all_patch_labels.append(patch_labels)
            all_patient_groups.append(patient_groups)

            del data, labels, flat, processed

        # free all original patient arrays — del inside the loop only removed loop-variable
        # names; patient_records still held references until here
        del patient_records

        # --- phase 4: concatenate ---
        self.patch_indices = np.concatenate(all_patch_indices, axis=0)          # shape: (N, 3)
        self.patch_labels = np.concatenate(all_patch_labels, axis=0)            # shape: (N,)
        self.patch_patient_groups = np.concatenate(all_patient_groups, axis=0)  # shape: (N,)

        unique_groups, group_counts = np.unique(
            self.patch_patient_groups, return_counts=True)
        tprint(f"  total patches: {len(self.patch_indices):,} from "
               f"{len(unique_groups)} patients")
        print(f"  patches per patient: min={group_counts.min()}, "
              f"max={group_counts.max()}, mean={group_counts.mean():.0f}")
        print(f"  patch shape: ({self.patch_size}, {self.patch_size}, {c_out}), "
              f"dtype: float32")

    def _get_patch(self, idx: int) -> np.ndarray:
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

    def _get_label_patch(self, idx: int) -> np.ndarray:
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

    def _pair_data_and_labels(self) -> List[Tuple[str, str]]:
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
    
    def create_data_loader(self, num_workers=4, batch_size=None, pin_memory=True, 
                           prefetch_factor=2, persistent_workers=False):
        """
        Create PyTorch DataLoaders with optimized performance settings.
        
        Args:
            num_workers: Number of worker threads for parallel data loading.
                        Recommended: 4 for optimal throughput (prevents I/O bottleneck).
                        Set to 0 only to debug or on single-core systems.
            batch_size: Batch size for each iteration. If None, uses config value or 32.
            pin_memory: Whether to pin memory for GPU transfer, better True on GPU.
            prefetch_factor: preload batches per worker.
            persistent_workers: keep workers from frequent recreation.
        
        Returns:
            Tuple of (train_loader, test_loader)
        """

        from sklearn.model_selection import GroupShuffleSplit
        
        total_indices = np.arange(len(self.patch_indices))
        all_classes = set(np.unique(self.patch_labels))
        
        # Patient-level split to prevent data leakage.
        # NOTE: Retry with different seeds to find a split where ALL classes
        # appear in both train and test sets (rare class 3, 7 may only exist in
        # a few patients (need check), so some seeds will strand them in one set).
        best_split = None
        best_missing = len(all_classes) * 2 + 1  # guarantee any real result beats this
        base_seed = 350234
        
        for seed in range(base_seed, base_seed + 100):
            gss = GroupShuffleSplit(n_splits=1, test_size=self.test_rate,
                                   random_state=seed)
            t_idx, v_idx = next(gss.split(
                total_indices, self.patch_labels,
                groups=self.patch_patient_groups
            ))
            
            test_cls = set(np.unique(self.patch_labels[v_idx]))
            train_cls = set(np.unique(self.patch_labels[t_idx]))
            n_missing = len(all_classes - test_cls) + len(all_classes - train_cls)
            
            if n_missing == 0:
                train_idx, test_idx = t_idx, v_idx
                tprint(f"  GroupShuffleSplit: found full-coverage split (seed={seed})")
                break
            if n_missing < best_missing:
                best_missing = n_missing
                best_split = (t_idx, v_idx, seed)
        else:
            # No perfect split found — use the best available one
            if best_split is None:
                # every seed had the same n_missing; just use the last one
                train_idx, test_idx = t_idx, v_idx
                tprint("  WARNING: No split improved over baseline; using last seed.")
            else:
                train_idx, test_idx, seed = best_split
                tprint(f"  WARNING: No split covers all classes in both sets after 100 seeds. "
                       f"Using best seed={seed} ({best_missing} missing class(es)).")
        
        # Verify no patient-level leakage
        train_patients = set(self.patch_patient_groups[train_idx])
        test_patients = set(self.patch_patient_groups[test_idx])
        assert len(train_patients & test_patients) == 0, \
            "Data leakage detected: patients appear in both train and test sets!"
        tprint(f"Patient-level split: {len(train_patients)} train patients, "
              f"{len(test_patients)} test patients (0 overlap)")
        
        # Check class coverage in both splits
        train_classes = set(np.unique(self.patch_labels[train_idx]))
        test_classes = set(np.unique(self.patch_labels[test_idx]))
        if train_classes != all_classes:
            missing = all_classes - train_classes
            print(f"  WARNING: Training set missing classes: {missing}")
        if test_classes != all_classes:
            missing = all_classes - test_classes
            print(f"  WARNING: Test set missing classes: {missing}")
        
        # create indexed subsets:
        # Training: augmented subset with spatial/spectral augmentations
        # Testing: plain subset without augmentations
        train_subset = _AugmentedSubset(
            self, train_idx,
            noise_std=0.02,
            band_drop_rate=0.05,
            cutout_ratio=0.15
        )
        test_subset = _IndexedSubset(self, test_idx)
        print(f"Training augmentations enabled: flip, rotate, spectral noise, band dropout, cutout")

        # memory： prefer config value, then explicit attr, then safe default
        if batch_size is None:
            batch_size = getattr(self.config.split, 'batch_size',
                         getattr(self, 'batch_size', 32))
        
        actual_pin_memory = pin_memory and torch.cuda.is_available()
        
        train_loader = torch.utils.data.DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=actual_pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers and (num_workers > 0),
            drop_last=True,  # drop last batch if it's smaller than batch_size
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
        print(f"Training set: {len(train_loader)} batches ({len(train_idx)} samples)")
        print(f"Test set: {len(test_loader)} batches ({len(test_idx)} samples)")
        print(f"DataLoader config: num_workers={num_workers}, batch_size={batch_size}, pin_memory={actual_pin_memory}")
        print(f"  prefetch_factor={prefetch_factor}, persistent_workers={persistent_workers and (num_workers > 0)}")
        
        return train_loader, test_loader

    def create_cv_data_loaders(self, n_folds=5, num_workers=4, batch_size=None,
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

        sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True,
                                    random_state=350234)

        if batch_size is None:
            batch_size = self.batch_size if hasattr(self, 'batch_size') else 32
        actual_pin_memory = pin_memory and torch.cuda.is_available()

        fold_loaders = []

        tprint(f"\n[Cross-Validation Split] {n_folds} folds, "
              f"patient-level StratifiedGroupKFold")

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

            # Balanced sampler per fold
            train_sampler = self._sampler_balance(train_idx)

            train_loader = torch.utils.data.DataLoader(
                train_subset,
                batch_size=batch_size,
                sampler=train_sampler,
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
    
    Labels are augmented consistently for dense (segmentation) tasks.
    """
    def __init__(self, dataset: AbstractHSDataset, indices: np.ndarray,
                 noise_std: float = 0.02,
                 band_drop_rate: float = 0.05,
                 cutout_ratio: float = 0.15):
        self.dataset = dataset
        self.indices = indices
        self.noise_std = noise_std
        self.band_drop_rate = band_drop_rate
        self.cutout_ratio = cutout_ratio
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        original_idx = self.indices[idx]
        
        # Get raw numpy patch (H, W, C) and label (H, W)
        patch = self.dataset._get_patch(original_idx)      # (H, W, C)
        label = self.dataset._get_label_patch(original_idx) # (H, W)
        
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
        
        # convert to (C, H, W) tensor
        patch = np.transpose(patch, (2, 0, 1))
        
        if self.dataset.transform:
            patch = self.dataset.transform(patch)
        
        return torch.FloatTensor(patch), torch.LongTensor(label)
