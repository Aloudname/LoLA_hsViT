# dataset.py - Hyperspectral Dataset Handling.
# todo:
# - Implement class distribution analysis and visualization methods.
import os
import torch
import numpy as np
from munch import Munch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional, List
from sklearn.model_selection import train_test_split


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
        self.pca_component = config.preprocess.pca_components

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
        """
        # Check data and label spatial dimensions match
        if self.raw_data.shape[:2] != self.raw_labels.shape[:2]:
            raise ValueError(
                f"Spatial dimensions mismatch: data {self.raw_data.shape[:2]}, "
                f"labels {self.raw_labels.shape[:2]}"
            )
            
        # Check for non-finite values in data
        if not np.isfinite(self.raw_data).all():
            raise ValueError("Raw data contains non-finite values (NaN/inf)")
            
        # Validate label range
        unique_labels = np.unique(self.raw_labels)
        if not np.all((unique_labels >= 0) & (unique_labels <= self.num)):
            raise ValueError(
                f"Labels contain values outside valid range [0, {self.num}]"
            )

    def _create_patches(self) -> None:
        """
        Extract spatial patches from preprocessed data.
        
        Patches are extracted with zero-padding at image boundaries. Only patches
        with non-zero labels are included (background/zero labels are excluded).
        
        OPTIMIZATION: Padding is done once and cached to avoid re-padding for each sample.
        """
        # Add zero padding around the image (ONCE at initialization)
        self.padded_data = self._pad_with_zeros(self.processed_data, self.margin)
        
        # Collect valid patches
        patch_indices = []
        labels_list = []
        
        # Iterate over all spatial positions in original image
        for r in range(self.margin, self.padded_data.shape[0] - self.margin):
            for c in range(self.margin, self.padded_data.shape[1] - self.margin):
                # Get corresponding position in original (unpadded) image
                orig_r = r - self.margin
                orig_c = c - self.margin
                label = self.raw_labels[orig_r, orig_c]
                
                # Skip background (zero labels)
                if label > 0:
                    patch_indices.append((r, c))
                    labels_list.append(label - 1)   # Convert to 0-based index
        
        # Convert to numpy arrays
        # store indices and labels
        self.patch_indices = np.array(patch_indices, dtype=np.int32)
        self.patch_labels = np.array(labels_list, dtype=np.int32)
        print(f"Created {len(self.patch_indices)} indices for patch.")
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
        print(f"Created padded label map: {self.padded_labels.shape}")

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

    def splitTrainTestDataset(self, X, y, randomState=350234):
        """
            Splitter for test and training dataloader.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=self.test_rate,
                                                    random_state=randomState,
                                                    stratify=y)
        # Validate split results
        assert len(np.unique(y_train)) == len(np.unique(y)), "Training set missing some classes"
        assert len(np.unique(y_test)) == len(np.unique(y)), "Test set missing some classes"
    
        return X_train, X_test, y_train, y_test

    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get basic information about the dataset.
        
        Returns:
            Dictionary containing dataset info such as number of samples,
            number of classes, class names, and patch size.
        """
        return {
            "dataset_name": self.config.common.dataset_name,
            "total_samples": len(self.patch_indices),
            "test_samples": int(len(self.patch_indices) * self.test_rate),
            "num_classes": self.num,
            "class_names": self.targets,
            "patch_size": self.patch_size
        }

    def get_class_distribution(self) -> Dict[int, int]:
        """
        Get distribution of classes in the dataset.
        
        Returns:
            Dictionary mapping class indices to their counts
        """
        unique, counts = np.unique(self.patch_labels, return_counts=True)
        return dict(zip(unique, counts))

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
        Initialize a NumPy format hyperspectral dataset.
        
        See parent class for parameters.
        """
        super().__init__(config, transform, **kwargs)
        self.data_path = config.path.data
    
    
    def _pair_data_and_labels(self) -> List[Tuple[str, str]]:
        """
        Pair data and label files based on naming convention.
        
        Returns:
            List of tuples containing (data_file, label_file) paths
        """
        
        all_files = os.listdir(self.data_path)
        
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
                print(f"Warning: No label file found for {data_file}")
        return pairs
    
    def _load_data(self) -> None:
        """Load data from .npy files with memory-efficient approach"""
        pairs = self._pair_data_and_labels()
        
        # calculate total shape
        total_h, total_w = 0, 0
        for data_file, label_file in pairs:
            data = np.load(data_file, mmap_mode='r')  # read only metadata.
            total_h += data.shape[0]
            total_w = max(total_w, data.shape[1])
            c = data.shape[2]
        
        print(f"Detected concatenation shape: ({total_h}, {total_w}, {c})")
        
        # Second pass: allocate once and fill
        self.raw_data = np.zeros((total_h, total_w, c), dtype=np.float32)
        self.raw_labels = np.zeros((total_h, total_w), dtype=np.int32)
        
        row_offset = 0
        for idx, (data_file, label_file) in enumerate(pairs):
            data = np.load(data_file)
            labels = np.load(label_file)
            
            h, w = data.shape[:2]
            # Validate shapes
            if data.shape[:2] != labels.shape[:2]:
                raise ValueError(f"Shape mismatch between {data_file} and {label_file}")
            
            # Fill allocated arrays
            self.raw_data[row_offset:row_offset+h, :w, :] = data
            self.raw_labels[row_offset:row_offset+h, :w] = labels
            row_offset += h
            
            print(f"  Loaded pair {idx+1}/{len(pairs)}: {data_file.split('/')[-1]}")
            
            # Explicitly delete to free memory
            del data, labels
        
        print(f"  Loaded {len(pairs)} pairs of .npy files")
        print(f"  Concatenated data shape: {self.raw_data.shape}, labels shape: {self.raw_labels.shape}")
        
    def _preprocess_data(self) -> None:
        """
        Preprocess data with normalization and optional PCA,
        Process in chunks (100k pixels per chunk) to reduce memory usage.
        """
        from sklearn.decomposition import PCA, IncrementalPCA
        from sklearn.preprocessing import StandardScaler
        
        h, w, c = self.raw_data.shape
        chunk_size = 100_000
        flat_data_chunks = []
        
        print("Normalizing data in chunks...")
        for i in range(0, h * w, chunk_size):
            end_idx = min(i + chunk_size, h * w)
            chunk = self.raw_data.reshape(-1, c)[i:end_idx]
            
            scaler = StandardScaler()
            scaled_chunk = scaler.fit_transform(chunk)
            flat_data_chunks.append(scaled_chunk)
            
            if (i // chunk_size + 1) % 100 == 0:
                print(f"  Processed {min(end_idx, h*w)}/{h*w} pixels")
        
        flat_data = np.vstack(flat_data_chunks)
        del flat_data_chunks  # release buffer results
        
        # Apply PCA if specified
        if self.pca_component and self.pca_component < c:
            print(f"Applying PCA: Channel {c} -> {self.pca_component} components...")
            # Use IncrementalPCA for large datasets
            if h * w > 10_000_000:
                pca = IncrementalPCA(n_components=self.pca_component, batch_size=100_000)
            else:
                pca = PCA(n_components=self.pca_component, random_state=42)
            
            pca_data = pca.fit_transform(flat_data)
            self.processed_data = pca_data.reshape(h, w, self.pca_component)
            print(f"  Applied PCA: reduced from {c} to {self.pca_component} components")
            print(f"  Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
        else:
            self.processed_data = flat_data.reshape(h, w, c)
            print("  PCA not applied, using all original components")
            
        del flat_data  # release normalized data
            
    def validate_data_quality(self) -> None:
        """Check data quality: distribution, ranges, duplicates"""
        print("\n[Dataset Quality Check]")
        unique_labels, counts = np.unique(self.patch_labels, return_counts=True)
        print(f"  Class distribution: {dict(zip(unique_labels, counts))}")
        
        # Check imbalance
        imbalance = counts.max() / counts.min() if len(counts) > 0 else 1
        if imbalance > 10:
            print(f"  Class imbalance detected (ratio: {imbalance:.1f}x)")
        
        # Check data range
        sample = self._get_patch(0)
        print(f"  Data range: [{sample.min():.4f}, {sample.max():.4f}]")
    
    def create_data_loader(self, num_workers=4, batch_size=None, pin_memory=True, 
                           prefetch_factor=2, persistent_workers=False):
        """
        Create PyTorch DataLoaders with optimized performance settings.
        
        Args:
            num_workers: Number of worker threads for parallel data loading.
                        Recommended: 4 for optimal throughput (prevents I/O bottleneck).
                        Set to 0 only to debug or on single-core systems.
            batch_size: Batch size for each iteration. If None, uses config value or 32.
            pin_memory: Whether to pin memory for GPU transfer (True recommended on GPU systems).
            prefetch_factor: preload batches per worker(default=2).
            persistent_workers: keep workers from frequent recreation (default=False).
        
        Returns:
            Tuple of (train_loader, test_loader)
        """

        total_indices = np.arange(len(self.patch_indices))
        
        train_idx, test_idx, _, _ = train_test_split(
            total_indices,
            self.patch_labels,
            test_size=self.test_rate,
            random_state=350234,
            stratify=self.patch_labels
        )
        
        # create an indexed subset
        train_subset = _IndexedSubset(self, train_idx)
        test_subset = _IndexedSubset(self, test_idx)
        
        # Determine batch size and pin_memory settings
        if batch_size is None:
            batch_size = self.batch_size if hasattr(self, 'batch_size') else 32
        
        actual_pin_memory = pin_memory and torch.cuda.is_available()
        
        train_loader = torch.utils.data.DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=actual_pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else 1,
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
            prefetch_factor=prefetch_factor if num_workers > 0 else 1,
            persistent_workers=persistent_workers and (num_workers > 0),
            drop_last=False,
            timeout=60 if num_workers > 0 else 0
        )
        
        print(f"Training set: {len(train_loader)} batches ({len(train_idx)} samples)")
        print(f"Test set: {len(test_loader)} batches ({len(test_idx)} samples)")
        print(f"DataLoader config: num_workers={num_workers}, batch_size={batch_size}, pin_memory={actual_pin_memory}")
        print(f"  prefetch_factor={prefetch_factor}, persistent_workers={persistent_workers and (num_workers > 0)}")
        
        return train_loader, test_loader

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
                
            print(f"Loaded .mat data: {self.raw_data.shape}, labels: {self.raw_labels.shape}")
            
        except KeyError as e:
            raise ValueError(f"Missing key in .mat file: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading .mat files: {e}")

    def _preprocess_data(self):
        """
        Preprocess data with PCA dimensionality reduction and normalization.
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        # Reshape data for PCA (flatten spatial dimensions)
        h, w, c = self.raw_data.shape
        flat_data = self.raw_data.reshape(-1, c)
        
        # Handle outliers and normalization
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(flat_data)
        
        # Apply PCA
        pca = PCA(n_components=self.pca_components, random_state=42)
        pca_data = pca.fit_transform(scaled_data)
        
        # Reshape back to original spatial dimensions
        self.processed_data = pca_data.reshape(h, w, self.pca_components)
        return self.processed_data, pca

    def validate_data_quality(self) -> None:
        """Check data quality: distribution, ranges, duplicates"""
        print("\n[Dataset Quality Check]")
        unique_labels, counts = np.unique(self.patch_labels, return_counts=True)
        print(f"  Class distribution: {dict(zip(unique_labels, counts))}")
        
        # Check imbalance
        imbalance = counts.max() / counts.min() if len(counts) > 0 else 1
        if imbalance > 10:
            print(f"  Class imbalance detected (ratio: {imbalance:.1f}x)")
        
        # Check data range
        sample = self._get_patch(0)
        print(f"  Data range: [{sample.min():.4f}, {sample.max():.4f}]")
    
    def create_data_loader(self, num_workers=0):
        """
            Create enhanced data loaders with comprehensive processing
        
        Args:
            batch_size (int): Batch size for data loaders
            test_rate (float): Ratio of data to use for testing (0.02 = 2% test, 98% train)
            patch_size (int): Size of image patches
            dataset_name (str): Dataset to use (see loadData for options)
    
        Returns:
            tuple: (train_loader, test_loader, all_loader, y_all, pca_components, dataset_info)
        """
        x_pca = self.processed_data  # (H, W, C)
        y = self.raw_labels  # (H, W)
        
        print('Hyperspectral data shape after PCA: ', x_pca.shape)
        print('Label shape: ', y.shape)
        print('Original label range:', np.min(y), 'to', np.max(y))
        print('Unique labels:', np.unique(y))
        
        x_patches, y_all = self._create_cube(x_pca, y, windowSize=self.patch_size)
        print('Data cube X shape: ', x_patches.shape)
        print('Processed label range:', np.min(y_all), 'to', np.max(y_all))
        print('Unique processed labels:', np.unique(y_all))
        
        Xtrain, Xtest, ytrain, ytest = self.splitTrainTestDataset(x_patches, y_all, randomState=350234)
        print('Xtrain shape: ', Xtrain.shape)
        print('Xtest shape: ', Xtest.shape)
        
        X = x_patches.reshape(-1, self.patch_size, self.patch_size, self.pca_components, 1)
        Xtrain = Xtrain.reshape(-1, self.patch_size, self.patch_size, self.pca_components, 1)
        Xtest = Xtest.reshape(-1, self.patch_size, self.patch_size, self.pca_components, 1)
        
        X = X.transpose(0, 4, 3, 1, 2).squeeze(1)
        Xtrain = Xtrain.transpose(0, 4, 3, 1, 2).squeeze(1)
        Xtest = Xtest.transpose(0, 4, 3, 1, 2).squeeze(1)
        
        # temp container.
        trainset = container(Xtrain, ytrain)
        testset = container(Xtest, ytest)
        
        pin_memory=self.pin_memory

        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
        )
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        )
        return train_loader, test_loader

class TiffHSDataset(AbstractHSDataset):
    """
    Hyperspectral dataset loader for TIFF file format.
    
    Handles datasets stored in multi-band TIFF files, another common format
    for hyperspectral imagery.
    """
    
    def __init__(self, 
                 data_path: str,
                 label_path: str,
                 num_classes: int,
                 target_names: List[str],
                 patch_size: int = 15,
                 transform: Optional[Any] = None,
                 spectral_subset: Optional[List[int]] = None,** kwargs: Any) -> None:
        """
        Initialize a TIFF format hyperspectral dataset.
        
        Args:
            spectral_subset: Optional list of band indices to select
            See parent class for other parameters
        """
        self.spectral_subset = spectral_subset
        super().__init__(data_path, label_path, num_classes, target_names, 
                         patch_size, transform, **kwargs)

    def _load_data(self) -> None:
        """Load data from TIFF files using rasterio"""
        try:
            import rasterio
        except ImportError:
            raise ImportError("rasterio is required for TIFF datasets. Install with: pip install rasterio")
        
        try:
            # Load hyperspectral data (multi-band TIFF)
            with rasterio.open(self.data_path) as src:
                # Read all bands (C, H, W)
                data = src.read()
                # Transpose to (H, W, C) format
                self.raw_data = np.transpose(data, (1, 2, 0))
                
            # Load label data (single-band TIFF)
            with rasterio.open(self.label_path) as src:
                # Read label band and squeeze to (H, W)
                self.raw_labels = src.read(1).squeeze()
                
            print(f"Loaded TIFF data: {self.raw_data.shape}, labels: {self.raw_labels.shape}")
            
        except Exception as e:
            raise RuntimeError(f"Error loading TIFF files: {e}")

    def _preprocess_data(self) -> None:
        """
        Preprocess data with band selection and normalization.
        """
        from sklearn.preprocessing import StandardScaler
        
        # Select specific spectral bands if requested
        if self.spectral_subset is not None:
            self.processed_data = self.raw_data[..., self.spectral_subset]
            print(f"Selected spectral bands: {self.spectral_subset}")
        else:
            self.processed_data = self.raw_data.copy()
        
        # Normalize each band to zero mean and unit variance
        h, w, c = self.processed_data.shape
        flat_data = self.processed_data.reshape(-1, c)
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(flat_data)
        
        self.processed_data = scaled_data.reshape(h, w, c)
        print(f"Preprocessed data shape: {self.processed_data.shape}")

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

class container:
    """
    A simple container for temporary storage with deferred tensor conversion.
    
    Note: 
    magic method ``__len__``, ``__getitem__``
    is needed for such classes.
    """
    def __init__(self, hyperspectral_data, labels):
        self.hyperspectral_data = hyperspectral_data  # Already (N, C, H, W)
        self.labels = labels

    def __len__(self):
        return len(self.hyperspectral_data)
    
    def __getitem__(self, idx):
        # Get data (already in correct shape from create_data_loader)
        hyperspectral = self.hyperspectral_data[idx]  # Shape: (C, H, W) - already transposed
        
        # Convert to tensor directly
        hyperspectral_tensor = torch.FloatTensor(hyperspectral)
        
        # Create pretrained input (RGB-like from 15 bands)
        if hyperspectral.shape[0] == 15:  # (15, H, W)
            # Select 3 representative bands
            r_band = hyperspectral[0]   # First band as red
            g_band = hyperspectral[7]   # Middle band as green  
            b_band = hyperspectral[14]  # Last band as blue
            
            # Stack to create RGB image
            rgb = np.stack([r_band, g_band, b_band], axis=0)  # (3, H, W)
            
            # Normalize to [0, 1] range
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
            
            # Convert to tensor
            pretrained_input = torch.FloatTensor(rgb)
            
            # Resize from 15x15 to 224x224 for pretrained model
            pretrained_input = torch.nn.functional.interpolate(
                pretrained_input.unsqueeze(0),  # Add batch dimension
                size=(224, 224), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)  # Remove batch dimension
        else:
            # Fallback: use first 3 channels
            pretrained_input = torch.FloatTensor(hyperspectral[:3].copy())
            
            # Resize to 224x224
            pretrained_input = torch.nn.functional.interpolate(
                pretrained_input.unsqueeze(0),
                size=(224, 224),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        # Get label
        label = self.labels[idx]
        
        return hyperspectral_tensor, pretrained_input, torch.LongTensor([label]).squeeze()
    
if __name__ == "__main__":
        """
        test dataset functionalities with .mat format data.
        """
        
        from config import load_config
        config = load_config()
        # dataset = MatHSDataset(config=config)
        # train_loader, test_loader = dataset.create_data_loader(num_workers=0)
        
        # # Print dataset info
        # pca_components = config.preprocess.pca_components
        # print(f"Dataset loaded with {len(train_loader.dataset)} training samples and {len(test_loader.dataset)} test samples")
        # print(f"Patch size: {config.split.patch_size}x{config.split.patch_size}")
        # print(f"Number of classes: {dataset.num}")
        # print(f"Class distribution in training set: {dataset.get_class_distribution()}")
        # print(f"Class distribution in test set: {dataset.get_class_distribution()}")
        
        # # Print sample shapes
        # sample_hyperspectral, sample_pretrained_input, sample_label = train_loader.dataset[0]
        # print(f"Sample hyperspectral data shape: {sample_hyperspectral.shape}")
        
        # print(f"Sample pretrained input shape: {sample_pretrained_input.shape}")
        # print(f"Sample label: {sample_label.item()}")
    