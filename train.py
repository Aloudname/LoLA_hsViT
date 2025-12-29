import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler 
from torch.cuda.amp.autocast_mode import autocast
from operator import truediv
import time
from torch.utils.data import Dataset
from typing import Tuple, Dict, Any, Optional, List
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import os
import cv2
import torch.multiprocessing as mp
from torch.nn import functional as F
from sklearn.utils.class_weight import compute_class_weight
import wandb
from repo import *
from LoLA_hsViT import (
    LoLA_hsViT,
    analyze_model_efficiency,
    create_model,
    load_backbone_into_model,
    prepare_model_for_lora_finetuning,
    merge_lora_for_inference,
)
import os
import timm
from torchvision import transforms
import warnings



warnings.filterwarnings("ignore")

# Set environment variable for wandb
os.environ["WANDB_API_KEY"] = "b68375e4a0cbcbc284700f0627c966e5d78181b6"

# limit max threads.
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'

mp.set_start_method('spawn', force=True)

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional, List

class AbstractHyperspectralDataset(ABC, Dataset):
    """
    Abstract Base Class for different types of hyperspectral data.

    NOTE: ``_load_data`` & ``_preprocess_data`` are abstract methods,
         which means re-implementation in inherit classes are necessary.
    """
    
    def __init__(self, 
                 data_path: str,
                 label_path: str,
                 num_classes: int,
                 target_names: List[str],
                 patch_size: int = 15,
                 transform: Optional[Any] = None,
                 test_rate: float = 0.2,
                 pca_component: int = 15,
                 **kwargs: Any) -> None:
        """
        Initialize the hyperspectral dataset.
        
        Args:
            data_path: Path to the main hyperspectral data file
            label_path: Path to the label file
            num_classes: Number of classes in the dataset
            target_names: List of class names
            patch_size: Size of spatial patches to extract (must be odd)
            transform: Optional transform to apply to samples
            test_rate: test rate for split test dataset
            pca_component: how many dims (main features) picked for pca.
           ** kwargs: Additional format-specific parameters
        """
        if patch_size % 2 == 0:
            raise ValueError(f"patch_size must be odd, got {patch_size}")
            
        self.data_path = data_path
        self.label_path = label_path
        self.num_classes = num_classes
        self.target_names = target_names
        self.patch_size = patch_size
        self.margin = (patch_size - 1) // 2  # Calculate padding margin
        self.transform = transform
        self.kwargs = kwargs
        self.test_rate = test_rate
        self.pca_component = pca_component
        
        # Core data structures to be populated by subclasses
        self.raw_data: np.ndarray = None  # Original hyperspectral data (H, W, C)
        self.raw_labels: np.ndarray = None  # Original labels (H, W)
        self.processed_data: np.ndarray = None  # Preprocessed data (e.g., after PCA)
        self.patches: np.ndarray = None  # Extracted image patches
        self.patch_labels: np.ndarray = None  # Labels corresponding to patches
        
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
        if not np.all((unique_labels >= 0) & (unique_labels <= self.num_classes)):
            raise ValueError(
                f"Labels contain values outside valid range [0, {self.num_classes}]"
            )

    def _create_patches(self) -> None:
        """
        Extract spatial patches from preprocessed data.
        
        Patches are extracted with zero-padding at image boundaries. Only patches
        with non-zero labels are included (background/zero labels are excluded).
        """
        # Add zero padding around the image
        padded_data = self._pad_with_zeros(self.processed_data, self.margin)
        
        # Collect valid patches
        patches_list = []
        labels_list = []
        
        # Iterate over all spatial positions in original image
        for r in range(self.margin, padded_data.shape[0] - self.margin):
            for c in range(self.margin, padded_data.shape[1] - self.margin):
                # Get corresponding position in original (unpadded) image
                orig_r = r - self.margin
                orig_c = c - self.margin
                label = self.raw_labels[orig_r, orig_c]
                
                # Skip background (zero labels)
                if label > 0:
                    # Extract patch (patch_size x patch_size x C)
                    patch = padded_data[
                        r - self.margin : r + self.margin + 1,
                        c - self.margin : c + self.margin + 1,
                        :
                    ]
                    patches_list.append(patch)
                    labels_list.append(label - 1)  # Convert to 0-based index
        
        # Convert to numpy arrays
        self.patches = np.array(patches_list, dtype=np.float32)
        self.patch_labels = np.array(labels_list, dtype=np.int32)
        
        # Validate patch creation
        if len(self.patches) == 0:
            raise RuntimeError("No valid patches created - check label data")
        
        print(f"Created {len(self.patches)} patches of size {self.patch_size}x{self.patch_size}")

    def _create_cube(self, x: np.ndarray, y: np.ndarray, windowSize=15, removeZeroLabels=True):
        """Create image cubes for hyperspectral data processing with memory-efficient approach"""
        margin = int((windowSize -1) / 2)
        zeroPaddedX = self._pad_with_zeros(x, margin=margin)

        patches_list, labels_list = [], []

        print(f"Creating patches for {x.shape[0]}x{x.shape[1]} image with {windowSize}x{windowSize} window...")
    
        for r in range(margin, zeroPaddedX.shape[0] - margin):
            for c in range(margin, zeroPaddedX.shape[1] - margin):
                # Get the original position in the unpadded image
                orig_r = r - margin
                orig_c = c - margin
                label = y[orig_r, orig_c]
            
                # Only collect patches with non-zero labels (if removeZeroLabels=True)
                if not removeZeroLabels or label > 0:
                    patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
                    patches_list.append(patch)
                    labels_list.append(label)
    
        # Convert to numpy arrays only after filtering
        patchesData = np.array(patches_list, dtype=np.float32)
        patchesLabels = np.array(labels_list, dtype=np.int32)
    
        if removeZeroLabels:
            # Labels are already filtered, just convert to 0-based indexing
            patchesLabels -= 1
    
        # Validate label range
        min_label = np.min(patchesLabels)
        max_label = np.max(patchesLabels)
        num_classes = len(np.unique(patchesLabels))
        print(f"Created {len(patchesData)} patches")
        print(f"Label range: [{min_label}, {max_label}], Number of classes: {num_classes}")
    
        # Check for non-finite values
        assert np.isfinite(patchesData).all(), "Patch data contains non-finite values"
        return patchesData, patchesLabels
    
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

    def get_class_distribution(self) -> Dict[int, int]:
        """
        Get distribution of classes in the dataset.
        
        Returns:
            Dictionary mapping class indices to their counts
        """
        unique, counts = np.unique(self.patch_labels, return_counts=True)
        return dict(zip(unique, counts))

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get dataset metadata.
        
        Returns:
            Dictionary containing dataset metadata
        """
        return {
            "data_path": self.data_path,
            "label_path": self.label_path,
            "num_classes": self.num_classes,
            "target_names": self.target_names,
            "patch_size": self.patch_size,
            "raw_data_shape": self.raw_data.shape,
            "processed_data_shape": self.processed_data.shape,
            "num_patches": len(self),
            "class_distribution": self.get_class_distribution()
        }

    # Magic methods.
    def __len__(self) -> int:
        """Return number of patches in dataset"""
        return len(self.patches)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Tuple containing:
                - patch: Tensor of shape (C, H, W)
                - label: Tensor containing class index
        """
        patch = self.patches[idx]
        label = self.patch_labels[idx]
        
        # Convert to (C, H, W) format for PyTorch
        patch = np.transpose(patch, (2, 0, 1))
        
        # Apply transforms if specified
        if self.transform:
            patch = self.transform(patch)
        
        return torch.FloatTensor(patch), torch.LongTensor([label]).squeeze()


class MatHyperspectralDataset(AbstractHyperspectralDataset):
    """
    Hyperspectral dataset loader for .mat file format.
    
    Handles datasets stored in MATLAB .mat files, commonly used for
    hyperspectral image datasets.
    """
    
    def __init__(self, 
                 data_path: str,
                 label_path: str,
                 num_classes: int,
                 target_names: List[str],
                 data_key: str,
                 label_key: str,
                 patch_size: int = 15,
                 batch_size: int = 32,
                 transform: Optional[Any] = None,
                 test_rate: float = 0.2,
                 pin_memory: bool = False,
                 pca_components: int = 15,** kwargs: Any) -> None:
        """
        Initialize a MATLAB format hyperspectral dataset.
        
        Args:
            data_key: Key in .mat file for hyperspectral data
            label_key: Key in .mat file for label data
            pca_components: Number of PCA components for dimensionality reduction
            See parent class for other parameters
        """
        self.data_key = data_key
        self.label_key = label_key
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.pca_components = pca_components
        super().__init__(data_path, label_path, num_classes, target_names, 
                         patch_size, transform, test_rate, pca_components, **kwargs)

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
        allset = container(X, y_all)
        
        pin_memory=self.pin_memory

        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
        )
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        )
        all_loader = torch.utils.data.DataLoader(
            allset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        )
        
        dataset_info = {
            'num_classes': self.num_classes,
            'target_names': self.target_names,
            'data_shape': self.raw_data.shape,
            'processed_shape': self.processed_data.shape,
            'patch_size': self.patch_size
        }
        return train_loader, test_loader, all_loader, y_all, self.pca_components, dataset_info

class TiffHyperspectralDataset(AbstractHyperspectralDataset):
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
        print(f"Preprocessed TIFF data: {self.processed_data.shape}")


class container:
    """
        A simple container for temporary storage.
        Before data & labels are split into dataLoaders.

        Note: 
        magic method ``__len__``, ``__getitem__``
        is needed for such classes.
    """
    def __init__(self, hyperspectral_data, labels):
        self.hyperspectral_data = hyperspectral_data
        self.labels = labels

    def __len__(self):
        return len(self.hyperspectral_data)
    
    def __getitem__(self, idx):
        # Get hyperspectral data - ensure it's properly shaped
        hyperspectral = self.hyperspectral_data[idx]  # Shape: (H, W, 15)
        
        # Convert to tensor and transpose to (C, H, W) for PyTorch
        if len(hyperspectral.shape) == 3:
            hyperspectral = np.transpose(hyperspectral, (2, 0, 1))  # (15, H, W)
        
        # Convert hyperspectral data to RGB-like input for pretrained model
        # Select 3 representative bands from the 15 hyperspectral bands
        if hyperspectral.shape[0] == 15:  # (15, H, W)
            # Select bands that roughly correspond to RGB wavelengths
            r_band = hyperspectral[0]  # First band as red
            g_band = hyperspectral[7]  # Middle band as green  
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
            # Fallback: use first 3 channels if not 15
            pretrained_input = torch.FloatTensor(hyperspectral[:3])
            
            # Resize to 224x224
            pretrained_input = torch.nn.functional.interpolate(
                pretrained_input.unsqueeze(0),
                size=(224, 224),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        # Get label
        label = self.labels[idx]
        
        return torch.FloatTensor(hyperspectral), pretrained_input, torch.LongTensor([label]).squeeze()

class HyperspectralTrainer:
    def __init__(self, \
                  config: dict, \
                  dataLoader: MatHyperspectralDataset, \
                  epochs: int = 20):
        self.config = config
        self.epochs = epochs
        self.dataLoader = dataLoader
        self.process_monitor = ProcessMonitor(max_processes=8)
        
        # FORCE CUDA USAGE - bypass torch.cuda.is_available() check
        try:
            # Force CUDA device creation
            self.device = torch.device('cuda')
            print("Initializing HyperspectralTrainer class...")
            print(f"✓ CUDA forced: {torch.cuda.get_device_name()}")
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ Device count: {torch.cuda.device_count()}")
            torch.cuda.empty_cache()
            print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        except Exception as e:
            # Fallback to CPU only if CUDA not available.
            self.device = torch.device('cpu')
            print(f"⚠️  CUDA failed, using CPU: {e}")
            print("⚠️  CPU training will be extremely slow!")

        os.environ['OMP_NUM_THREADS'] = '4'
        os.environ['MKL_NUM_THREADS'] = '4'
        print('Set CPU threats to 4 to reduce CPU usage.')
        
        # Initialize wandb (disabled by default for stability)
        if config.get('use_wandb', False):
            try:
                wandb.init(
                    project="enhanced-lola-specvit",
                    config=config,
                    name=f"{config.get('hf_backbone','LoLA_hsViT')}_{config['dataset_name']}_{self.epochs}epochs"
                )
            except Exception as e:
                print(f"Warning: wandb initialization failed: {e}")
                print("Continuing without wandb logging...")
                config['use_wandb'] = False
        
        # Load and preprocess data
        self.load_data()
        
        # Create model
        self.create_model()
        
        # Setup training components
        self.setup_training()
        
        # Training metrics tracking
        self.train_losses = []
        self.train_accuracies = []
        self.eval_accuracies = []
        self.learning_rates = [] # Added for warmup and adjustment

    @staticmethod
    def Monitor(f):
        """
        Monitor memory usage in process ``f``.

        ``f``: a function / method with attribute ``__name__``.
        """
        import psutil, GPUtil
        process = psutil.Process()

        cpu_percent = process.cpu_percent()
        cpu_num = process.cpu_num()
        memory_info = process.memory_info()
        
        try:
            print(f"Running process {f.__name__} with:\n",
                f"CPU (num / perc) = ({cpu_num}, {cpu_percent}%)\n",
                f"Using memory = {memory_info.rss/1024**3:.2f}GB."
                )
        
            if torch.cuda.is_available():
                gpu = GPUtil.getGPUs()[0]
                print(f"GPU memory used/total: {gpu.memoryUsed:.1f}/{gpu.memoryTotal:.1f} GB.")
        except:
            raise AttributeError(f"method {f} has no attribute __name__!")

    def load_data(self):
        """Load and preprocess hyperspectral data"""

        print("Loading hyperspectral data...")

        self.train_loader, self.test_loader, \
        all_loader, y_all, pca_components, self.dataset_info \
            = self.dataLoader.create_data_loader(num_workers=self.config['num_workers'])   
                         
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}")
        print(f"Total samples: {len(all_loader.dataset)}")
        print(f"PCA components: {pca_components}")
        print(f"Dataset: {self.config['dataset_name']} ({self.dataset_info['num_classes']} classes)")
        
    def create_model(self):
        """Create model with pretrained integration"""

        print(f"Creating model...")
        
        # Update num_classes based on dataset
        self.config['num_classes'] = self.dataset_info['num_classes']
        
        self.model, self.efficiency_results = create_model(
            spatial_size=15,
            num_classes=self.config['num_classes'],
            lora_rank=self.config.get('lora_rank', 16),
            lora_alpha=self.config.get('lora_alpha', 32),
            freeze_non_lora=True
        )

        # Load pretrained ViT weights if pretrained.
        if self.config.get('hf_backbone', None) and not self.config.get('skip_pretrained', False):
            print(f"Loading hsViT backbone (transformers): {self.config['hf_backbone']}")
            load_backbone_into_model(self.model, self.config['hf_backbone'])
        elif self.config.get('skip_pretrained', False):
            print("Skipping pretrained weights loading (training from scratch)")
        else:
            print("No pretrained backbone specified, training from scratch")

        # freeze non-LoRA params, shift model onto GPU.
        prepare_model_for_lora_finetuning(self.model)
        self.model = self.model.to(self.device)
        
    def setup_training(self):
        """
            Setup training components. Mainly on:
            - Optimizer groups: AMP on GPU, AdamW, cosine warm-starter, mix precision.
            - Loss function: Cross-Entropy.
        """
        # MEMORY OPTIMIZATION: Clear cache before setup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Set memory efficient settings
            torch.backends.cudnn.benchmark = False  # More memory efficient
            torch.backends.cudnn.deterministic = True
        
        # Disable AMP for CPU training
        if self.device.type == 'cpu':
            self.config['use_amp'] = False
            print("AMP disabled for CPU training")
        
        # Simple parameter grouping: detach lora from normal params.
        # for better fine-tuning.
        lora_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'lora_' in name:
                    lora_params.append(param)
                else:
                    other_params.append(param)
        
        # Use working model's exact optimizer setup for stability
        self.optimizer = optim.AdamW([
            {'params': lora_params, 'lr': self.config['learning_rate']},
            {'params': other_params, 'lr': self.config['learning_rate'] * 0.1}
        ], weight_decay=self.config['weight_decay'])
        
        # Create scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config['scheduler']['T_0'],
            T_mult=self.config['scheduler']['T_mult'],
            eta_min=self.config['scheduler']['eta_min']
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.config['label_smoothing'])
        
        # Gradient scaler for mixed precision, only for GPU
        self.scaler = GradScaler() if (self.config['use_amp'] and self.device.type == 'cuda') else None
        
        # Training metrics with best model state tracking
        self.best_accuracy = 0
        self.patience_counter = 0
        self.best_epoch = 0
        self.best_model_state = None  # Track best model state for restoration
 
    def visualize_cam(self, epoch, save_path=f'./outputs/CAM'):
        """
        Generate and save CAM visualizations for sample test images.
        """

        os.makedirs(save_path, exist_ok=True)
        self.model.eval()
    
        try:
            # Get first batch from test_loader.
            test_iter = iter(self.test_loader)
            batch = next(test_iter)

            if len(batch) == 3:
                hyperspectral, pretrained_input, labels = batch
            elif len(batch) == 2:
                hyperspectral, labels = batch
                pretrained_input = None
            else:
                raise ValueError(f"Unexpected batch format: {len(batch)} elements")
        
            # Move to device and ensure correct tensor shape [B, C, H, W]
            hyperspectral = hyperspectral.to(self.device)
            if hyperspectral.dim() == 4 and hyperspectral.shape[1] != self.config.get('in_channels', 15):
                # Fix channel dimension if needed (common hyperspectral data format issue)
                hyperspectral = hyperspectral.permute(0, 3, 1, 2)  # if not: [B, H, W, C] -> [B, C, H, W]
        
            # Generate CAM for first 5 samples (or fewer if batch is smaller)
            num_samples = min(5, hyperspectral.shape[0])
            for sample_idx in range(num_samples):
                # Get single sample (keep batch dimension for model input)
                sample = hyperspectral[sample_idx:sample_idx+1]
                true_label = labels[sample_idx].item()
            
                # CAM generation.
                cam_image, pred_label, raw_cam = generate_cam_plot(
                    model=self.model,
                    input_tensor=sample,
                    target_class=None,  # Use predicted class
                    device=self.device
                   )
            
                # Create visualization plot
                plt.figure(figsize=(10, 4))
            
                # Plot 1: Original hyperspectral data (use first 3 bands as RGB for visualization)
                plt.subplot(1, 2, 1)
                # Normalize sample for display
                vis_sample = sample[0].cpu().numpy()
                vis_sample = (vis_sample - vis_sample.min()) / (vis_sample.max() - vis_sample.min() + 1e-8)
                # Use first 3 bands for simulating RGB.
                if vis_sample.shape[0] >= 3:
                    rgb_sample = vis_sample[:3].transpose(1, 2, 0)
                else:
                    # If fewer than 3 bands, repeat single band for RGB
                    rgb_sample = np.repeat(vis_sample[0:1].transpose(1, 2, 0), 3, axis=2)
                plt.imshow(rgb_sample)
                plt.title(f'ComposedRGB (Label: {true_label})')
                plt.axis('off')
                
                # Plot 2: CAM visualization
                plt.subplot(1, 2, 2)
                plt.imshow(cam_image, cmap='jet')
                plt.title(f'CAM (Pred: {pred_label})')
                plt.axis('off')
                
                # Add color bar for CAM intensity
                plt.colorbar(fraction=0.046, pad=0.04)
                
                # Save the plot.
                save_filename = f'cam_epoch_{epoch}_sample_{sample_idx}.png'
                full_save_path = os.path.join(save_path, save_filename)
                plt.tight_layout()
                plt.savefig(full_save_path, dpi=150, bbox_inches='tight')
                plt.close()
        
            # Handle exception to avoid training break.
        except Exception as e:
            print(f'Failed to generate CAM for epoch{epoch} to: {save_path}.')

            import traceback
            traceback.print_exc()
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        device_type = self.device.type
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Learning rate warmup.
        if epoch < self.config.get('warmup_epochs', 10):
            warmup_factor = (epoch + 1) / self.config.get('warmup_epochs', 10)
            current_lr = self.config['learning_rate'] * warmup_factor
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.epochs}')
        accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        
        for batch_idx, data in enumerate(pbar):
            # Handle different data loader outputs
            if len(data) == 3:
                hyperspectral, pretrained, labels = data
                pretrained = pretrained.to(self.device)
            elif len(data) == 2:
                hyperspectral, labels = data
            else:
                raise ValueError(f"Unexpected data format: {len(data)} items")
            
            hyperspectral = hyperspectral.to(self.device)
            labels = labels.to(self.device)
            
            # DEBUG: Check tensor shapes
            # if batch_idx == 0:
            #     print(f"DEBUG - hyperspectral shape: {hyperspectral.shape}, labels shape: {labels.shape}")
            
            # Ensure correct tensor format for hyperspectral data
            if hyperspectral.dim() == 4 and hyperspectral.shape[-1] == 15:  # [B, H, W, C]
                hyperspectral = hyperspectral.permute(0, 3, 1, 2)  # [B, C, H, W]
            
            # Ensure labels are not empty and have correct shape
            if labels.numel() == 0:
                print(f"ERROR: Empty labels at batch {batch_idx}, skipping...")
                continue
                
            # Ensure labels are 1D for cross_entropy.
            if labels.dim() > 1:
                labels = labels.squeeze()
            
            # Enhanced normalization.
            hyperspectral = (hyperspectral - hyperspectral.mean(dim=(2,3), keepdim=True)) / (hyperspectral.std(dim=(2,3), keepdim=True) + 1e-8)
            
            # Zero gradients only at the start of accumulation cycle
            if batch_idx % accumulation_steps == 0:
                self.optimizer.zero_grad()
            
            # Handle AMP for GPU.
            if self.config['use_amp'] and self.device.type == 'cuda':
                # with autocast(device_type='cuda'):
                with autocast():
                    outputs = self.model(hyperspectral)
                    loss = self.criterion(outputs, labels)
                    loss = loss / accumulation_steps
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    if self.config.get('grad_clip', 0) > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                # CPU training has no AMP.
                outputs = self.model(hyperspectral)
                loss = self.criterion(outputs, labels)
                loss = loss / accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    if self.config.get('grad_clip', 0) > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
                    self.optimizer.step()
            
            # Update scheduler.
            if epoch >= self.config.get('warmup_epochs', 10):
                self.scheduler.step(epoch + batch_idx / len(self.train_loader))
            
            # Calculate accuracy.
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Use actual loss for metrics, undo accumulation scaling.
            actual_loss = loss.item() * accumulation_steps if accumulation_steps > 1 else loss.item()
            total_loss += actual_loss
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{actual_loss:.4f}',
                'acc': f'{100.*correct/total:.2f}%',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # Log to wandb (disabled by default for memory)
            if self.config.get('use_wandb', False):
                wandb.log({
                    'train_loss': actual_loss,
                    'train_accuracy': 100.*correct/total,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # MEMORY OPTIMIZATION: Clear cache and delete tensors
            if batch_idx % 5 == 0 and self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Delete intermediate variables to free memory
            del predicted
            if 'actual_loss' in locals():
                del actual_loss
        
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        # Store metrics
        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_acc)
        self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
        
        return epoch_loss, epoch_acc
    
    def evaluate(self):
        """Evaluate model on the full test set (uses larger eval batch size)."""
        self.model.eval()
        total_loss = 0
        predictions = []
        targets_all = []
        eval_batch_size = self.config.get('eval_batch_size', 64)
        
        # Reduce batch size for CPU training
        if self.device.type == 'cpu':
            eval_batch_size = min(eval_batch_size, 16)  # Smaller batches for CPU
            print(f"CPU training detected, reducing eval batch size to {eval_batch_size}")
        
        # full evaluation on entire test set
        eval_loader = torch.utils.data.DataLoader(
            self.test_loader.dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=False,
        )
        print(f"Full evaluation: {len(self.test_loader.dataset)} samples with batch size: {eval_batch_size}")
        
        with torch.no_grad():
            for hyperspectral, pretrained, labels in tqdm(eval_loader, desc='Evaluating'):
                hyperspectral = hyperspectral.to(self.device, non_blocking=True)
                pretrained = pretrained.to(self.device, non_blocking=True)
                labels = labels.squeeze().to(self.device, non_blocking=True)
                
                if hyperspectral.dim() == 4 and hyperspectral.shape[-1] == 15:  # [B, H, W, C]
                    hyperspectral = hyperspectral.permute(0, 3, 1, 2)  # [B, C, H, W]
                
                hyperspectral = (hyperspectral - hyperspectral.mean(dim=(2,3), keepdim=True)) / (hyperspectral.std(dim=(2,3), keepdim=True) + 1e-8)
                outputs = self.model(hyperspectral)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                predictions.extend(predicted.cpu().numpy())
                targets_all.extend(labels.cpu().numpy())
        
        predictions = np.array(predictions)
        targets_all = np.array(targets_all)
        accuracy = accuracy_score(targets_all, predictions) * 100
        kappa = cohen_kappa_score(targets_all, predictions) * 100
        self.eval_accuracies.append(accuracy)
        return total_loss / len(eval_loader), accuracy, kappa, predictions, targets_all
    
    def train(self, debug_mode=False):
        """Main training loop"""
        if debug_mode:
            print(f'Debug mode on. Eval-cam 4 every epoch.')
        print("Starting training...")
        
        # Initialize wandb.
        if self.config.get('use_wandb', False):
            try:
                wandb.init(
                    project="enhanced-lola-specvit",
                    config=self.config,
                    name=f"{self.config['pretrained_model_name']}_{self.config['dataset_name']}_{self.epochs}epochs"
                )
            except Exception as e:
                print(f"Warning: wandb initialization failed: {e}")
                print("Continuing without wandb logging...")
                self.config['use_wandb'] = False
        
        # Training time measurement
        tic1 = time.perf_counter()
        from configs import output_path

        for epoch in range(self.epochs):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Evaluate per eval_interval, draw CAM.
            should_eval = ((epoch + 1) % self.config.get('eval_interval', 1) == 0) or (epoch + 1 == self.epochs)
            
            if debug_mode or should_eval:
                print(f"\nEvaluating epoch {epoch+1}...")
                test_loss, test_acc, kappa, predictions, labels = self.evaluate()
                self.visualize_cam(epoch=epoch+1, save_path=output_path+'/CAM')  # Use 1-indexed for display
                self.process_monitor.check_process_count()
            else:
                test_loss, test_acc, kappa = None, None, None
            
            print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')

            if should_eval:
                print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, Kappa: {kappa:.4f}')
            
            # Log to wandb
            if self.config.get('use_wandb', False):
                wandb.log({
                    'train_loss': train_loss,
                    'train_accuracy': train_acc,
                    **({'test_loss': test_loss, 'test_accuracy': test_acc, 'kappa': kappa} if should_eval else {}),
                    'epoch': epoch
                })
            
            # Save best model with stability monitoring
            if should_eval and test_acc > self.best_accuracy:
                self.best_accuracy = test_acc
                self.best_epoch = epoch
                self.patience_counter = 0
                
                # Save best model state for potential restoration
                self.best_model_state = self.model.state_dict().copy()
                
                # Save best model with full state.
                checkpoint_data = {
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'accuracy': test_acc,
                    'config': self.config
                }
                
                # Only save scheduler if it has state_dict method
                if hasattr(self.scheduler, 'state_dict'):
                    checkpoint_data['scheduler'] = self.scheduler.state_dict()
                
                # Save best model
                best_model_path = output_path + f'/model_{self.config["dataset_name"]}.pth'
                torch.save(checkpoint_data, best_model_path)
                print(f'Best model saved: {best_model_path}')
            elif should_eval:
                self.patience_counter += 1
                
                # If accuracy drops significantly, restore best model
                if test_acc < self.best_accuracy - 2.0:  # 2% drop threshold
                    print(f"Accuracy dropped significantly from {self.best_accuracy:.2f}% to {test_acc:.2f}%, restoring best model from epoch {self.best_epoch}")
                    if self.best_model_state is not None:
                        self.model.load_state_dict(self.best_model_state)
                        print("Best model state restored")
                
            # Early stopping
            if should_eval and self.patience_counter >= self.config['patience']:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        toc1 = time.perf_counter()
        training_time = toc1 - tic1
        
        print(f"Training completed. Best accuracy: {self.best_accuracy:.2f}% at epoch {self.best_epoch}")
        
        # Finish wandb.
        if self.config.get('use_wandb', False):
            wandb.finish()
        
        # Final eval.
        # try to load the best model otherwise current one.
        try:
            print('For final evaluation:')
            print(f"Loading model from: model_{self.config['dataset_name']}.pth")
            checkpoint = torch.load(output_path + f'model_{self.config["dataset_name"]}.pth')
            
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                print("Loading full checkpoint with state...")
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                print("Loading state dict only...")
                self.model.load_state_dict(checkpoint)
            print("Successfully load model for final eval.")

        except Exception as e:
            print(f"Error loading model: {e}")
            print("Attempting to continue with current model state...")
        
        self.model.eval()
        print("Model set to evaluation mode for final testing.")

        # Final evaluation.
        tic2 = time.perf_counter()
        y_pred_test, y_test = test(self.device, self.model, self.test_loader)
        toc2 = time.perf_counter()
        test_time = toc2 - tic2

        # Calculate comprehensive metrics.
        classification, oa, confusion, each_acc, aa, kappa, target_names = acc_reports(
            y_test, y_pred_test, dataset_name=self.config['dataset_name']
        )

        # Save results for best model.
        file_name = save_results(
            self.model, self.config, self.best_accuracy, self.best_epoch, 
            training_time, test_time, classification, oa, confusion, 
            each_acc, aa, kappa, target_names, 
            'LoLA_hsViT', self.efficiency_results
        )

        save_model(
            self.model, self.config, self.best_accuracy, kappa, 
            training_time, test_time, self.best_epoch, each_acc, confusion,
            'LoLA_hsViT', self.efficiency_results
        )

        # Optional Step: Merge LoRA into base for inference
        if self.config.get('merge_lora_for_inference', False):
            print("Merging LoRA adapters into base linear layers for inference...")
            merge_lora_for_inference(self.model)
            print("LoRA successfully merged.")
        
        # Plot training curves
        plt_curves(self.train_losses, self.train_accuracies, self.eval_accuracies, epoch)
        
        # Display comprehensive final results.
        print(f"\n{'='*20}")
        print(f"FINAL TRAINING RESULTS ON {self.config['dataset_name']}")
        print(f"{'='*20}")
        print(f"Best Accuracy: {self.best_accuracy:.2f}% at epoch {self.best_epoch}")
        print(f"Overall Accuracy: {oa:.2f}%")
        print(f"Average Accuracy: {aa:.2f}%")
        print(f"Kappa Score: {kappa:.2f}%")
        print(f"Training Time: {training_time:.2f} seconds")
        print(f"Test Time: {test_time:.2f} seconds")
        print(f"\nPer-Class Accuracies:")
        
        for name, acc in zip(target_names, each_acc):
            print(f"  {name}: {acc:.2f}%")

        print(f"\nDetailed Classification Report:")
        print(classification)
        print(f"\nConfusion Matrix:")
        print(confusion)
        print(f"\nResults saved to: {file_name}")
        print(f"{'='*20}")
        
        return self.best_accuracy, self.best_epoch

def main():
    # param group 1: get by argparse in terminal cmd:
    # python train.py --dataset 'LongKou' --epoch 40
    import argparse
    parser = argparse.ArgumentParser(description='Hyperspectral Image Classification')
    parser.add_argument('--dataset', type=str, default='LongKou', 
                       choices=['LongKou', 'IndianPines', 'PaviaU', 'PaviaC', 'Salinas', 'HongHu', 'Qingyun'],
                       help='Dataset to use for training/evaluation')
    parser.add_argument('--epoch', type=int, default=30,
                       help='Total epoch for training.')
    parser.add_argument('--model', type=str, default=None,
                       help='Backbone Hugging Face model to use (e.g., nvidia/GCViT, nvidia/GCViT-Tiny)')
    parser.add_argument('--skip-pretrained', action='store_true', default=False,
                       help='Skip loading pretrained weights (train from scratch)')
    parser.add_argument('--eval-only', action='store_true', default=False,
                       help='Run evaluation only, skip training')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to saved model for evaluation-only mode')
    parser.add_argument('--debug-mode', type=bool, default=False,
                        help='debug sets eval-and-cam every epoch.')
    
    args = parser.parse_args()

    # param group 2: import from configs.py.
    from configs import (data_path, label_path,\
          data_key, label_key, config, target_names)

    # Create trainer.
    dataLoader = MatHyperspectralDataset(
        data_path=data_path,
        label_path=label_path,
        data_key=data_key, label_key=label_key, 
        target_names=target_names, num_classes=len(target_names),
        patch_size=config['patch_size'],
        batch_size=config['batch_size'],
        test_rate=config['test_rate'],
        pca_components=config['pca_components'],
        pin_memory=config['pin_memory']
        )

    trainer = HyperspectralTrainer(config, dataLoader, epochs=args.epoch)

    if args.eval_only:
    # Evaluation-only mode
        if args.model_path is None:
            # Try to find the best model.
            model_path = f'model_{config["dataset_name"]}.pth'
            if not os.path.exists(model_path):
                print(f"Error: No saved model found at {model_path}")
                print("Provide a model path using --model-path or train the model first.")
                return
        else:
            model_path = args.model_path
    # train-and-evaluate mode
    else:
        trainer.train(debug_mode=args.debug_mode)

if __name__ == "__main__":
    main()
