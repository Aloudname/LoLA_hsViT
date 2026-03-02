"""
core.py
Unified training pipeline and model factory for hyperspectral image classification.

Public API:
    - TrainPipeline: training pipeline class.
    - TrainResult: single training result.
    - CVResult: for cross-validation results.
    - ModelFactory: models with different configs and structures.

Usage:
    >>> from config import load_config
        from pipeline import NpyHSDataset, TrainPipeline, ModelFactory
    
        config = load_config()
        dataset = NpyHSDataset(config)
        model_fn = ModelFactory.create("LoLA_hsViT", config)

        pipeline = TrainPipeline(config, dataset, model_fn, name="my_exp")
        result = pipeline.run(epochs=50)          # single training
        result = pipeline.run_cv(n_folds=5, epochs=50)  # cross-validation
"""

import os
import json
import torch
from typing import Callable, Optional, Dict, List, Any, Union
from dataclasses import dataclass, field, asdict
from munch import Munch
from torch.nn import Module

from pipeline.dataset import AbstractHSDataset
from pipeline.trainer import hsTrainer
from pipeline.monitor import tprint

__all__ = [
    'TrainPipeline',
    'TrainResult',
    'CVResult', 
    'ModelFactory',
]


@dataclass
class TrainResult:
    """Wrapper for single training result"""
    
    best_accuracy: float = 0.0
    final_accuracy: float = 0.0
    final_kappa: float = 0.0
    final_miou: float = 0.0
    
    best_epoch: int = 0
    training_time: float = 0.0
    total_epochs: int = 0
    
    model_path: str = ""
    output_dir: str = ""
    
    train_losses: List[float] = field(default_factory=list)
    eval_losses: List[float] = field(default_factory=list)
    train_accs: List[float] = field(default_factory=list)
    eval_accs: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def summary(self) -> str:
        return (f"TrainResult(acc={self.final_accuracy:.2f}%, "
                f"kappa={self.final_kappa:.2f}%, "
                f"mIoU={self.final_miou:.2f}%, "
                f"best_epoch={self.best_epoch})")
    
    def __repr__(self) -> str:
        return self.summary()


@dataclass
class CVResult:
    """Wrapper for cross-validation results"""
    
    # mean of metrics across folds
    accuracy_mean: float = 0.0
    accuracy_std: float = 0.0
    kappa_mean: float = 0.0
    kappa_std: float = 0.0
    miou_mean: float = 0.0
    miou_std: float = 0.0
    overfit_gap_mean: float = 0.0
    
    n_folds: int = 0
    total_time: float = 0.0
    
    # list of every folds
    fold_accuracies: List[float] = field(default_factory=list)
    fold_kappas: List[float] = field(default_factory=list)
    fold_mious: List[float] = field(default_factory=list)
    
    best_fold: int = 0
    best_model_path: str = ""
    output_dir: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def summary(self) -> str:
        return (f"CVResult({self.n_folds}-fold, "
                f"acc={self.accuracy_mean:.2f}±{self.accuracy_std:.2f}%, "
                f"kappa={self.kappa_mean:.2f}±{self.kappa_std:.2f}%)")
    
    def __repr__(self) -> str:
        return self.summary()

class ModelFactory:
    """
    Unified model creator for hsi classification.
    
    model options:
        - 'lola' -> LoLA_hsViT
        - 'common' -> CommonViT
        - 'unet' -> Unet
    
    Usage:
        model_fn = ModelFactory.create("LoLA_hsViT", config)
        model = model_fn()
    """
    
    _registry: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, model_class: type) -> None:
        """register a model class with a name"""
        cls._registry[name.lower()] = model_class
    
    @classmethod
    def available_models(cls) -> List[str]:
        """all optional model names"""
        cls._ensure_registered()
        return list(cls._registry.keys())
    
    @classmethod
    def create(cls, 
               model_name: str, 
               config: Munch,
               **override_kwargs) -> Callable[..., Module]:
        """
        model creator with config and optional overrides.
        
        Args:
            model_name: optional of LoLA_hsViT/CommonViT/Unet
            config: Munch
            **override_kwargs: key-value pairs could override default model params.
        
        Returns:
            Callable: return a model constructor (callable) to create a model instance.
        """
        cls._ensure_registered()
        
        name_lower = model_name.lower()
        if name_lower not in cls._registry:
            available = ', '.join(cls._registry.keys())
            raise ValueError(f"Unknown model '{model_name}'. "
                           f"Available: {available}")
        
        model_class = cls._registry[name_lower]
        
        # default params from config
        base_kwargs = {
            'in_channels': config.preprocess.pca_components,
            'num_classes': config.clsf.num,
            'patch_size': config.split.patch_size,
        }
        
        # LoLA params
        if name_lower == 'lola':
            base_kwargs.update({
                'r': config.lora.lora_rank,
                'lora_alpha': config.lora.lora_alpha,
            })
        
        # apply overrides
        base_kwargs.update(override_kwargs)
        
        # return a constructor function that creates the model instance when called
        def _model_constructor():
            return model_class(**base_kwargs)
        
        return _model_constructor
    
    @classmethod
    def _ensure_registered(cls):
        """
        NOTE: This method is called before any model creation,
            to ensure that the model classes are registered.
        """
        if cls._registry:
            return
        
        from model import LoLA_hsViT, CommonViT, Unet
        cls._registry = {
            'lola': LoLA_hsViT,
            'common': CommonViT,
            'unet': Unet,
        }

class TrainPipeline:
    """
    Entry point for training and evaluation.
    
    Public API:
        - run(epochs): single model train + eval
        - run_cv(n_folds, epochs): K-fold cross-validation
        - summary(): config and dataset summary printout
    
    Usage:
        >>> pipeline = TrainPipeline(config, dataset, model_fn, name="exp1")
            result = pipeline.run(epochs=50)
            print(result)
    """
    
    def __init__(self,
                 config: Munch,
                 dataset: AbstractHSDataset,
                 model_fn: Callable[..., Module],
                 name: str = "experiment",
                 num_gpus: Optional[int] = None,
                 debug_mode: bool = False):
        """        
        Args:
            config: Munch from load_conffig().
            dataset: torch.utils.dataset / AbstractHSDataset instance.
            model_fn: callable returns a model instance when called,
                    e.g. ModelFactory.create(...)
            name: experiment name for output directory.
            num_gpus: None=auto-detect
            debug_mode: if True, generate CAM per epoch (slow).
        """
        self._config = config
        self._dataset = dataset
        self._model_fn = model_fn
        self._name = name
        self._debug_mode = debug_mode
        
        if num_gpus is None:
            self._num_gpus = (torch.cuda.device_count() 
                            if torch.cuda.is_available() else 0)
        else:
            self._num_gpus = num_gpus
        
        self._output_dir = os.path.join(config.path.output, name)
        os.makedirs(self._output_dir, exist_ok=True)
        
        tprint(f"TrainPipeline initialized: {name}")
        print(f"  Dataset: {len(dataset)} samples")
        print(f"  GPUs: {self._num_gpus}")
        print(f"  Output: {self._output_dir}")
    
    def run(self, epochs: int = 50) -> TrainResult:
        """
        run single training and evaluation.
        
        Returns:
            TrainResult: A `TrainResult` wrapper.
        """
        tprint(f"Starting training: {self._name}")
        
        trainer = hsTrainer(
            config=self._config,
            dataLoader=self._dataset,
            epochs=epochs,
            model=self._model_fn,
            model_name=self._name,
            debug_mode=self._debug_mode,
            num_gpus=self._num_gpus,
        )
        
        raw_result = trainer.train()
        
        # wrap raw result into TrainResult dataclass
        result = TrainResult(
            best_accuracy=raw_result.get('best_accuracy', 0.0),
            final_accuracy=raw_result.get('final_accuracy', 0.0),
            final_kappa=raw_result.get('final_kappa', 0.0),
            final_miou=raw_result.get('final_miou', 0.0),
            best_epoch=raw_result.get('best_epoch', 0),
            training_time=raw_result.get('training_time', 0.0),
            total_epochs=epochs,
            model_path=raw_result.get('model_path', ''),
            output_dir=self._output_dir,
            train_losses=trainer.train_losses,
            eval_losses=trainer.eval_losses,
            train_accs=trainer.train_accs,
            eval_accs=trainer.eval_accs,
        )
        
        result.save(os.path.join(self._output_dir, 'result.json'))
        
        tprint(f"Training completed: {result.summary()}")
        return result
    
    def run_cv(self, 
               n_folds: int = 5, 
               epochs: int = 50) -> CVResult:
        """
        run K-fold cross-validation.
        
        Args:
            n_folds: if n_folds=0, run single training.
            epochs: training epochs for each fold.
        
        Returns:
            CVResult: A `CVResult` wrapper.
        """
        tprint(f"Starting {n_folds}-fold CV: {self._name}")
        
        raw_result = hsTrainer.cross_validate(
            config=self._config,
            dataLoader=self._dataset,
            n_folds=n_folds,
            epochs=epochs,
            model=self._model_fn,
            model_name=self._name,
            num_gpus=self._num_gpus,
            debug_mode=self._debug_mode,
        )
        
        # wrap raw result into CVResult dataclass
        import numpy as np
        fold_accs = raw_result.get('cv_fold_accuracies', [])
        fold_kappas = raw_result.get('cv_fold_kappas', [])
        
        result = CVResult(
            accuracy_mean=raw_result.get('cv_accuracy_mean', 0.0),
            accuracy_std=raw_result.get('cv_accuracy_std', 0.0),
            kappa_mean=raw_result.get('cv_kappa_mean', 0.0),
            kappa_std=raw_result.get('cv_kappa_std', 0.0),
            miou_mean=raw_result.get('cv_miou_mean', 0.0),
            miou_std=raw_result.get('cv_miou_std', 0.0),
            overfit_gap_mean=raw_result.get('cv_overfit_gap_mean', 0.0),
            n_folds=n_folds,
            total_time=raw_result.get('training_time', 0.0),
            fold_accuracies=fold_accs,
            fold_kappas=fold_kappas,
            best_fold=int(np.argmax(fold_accs)) + 1 if fold_accs else 0,
            best_model_path=raw_result.get('model_path', ''),
            output_dir=raw_result.get('output_dir', self._output_dir),
        )
        
        result.save(os.path.join(self._output_dir, 'cv_result.json'))
        
        tprint(f"CV completed: {result.summary()}")
        return result
    
    def summary(self) -> None:
        print("\n")
        print(f"TrainPipeline: {self._name}")
        print(f"  Dataset: {len(self._dataset)} patches")
        print(f"  Classes: {self._config.clsf.num} ({', '.join(self._config.clsf.targets)})")
        print(f"  Patch size: {self._config.split.patch_size}")
        print(f"  Input channels: {self._config.preprocess.pca_components}")
        print(f"  GPUs: {self._num_gpus}")
        print(f"  Output: {self._output_dir}")
        print("\n")
    
    # properties for read-only access to important attributes
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def output_dir(self) -> str:
        return self._output_dir
    
    @property
    def config(self) -> Munch:
        return self._config
