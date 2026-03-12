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

from sklearn.base import BaseEstimator
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
    
    Supports base models and tagged variants:
        - 'lola' / 'lola_reduced' / 'lola_tiny' / 'lola_mini' / 'lola_2layer'
        - 'common' / 'common_reduced' / 'common_tiny' / 'common_mini' / 'common_2layer'
        - 'unet'
    
    Usage:
        model_fn = ModelFactory.create("lola_mini", config)
        model = model_fn()
    """
    
    _registry: Dict[str, type] = {}
    
    # Valid tags for variant selection
    VALID_TAGS = ('full', 'reduced', 'tiny', 'mini', '2layer')
    
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
    def resolve_name(cls, model: str, tag: str = 'full') -> str:
        """Resolve a (model, tag) pair to a registry key.
        
        Examples:
            resolve_name('lola', 'mini')  -> 'lola_mini'
            resolve_name('common', 'full') -> 'common'
        """
        if tag == 'full':
            return model.lower()
        return f"{model.lower()}_{tag}"
    
    @classmethod
    def create(cls, 
               model_name: str, 
               config: Munch,
               **override_kwargs) -> Callable[..., Module]:
        """
        model creator with config and optional overrides.
        
        Args:
            model_name: e.g. 'lola', 'common_mini', 'lola_2layer', 'unet'
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
        
        # LoRA params for lola variants
        if name_lower.startswith('lola'):
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
        
        from model import (LoLA_hsViT, CommonViT, Unet,
                           LoLA_hsViT_reduced, LoLA_hsViT_tiny,
                           LoLA_hsViT_mini, LoLA_hsViT_2layer,
                           CommonViT_reduced, CommonViT_tiny,
                           CommonViT_mini, CommonViT_2layer)
        cls._registry = {
            'lola': LoLA_hsViT,
            'lola_reduced': LoLA_hsViT_reduced,
            'lola_tiny': LoLA_hsViT_tiny,
            'lola_mini': LoLA_hsViT_mini,
            'lola_2layer': LoLA_hsViT_2layer,
            'common': CommonViT,
            'common_reduced': CommonViT_reduced,
            'common_tiny': CommonViT_tiny,
            'common_mini': CommonViT_mini,
            'common_2layer': CommonViT_2layer,
            'unet': Unet,
        }

class TrainPipeline(BaseEstimator):
    """
    Sklearn-compatible entry point for training and evaluation.

    Follows sklearn API convention::

        pipeline = TrainPipeline(config, dataset, model_fn, name="exp1")
        pipeline.fit(epochs=50)        # train, stores result in pipeline.result_
        pipeline.predict()             # predict on test set
        pipeline.score()               # evaluate accuracy

    Backward-compatible aliases::

        result = pipeline.run(epochs=50)      # returns TrainResult directly
        result = pipeline.run_cv(n_folds=5)   # returns CVResult directly
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
    
    def fit(self, X=None, y=None, epochs: int = 50) -> 'TrainPipeline':
        """
        Train the model. Follows sklearn ``fit`` convention.

        Args:
            X : ignored (data comes from internal dataset).
            y : ignored.
            epochs : training epochs.

        Returns:
            self
        """
        tprint(f"Starting training: {self._name}")
        
        self._trainer = hsTrainer(
            config=self._config,
            dataLoader=self._dataset,
            epochs=epochs,
            model=self._model_fn,
            model_name=self._name,
            debug_mode=self._debug_mode,
            num_gpus=self._num_gpus,
        )
        
        self._trainer.fit()
        raw_result = self._trainer.results_
        
        # wrap raw result into TrainResult dataclass
        self.result_ = TrainResult(
            best_accuracy=raw_result.get('best_accuracy', 0.0),
            final_accuracy=raw_result.get('final_accuracy', 0.0),
            final_kappa=raw_result.get('final_kappa', 0.0),
            final_miou=raw_result.get('final_miou', 0.0),
            best_epoch=raw_result.get('best_epoch', 0),
            training_time=raw_result.get('training_time', 0.0),
            total_epochs=epochs,
            model_path=raw_result.get('model_path', ''),
            output_dir=self._output_dir,
            train_losses=self._trainer.train_losses,
            eval_losses=self._trainer.eval_losses,
            train_accs=self._trainer.train_accs,
            eval_accs=self._trainer.eval_accs,
        )
        
        self.result_.save(os.path.join(self._output_dir, 'result.json'))
        
        tprint(f"Training completed: {self.result_.summary()}")
        return self

    def predict(self, X=None):
        """
        Predict using the trained model.

        Args:
            X : DataLoader or None (uses test set).

        Returns:
            np.ndarray of predictions.
        """
        if not hasattr(self, '_trainer'):
            raise RuntimeError("Call fit() before predict().")
        return self._trainer.predict(X)

    def score(self, X=None, y=None):
        """
        Evaluate and return balanced accuracy.

        Args:
            X : DataLoader or None (uses val/test set).
            y : ignored.

        Returns:
            float: balanced accuracy percentage.
        """
        if not hasattr(self, '_trainer'):
            raise RuntimeError("Call fit() before score().")
        return self._trainer.score(X)

    def run(self, epochs: int = 50) -> TrainResult:
        """
        Backward-compatible alias for ``fit()``.
        Returns TrainResult directly.
        """
        self.fit(epochs=epochs)
        return self.result_
    
    def fit_cv(self, n_folds: int = 5, epochs: int = 50) -> 'TrainPipeline':
        """
        Cross-validation fit. Follows sklearn convention, returns ``self``.

        Result is stored in ``self.result_`` (a CVResult).

        Args:
            n_folds : number of folds (0 falls back to single training).
            epochs  : training epochs per fold.

        Returns:
            self
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
        
        self.result_ = CVResult(
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
        
        self.result_.save(os.path.join(self._output_dir, 'cv_result.json'))
        
        tprint(f"CV completed: {self.result_.summary()}")
        return self
    
    def run_cv(self, 
               n_folds: int = 5, 
               epochs: int = 50) -> CVResult:
        """
        Backward-compatible alias for ``fit_cv()``.
        Returns CVResult directly.
        """
        self.fit_cv(n_folds=n_folds, epochs=epochs)
        return self.result_
    
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
