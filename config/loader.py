# -*- coding: utf-8 -*-
# config.py using munch to load .yaml to an easy-accessible object.
import yaml
from munch import Munch


def load_config(yaml_file='config/config.yaml'):
    with open(yaml_file, 'r', encoding='utf-8') as f:
        config = Munch.fromDict(yaml.safe_load(f))
    
    _ = check_form(config)
    if _:
        return config
    else:
        raise ValueError(f"{yaml_file}.yaml format check failed!")

def check_form(config: Munch = None) -> bool:
    # Add any necessary checks for the config here
    for key in config.keys():
        if key not in ['dataset_name', 'common', 'lora', 'split', 'preprocess', 'memory', 'path', 'key', 'clsf', 'rgb']:
            raise ValueError(f"Unexpected key '{key}' found in config")
        
    for key in config.keys():
        if getattr(config, key) is None:
            raise ValueError(f"Config value for key '{key}' cannot be None")
        for subkey in getattr(config, key).keys():
            if getattr(config, key)[subkey] is None:
                raise ValueError(f"Config value for key '{key}.{subkey}' cannot be None")
    
    if config.split.patch_size % 2 == 0:
        raise ValueError(f"patch_size must be odd, got {config.split.patch_size}")
    if config.clsf.num != len(config.clsf.targets):
        raise ValueError(f"clsf.num must match length of targets, got {config.clsf.num} and {len(config.clsf.targets)}")
    if config.clsf.num < 2:
        raise ValueError(f"clsf.num must be >= 2, got {config.clsf.num}")

    # Dataset-stage validation for remap and preprocessor-fit settings
    unknown_policy = str(getattr(config.clsf, 'unknown_label_policy', 'error')).lower()
    if unknown_policy not in ['error', 'map_to_class0']:
        raise ValueError(
            f"clsf.unknown_label_policy must be 'error' or 'map_to_class0', got {unknown_policy}"
        )

    if hasattr(config.preprocess, 'class0_sample_ratio'):
        class0_ratio = float(config.preprocess.class0_sample_ratio)
        if class0_ratio < 0:
            raise ValueError(f"preprocess.class0_sample_ratio must be >= 0, got {class0_ratio}")

    # New split-first cache pipeline checks
    if hasattr(config.split, 'train_ratio') and hasattr(config.split, 'val_ratio') and hasattr(config.split, 'test_ratio'):
        tr = float(config.split.train_ratio)
        vr = float(config.split.val_ratio)
        ter = float(config.split.test_ratio)
        if tr <= 0 or vr < 0 or ter < 0:
            raise ValueError("split train/val/test ratios must satisfy train>0 and val/test>=0")
        if tr + vr + ter <= 0:
            raise ValueError("split train/val/test ratio sum must be > 0")

    if hasattr(config.split, 'samples_per_step'):
        sps = int(config.split.samples_per_step)
        if sps <= 0:
            raise ValueError("split.samples_per_step must be > 0")

    if hasattr(config.split, 'max_patches_per_sample_chunk'):
        mpsc = int(config.split.max_patches_per_sample_chunk)
        if mpsc < 0:
            raise ValueError("split.max_patches_per_sample_chunk must be >= 0")

    if hasattr(config.split, 'target_patches_per_step'):
        tpps = int(config.split.target_patches_per_step)
        if tpps <= 0:
            raise ValueError("split.target_patches_per_step must be > 0")

    if hasattr(config.memory, 'cached_loader_num_workers'):
        cwn = int(config.memory.cached_loader_num_workers)
        if cwn < 0:
            raise ValueError("memory.cached_loader_num_workers must be >= 0")

    if hasattr(config.memory, 'cached_loader_prefetch_factor'):
        cpf = int(config.memory.cached_loader_prefetch_factor)
        if cpf <= 0:
            raise ValueError("memory.cached_loader_prefetch_factor must be > 0")

    if hasattr(config.memory, 'cached_loader_patch_budget_mb'):
        cpb = int(config.memory.cached_loader_patch_budget_mb)
        if cpb < 64:
            raise ValueError("memory.cached_loader_patch_budget_mb must be >= 64")

    if hasattr(config.preprocess, 'enable_split_cache_pipeline'):
        _ = bool(config.preprocess.enable_split_cache_pipeline)

    if hasattr(config.common, 'early_stop_metric'):
        metric = str(config.common.early_stop_metric).lower()
        if metric not in ['eval']:
            raise ValueError(
                f"common.early_stop_metric must be 'eval', got {metric}"
            )

    if hasattr(config, 'rgb') and config.rgb is not None:
        rgb = config.rgb
        if hasattr(rgb, 'split'):
            rgb_split = rgb.split
            if hasattr(rgb_split, 'patch_size'):
                if int(rgb_split.patch_size) % 2 == 0:
                    raise ValueError(f"rgb.split.patch_size must be odd, got {rgb_split.patch_size}")
            for key in ['train_ratio', 'val_ratio', 'test_ratio']:
                if hasattr(rgb_split, key) and float(getattr(rgb_split, key)) < 0:
                    raise ValueError(f"rgb.split.{key} must be >= 0")

        if hasattr(rgb, 'preprocess') and hasattr(rgb.preprocess, 'in_channels'):
            if int(rgb.preprocess.in_channels) <= 0:
                raise ValueError("rgb.preprocess.in_channels must be > 0")

        if hasattr(rgb, 'path'):
            if not hasattr(rgb.path, 'data') or not hasattr(rgb.path, 'label'):
                raise ValueError("rgb.path must define both data and label")
    return True
    # Add more checks as needed

if __name__ == "__main__":
    config = load_config('config/config.yaml')
    common = config.common
    memory = config.memory
    check_form(config)