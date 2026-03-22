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
        if key not in ['dataset_name', 'common', 'lora', 'split', 'preprocess', 'memory', 'path', 'key', 'clsf']:
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
    if unknown_policy not in ['error', 'map_to_bg']:
        raise ValueError(
            f"clsf.unknown_label_policy must be 'error' or 'map_to_bg', got {unknown_policy}"
        )

    if hasattr(config.preprocess, 'bg_sample_ratio'):
        bg_ratio = float(config.preprocess.bg_sample_ratio)
        if bg_ratio < 0:
            raise ValueError(f"preprocess.bg_sample_ratio must be >= 0, got {bg_ratio}")

    if hasattr(config.split, 'boundary_pair'):
        pair = config.split.boundary_pair
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(
                f"split.boundary_pair must be a list/tuple of 2 class names, got {pair}"
            )

    if hasattr(config.common, 'early_stop_metric'):
        metric = str(config.common.early_stop_metric).lower()
        if metric not in ['composite', 'fg', 'all']:
            raise ValueError(
                f"common.early_stop_metric must be one of composite/fg/all, got {metric}"
            )

    if hasattr(config.common, 'metric_weight_fg') and hasattr(config.common, 'metric_weight_all'):
        w_fg = float(config.common.metric_weight_fg)
        w_all = float(config.common.metric_weight_all)
        if w_fg < 0 or w_all < 0:
            raise ValueError("common.metric_weight_fg and metric_weight_all must be >= 0")
    return True
    # Add more checks as needed

if __name__ == "__main__":
    config = load_config('config/config.yaml')
    common = config.common
    memory = config.memory
    check_form(config)