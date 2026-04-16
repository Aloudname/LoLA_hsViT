# -*- coding: utf-8 -*-
# config.py using munch to load .yaml to an easy-accessible object.
import yaml
from munch import Munch


def _ensure_munch(obj):
    if isinstance(obj, Munch):
        return obj
    if isinstance(obj, dict):
        return Munch.fromDict(obj)
    return obj


def _get_nested(cfg: Munch, path):
    cur = cfg
    for key in path:
        if not isinstance(cur, (dict, Munch)) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _set_nested(cfg: Munch, path, value):
    cur = cfg
    for key in path[:-1]:
        nxt = cur.get(key, None)
        if nxt is None or not isinstance(nxt, (dict, Munch)):
            cur[key] = Munch()
        cur[key] = _ensure_munch(cur[key])
        cur = cur[key]
    cur[path[-1]] = value


def _alias_from_nested(section: Munch, flat_key: str, nested_path):
    if flat_key not in section:
        value = _get_nested(section, nested_path)
        if value is not None:
            section[flat_key] = value


def _alias_to_nested(section: Munch, flat_key: str, nested_path):
    if flat_key in section and _get_nested(section, nested_path) is None:
        _set_nested(section, nested_path, section[flat_key])


def _normalize_common_split(config: Munch):
    """Backfill legacy flat keys from nested common/split sections (and vice versa)."""
    config.common = _ensure_munch(config.common)
    config.split = _ensure_munch(config.split)

    common = config.common
    split = config.split

    common_map = [
        ('dataset_name', ('dataset', 'name')),
        ('lr', ('optim', 'lr')),
        ('weight_decay', ('optim', 'weight_decay')),
        ('grad_clip', ('optim', 'grad_clip')),
        ('gradient_accumulation_steps', ('optim', 'gradient_accumulation_steps')),
        ('warmup_epochs', ('optim', 'warmup_epochs')),
        ('scheduler', ('optim', 'scheduler')),
        ('focal_gamma', ('loss', 'focal_gamma')),
        ('loss_dice_weight', ('loss', 'dice_weight')),
        ('loss_boundary_weight', ('loss', 'boundary_weight')),
        ('loss_boundary_dilation', ('loss', 'boundary_dilation')),
        ('loss_dice_absent_prior', ('loss', 'dice_absent_prior')),
        ('label_smoothing', ('loss', 'label_smoothing')),
        ('patience', ('train', 'patience')),
        ('use_amp', ('train', 'use_amp')),
        ('eval_interval', ('train', 'eval_interval')),
        ('max_train_batches_per_epoch', ('train', 'max_train_batches_per_epoch')),
        ('vis_samples', ('eval', 'vis_samples')),
        ('early_stop_metric', ('eval', 'early_stop_metric')),
        ('exclude_class0_in_metrics', ('eval', 'exclude_class0_in_metrics')),
        ('eval_use_batch_cap', ('eval', 'use_batch_cap')),
        ('max_val_batches_per_epoch', ('eval', 'max_val_batches_per_epoch')),
        ('fixed_eval_cap', ('eval', 'fixed_batch_cap')),
        ('eval_sample_seed', ('eval', 'sample_seed')),
        ('eval_boundary_band_dilation', ('eval', 'boundary_band_dilation')),
        ('save_topk_models', ('checkpoint', 'save_topk_models')),
        ('class_weight_max_sample_patches', ('class_weight', 'max_sample_patches')),
        ('class_weight_max_sample_pixels', ('class_weight', 'max_sample_pixels')),
        ('class_weight_sample_seed', ('class_weight', 'sample_seed')),
        ('cv_folds', ('cv', 'folds')),
        ('ema_decay', ('ema', 'decay')),
        ('ema_skip_warmup', ('ema', 'skip_warmup')),
        ('ema_warmup_decay', ('ema', 'warmup_decay')),
        ('eval_fg_gate_threshold', ('eval', 'fg_gate_threshold')),
        ('early_stop_weight_miou', ('early_stop_weights', 'miou')),
        ('early_stop_weight_fg_dice', ('early_stop_weights', 'fg_dice')),
        ('early_stop_weight_bg_iou', ('early_stop_weights', 'bg_iou')),
    ]

    split_map = [
        ('split_seed', ('seed',)),
        ('train_ratio', ('ratio', 'train')),
        ('val_ratio', ('ratio', 'val')),
        ('test_ratio', ('ratio', 'test')),
        ('val_rate', ('ratio', 'legacy', 'val_rate')),
        ('test_rate', ('ratio', 'legacy', 'test_rate')),
        ('n_folds', ('cv', 'n_folds')),
        ('patch_size', ('patch', 'size')),
        ('batch_size', ('batch', 'train_batch_size')),
        ('eval_batch_size', ('batch', 'eval_batch_size')),
        ('samples_per_step', ('batch', 'samples_per_step')),
        ('target_patches_per_step', ('batch', 'target_patches_per_step')),
        ('max_patches_per_sample_chunk', ('batch', 'max_patches_per_sample_chunk')),
        ('sampler_mode', ('sampler', 'mode')),
        ('sampler_mix_fg', ('sampler', 'mix', 'fg')),
        ('sampler_mix_boundary', ('sampler', 'mix', 'boundary')),
        ('sampler_target_mix_fg', ('sampler', 'target_mix', 'fg')),
        ('sampler_target_mix_boundary', ('sampler', 'target_mix', 'boundary')),
        ('sampler_adapt_momentum', ('sampler', 'adaptive', 'momentum')),
        ('sampler_diversity_strength', ('sampler', 'adaptive', 'diversity_strength')),
        ('sampler_fg_ratio_min', ('sampler', 'fg_constraints', 'ratio_min')),
        ('sampler_fg_ratio_max', ('sampler', 'fg_constraints', 'ratio_max')),
        ('sampler_fg_inverse_pow', ('sampler', 'fg_constraints', 'inverse_pow')),
        ('sampler_fg_min_per_class', ('sampler', 'fg_constraints', 'min_per_class')),
        ('min_fg_centers_per_patient', ('sampler', 'fg_constraints', 'min_centers_per_patient')),
        ('min_boundary_centers_per_patient', ('sampler', 'boundary', 'min_centers_per_patient')),
        ('boundary_sampling_dilation', ('sampler', 'boundary', 'sampling_dilation')),
        ('boundary_include_fg_edges', ('sampler', 'boundary', 'include_fg_edges')),
        ('boundary_pair', ('sampler', 'boundary', 'pair')),
    ]

    for flat_key, path in common_map:
        _alias_from_nested(common, flat_key, path)
        _alias_to_nested(common, flat_key, path)

    for flat_key, path in split_map:
        _alias_from_nested(split, flat_key, path)
        _alias_to_nested(split, flat_key, path)


def load_config(yaml_file='config/config.yaml'):
    with open(yaml_file, 'r', encoding='utf-8') as f:
        config = Munch.fromDict(yaml.safe_load(f))

    _normalize_common_split(config)
    
    _ = check_form(config)
    if _:
        return config
    else:
        raise ValueError(f"{yaml_file}.yaml format check failed!")

def check_form(config: Munch = None) -> bool:
    # Add any necessary checks for the config here
    for key in config.keys():
        if key not in ['augment', 'common', 'lora', 'split', 'preprocess', 'memory', 'path', 'clsf']:
            raise ValueError(f"Unexpected key '{key}' found in config")
        
    for key in config.keys():
        if getattr(config, key) is None:
            raise ValueError(f"Config value for key '{key}' cannot be None")
        for subkey in getattr(config, key).keys():
            if getattr(config, key)[subkey] is None:
                raise ValueError(f"Config value for key '{key}.{subkey}' cannot be None")
    
    patch_size = int(_get_nested(config.split, ('patch', 'size')) if _get_nested(config.split, ('patch', 'size')) is not None else config.split.patch_size)
    if patch_size % 2 == 0:
        raise ValueError(f"patch_size must be odd, got {patch_size}")
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
    tr = _get_nested(config.split, ('ratio', 'train'))
    vr = _get_nested(config.split, ('ratio', 'val'))
    ter = _get_nested(config.split, ('ratio', 'test'))
    if tr is None:
        tr = getattr(config.split, 'train_ratio', None)
    if vr is None:
        vr = getattr(config.split, 'val_ratio', None)
    if ter is None:
        ter = getattr(config.split, 'test_ratio', None)
    if tr is not None and vr is not None and ter is not None:
        tr = float(tr)
        vr = float(vr)
        ter = float(ter)
        if tr <= 0 or vr < 0 or ter < 0:
            raise ValueError("split train/val/test ratios must satisfy train>0 and val/test>=0")
        if tr + vr + ter <= 0:
            raise ValueError("split train/val/test ratio sum must be > 0")

    sps = _get_nested(config.split, ('batch', 'samples_per_step'))
    if sps is None and hasattr(config.split, 'samples_per_step'):
        sps = config.split.samples_per_step
    if sps is not None:
        sps = int(sps)
        if sps <= 0:
            raise ValueError("split.samples_per_step must be > 0")

    mpsc = _get_nested(config.split, ('batch', 'max_patches_per_sample_chunk'))
    if mpsc is None and hasattr(config.split, 'max_patches_per_sample_chunk'):
        mpsc = config.split.max_patches_per_sample_chunk
    if mpsc is not None:
        mpsc = int(mpsc)
        if mpsc < 0:
            raise ValueError("split.max_patches_per_sample_chunk must be >= 0")

    tpps = _get_nested(config.split, ('batch', 'target_patches_per_step'))
    if tpps is None and hasattr(config.split, 'target_patches_per_step'):
        tpps = config.split.target_patches_per_step
    if tpps is not None:
        tpps = int(tpps)
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

    metric = _get_nested(config.common, ('eval', 'early_stop_metric'))
    if metric is None and hasattr(config.common, 'early_stop_metric'):
        metric = config.common.early_stop_metric
    if metric is not None:
        metric = str(metric).lower()
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