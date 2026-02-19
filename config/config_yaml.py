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
        raise ValueError(f"num must match the length of targets, got {config.clsf.num} and {len(config.clsf.targets)}")
    return True
    # Add more checks as needed

config = load_config()
common = config.common
memory = config.memory
check_form(config)