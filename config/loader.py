from __future__ import annotations

# config loader utilities for yaml-based experiments.
import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import yaml
from munch import Munch


def tprint(*args, **kwargs) -> None:
    """print logs with [hh:mm:ss] timestamp."""
    print(datetime.now().strftime("[%H:%M:%S]"), *args, **kwargs)


def _to_munch(obj: Any) -> Any:
    """convert nested dict/list objects into munch recursively."""
    if isinstance(obj, Munch):
        return obj
    if isinstance(obj, Mapping):
        return Munch({k: _to_munch(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_munch(v) for v in obj]
    return obj


def _set_nested(container: MutableMapping[str, Any], dotted_key: str, value: Any) -> None:
    """set value in nested mapping with dotted key path."""
    parts = dotted_key.split(".")
    current = container
    for key in parts[:-1]:
        if key not in current or not isinstance(current[key], Mapping):
            current[key] = {}
        current = current[key]
    current[parts[-1]] = value


def _deep_update(dst: MutableMapping[str, Any], src: Mapping[str, Any]) -> None:
    """recursive dict update used by merge_args."""
    for key, value in src.items():
        if isinstance(value, Mapping) and isinstance(dst.get(key), Mapping):
            _deep_update(dst[key], value)
        else:
            dst[key] = value


def load_config(path: str = "config/config.yaml") -> Munch:
    """load yaml config into munch.

    input:
        path(str): yaml file path.
    output:
        munch: config tree with attribute access.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}

    return _to_munch(payload)


def merge_args(config: Mapping[str, Any], cli_args: Mapping[str, Any] | argparse.Namespace) -> Munch:
    """merge command-line args into config tree.

    keys with dot notation are treated as nested paths.
    """
    if isinstance(config, Munch):
        merged: MutableMapping[str, Any] = deepcopy(config.toDict())
    else:
        merged = deepcopy(dict(config))

    if isinstance(cli_args, argparse.Namespace):
        args_map = vars(cli_args)
    else:
        args_map = dict(cli_args)

    for key, value in args_map.items():
        if value is None:
            continue
        if key.startswith("_"):
            continue

        if "." in key:
            _set_nested(merged, key, value)
            continue

        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            _deep_update(merged[key], value)
        else:
            merged[key] = value

    return _to_munch(merged)
