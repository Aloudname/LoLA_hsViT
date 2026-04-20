#!/usr/bin/env python3
# unified entry point for refactored hsi segmentation experiments.
import copy, json, argparse

from pathlib import Path
from __future__ import annotations
from typing import Any, Dict, List, Mapping
from config import load_config, merge_args, tprint
from pipeline import Pipeline, generate_synthetic_dataset


VALID_MODELS = [
    "hsi_adapter_learnable",
    "hsi_adapter_lda",
    "rgb_vit",
    "unet",
]


def parse_args() -> argparse.Namespace:
    """parse command-line arguments."""
    parser = argparse.ArgumentParser(description="refactored hsi segmentation runner")

    parser.add_argument("--config", '-c', type=str, default="config/config.yaml", help="path to yaml config")
    parser.add_argument("--models", '-m', nargs="+", default=None, help=f"models to run: {', '.join(VALID_MODELS)}")
    parser.add_argument("--epochs", '-e', type=int, default=None, help="override train.epochs")
    parser.add_argument("--output-dir", '-o', type=str, default=None, help="override path.output_dir")
    parser.add_argument("--seed", '-s', type=int, default=None, help="override runtime and split seed")

    parser.add_argument("--set", action="append", default=[], help="extra override as key=value, supports dotted key")

    # for short debugging
    parser.add_argument("--generate-synthetic", action="store_true", help="generate synthetic fisher/rgb dataset before run")
    parser.add_argument("--synthetic-root", type=str, default="./synthetic_data", help="synthetic dataset root")
    parser.add_argument("--synthetic-subjects", type=int, default=12, help="number of synthetic subjects")
    parser.add_argument("--synthetic-samples-per-subject", type=int, default=2, help="samples per synthetic subject")
    parser.add_argument("--synthetic-image-size", type=int, default=128, help="synthetic image size")

    return parser.parse_args()


def _parse_value(raw: str) -> Any:
    """parse scalar from text for --set overrides."""
    text = raw.strip()
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def _parse_set_overrides(items: List[str]) -> Dict[str, Any]:
    """parse --set key=value list into dict."""
    out: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"invalid --set format: {item}, expected key=value")
        key, value = item.split("=", 1)
        out[key.strip()] = _parse_value(value)
    return out


def _apply_model_profile(config: Mapping[str, Any], model_key: str) -> Dict[str, Any]:
    """apply experiment-specific profile from design document."""
    base = config.toDict() if hasattr(config, "toDict") else dict(config)
    cfg = copy.deepcopy(base)

    if model_key == "hsi_adapter_learnable":
        cfg["model"]["family"] = "hsi_adapter"
        cfg["data"]["preprocess"]["mode"] = "none"

    elif model_key == "hsi_adapter_lda":
        cfg["model"]["family"] = "hsi_adapter"
        cfg["data"]["preprocess"]["mode"] = "lda_pca"

    elif model_key == "rgb_vit":
        cfg["model"]["family"] = "rgb_vit"
        cfg["data"]["preprocess"]["mode"] = "none"

    elif model_key == "unet":
        cfg["model"]["family"] = "unet"
        cfg["data"]["preprocess"]["mode"] = "none"

    else:
        raise ValueError(f"unknown model key: {model_key}")

    return cfg


def main() -> None:
    """run selected experiments sequentially."""
    args = parse_args()
    config = load_config(args.config)

    override_map: Dict[str, Any] = {}
    if args.epochs is not None:
        override_map["train.epochs"] = int(args.epochs)
    if args.output_dir is not None:
        override_map["path.output_dir"] = str(args.output_dir)
    if args.seed is not None:
        override_map["runtime.seed"] = int(args.seed)
        override_map["data.split.seed"] = int(args.seed)

    override_map.update(_parse_set_overrides(args.set))
    config = merge_args(config, override_map)

    if args.generate_synthetic:
        synth_paths = generate_synthetic_dataset(
            root_dir=args.synthetic_root,
            num_subjects=int(args.synthetic_subjects),
            samples_per_subject=int(args.synthetic_samples_per_subject),
            image_size=int(args.synthetic_image_size),
            num_bands=int(config.data.hsi_bands),
            seed=int(config.runtime.seed),
        )
        config.path.hsi_dir = synth_paths["hsi_dir"]
        config.path.label_dir = synth_paths["label_dir"]
        config.path.rgb_dir = synth_paths["rgb_dir"]

    models = args.models or list(config.experiments.default_models)
    for m in models:
        if m not in VALID_MODELS:
            raise ValueError(f"invalid model '{m}', valid: {', '.join(VALID_MODELS)}")

    summary: List[Dict[str, Any]] = []
    tprint(f"run queue: {models}")

    for idx, model_key in enumerate(models, start=1):
        tprint(f"[{idx}/{len(models)}] start model: {model_key}")

        run_cfg = _apply_model_profile(config, model_key)
        pipeline = Pipeline(run_cfg, model_key=model_key)
        result = pipeline.run()

        summary.append(
            {
                "model": result.model_key,
                "output_dir": result.output_dir,
                "best_epoch": result.best_epoch,
                "best_eval_dice": result.best_eval_dice,
                "test_summary": result.test_summary,
                "metrics_json": result.metrics_json,
                "onnx_path": result.onnx_path,
            }
        )

        tprint(
            f"[{idx}/{len(models)}] done model: {model_key} | "
            f"best_eval_dice={result.best_eval_dice:.4f}"
        )

    output_root = Path(config.path.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "run_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    tprint(f"all done, summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
