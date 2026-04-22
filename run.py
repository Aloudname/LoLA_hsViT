#!/usr/bin/env python3
from __future__ import annotations

# unified entry point for refactored hsi segmentation experiments.
from munch import Munch
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Mapping
from contextlib import contextmanager, redirect_stderr, redirect_stdout

import io, sys, copy, json, argparse

from pipeline import Pipeline, generate_synthetic_dataset
from config import load_config, _to_munch, merge_args, tprint


VALID_MODELS = [
    "hsi",
    "rgb",
    "unet",
    "light"
]


class _Tee(io.TextIOBase):
    """Mirror writes to multiple text streams."""

    def __init__(self, *streams: io.TextIOBase) -> None:
        self._streams = streams

    def write(self, s: str) -> int:
        for stream in self._streams:
            stream.write(s)
        return len(s)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


@contextmanager
def _capture_console(log_path: Path):
    """Persist stdout/stderr to log file while keeping terminal output."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_f:
        tee_out = _Tee(sys.__stdout__, log_f)
        tee_err = _Tee(sys.__stderr__, log_f)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            yield


def parse_args() -> argparse.Namespace:
    """parse command-line arguments."""
    parser = argparse.ArgumentParser(description="refactored hsi segmentation runner")

    parser.add_argument("--tag", '-t', type=str, default=None, help="for dim reduction settings. `none`, `supervised_pca`, `lda_pca`, `kernel_lda`.")
    parser.add_argument("--config", '-c', type=str, default="config/config.yaml", help="path to yaml config")
    parser.add_argument("--models", '-m', nargs="+", default=None, help=f"models to run: {', '.join(VALID_MODELS)}")
    parser.add_argument("--epochs", '-e', type=int, default=None, help="override train.epochs")
    parser.add_argument("--output-dir", '-o', type=str, default=None, help="override path.output_dir")
    parser.add_argument("--seed", '-s', type=int, default=None, help="override runtime and split seed")

    parser.add_argument("--set", action="append", default=[], help="extra override as key=value, supports dotted key")

    # for short debugging
    parser.add_argument("--generate-synthetic", '-gg', action="store_true", help="generate synthetic fisher/rgb dataset before run")
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
        return float(text) if "." in text else int(text)
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


def _apply_model_profile(config: Munch, model_key: str, tag_key: str) -> Dict[str, Any]:
    """apply experiment-specific profile from design document."""
    base = config.toDict() if hasattr(config, "toDict") else dict(config)
    cfg = copy.deepcopy(base)

    if model_key == "hsi":
        cfg["model"]["family"] = "hsi_adapter"
        if tag_key == "kernel_lda":
            cfg["data"]["preprocess"]["mode"] = "kernel_lda"
        elif tag_key == "lda_pca":
            cfg["data"]["preprocess"]["mode"] = "lda_pca"

    elif model_key == "rgb":
        cfg["model"]["family"] = "rgb_vit"
        cfg["data"]["preprocess"]["mode"] = "none"

    elif model_key == "unet":
        cfg["model"]["family"] = "unet"
        cfg["data"]["preprocess"]["mode"] = "none"
        
    elif model_key == "light":
        cfg["model"]["family"] = "light_adapter"

    else:
        raise ValueError(f"unknown model key: {model_key}")

    return cfg


def main() -> None:
    """run selected experiments sequentially."""
    args = parse_args()
    tprint(f"load config from: {args.config}")
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
    if override_map:
        tprint(f"cli overrides: {override_map}")
    config = merge_args(config, override_map)

    tprint(
        "runtime config:\n "
        f"\tseed={int(config.runtime.seed)}, device={str(config.runtime.device)}\n"
        f"\tepochs={int(config.train.epochs)}, train_bs={int(config.train.batch_size)}, eval_bs={int(config.train.eval_batch_size)}\n"
        f"\tprogress_bar={bool(getattr(config.runtime, 'progress_bar', True))}\n"
    )

    output_root = Path(config.path.output_dir)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_path = output_root / "logs" / f"run_{run_stamp}.log"

    with _capture_console(run_log_path):
        _run(run_log_path, output_root, args, config)


def _run(run_log_path, output_root, args, config):
    tprint(f"cli log file: {run_log_path}\n"
           f"\toutput root: {output_root}\n")

    if args.generate_synthetic:
        _fake_data(config, args)
        
    # none: read config.
    models = args.models or list(config.experiments.default_models)
    tag_key = args.tag or str(config.data.preprocess.mode)
    for m in models:
        if m not in VALID_MODELS:
            raise ValueError(f"invalid model '{m}', valid: {', '.join(VALID_MODELS)}")

    summary: List[Dict[str, Any]] = []
    tprint(f"run queue: {models}")

    for idx, model_key in enumerate(models, start=1):
        tprint(f"[{idx}/{len(models)}] start model: {model_key}")

        run_cfg = _apply_model_profile(config, model_key, tag_key)
        run_cfg = _to_munch(run_cfg)

        print(
            f"[{idx}/{len(models)}] profile:\n"
            f"\tfamily={run_cfg.model.family}\n"
            f"\tpreprocess={run_cfg.data.preprocess.mode}\n"
            f"\tuse_pretrained={bool(run_cfg.model.get('use_pretrained', False))}\n"
            f"\tpretrained_weights={bool(run_cfg.model.get('pretrained_weights', True))}\n"
        )
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
                "onnx_note_path": result.onnx_note_path,
            }
        )

        tprint(
            f"[{idx}/{len(models)}] done model: {model_key}\n"
            f"  best_eval_dice={result.best_eval_dice:.4f}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "run_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    tprint(f"all done, summary saved to: {summary_path}")


def _fake_data(config, args):
    synth_cfg = getattr(config.data, "synthetic", None)
    domain_shift_strength = float(getattr(synth_cfg, "domain_shift_strength", 0.12))
    noise_std = float(getattr(synth_cfg, "noise_std", 0.03))
    boundary_mix_sigma = float(getattr(synth_cfg, "boundary_mix_sigma", 1.2))
    label_noise_prob = float(getattr(synth_cfg, "label_noise_prob", 0.01))

    tprint(
        "generate synthetic dataset: "
        f"  root={args.synthetic_root}, subjects={int(args.synthetic_subjects)} "
        f"  samples_per_subject={int(args.synthetic_samples_per_subject)} "
        f"  image_size={int(args.synthetic_image_size)} "
        f"  domain_shift_strength={domain_shift_strength}, noise_std={noise_std} "
        f"  boundary_mix_sigma={boundary_mix_sigma}, label_noise_prob={label_noise_prob}\n"
    )
    synth_paths = generate_synthetic_dataset(
        root_dir=args.synthetic_root,
        num_subjects=int(args.synthetic_subjects),
        samples_per_subject=int(args.synthetic_samples_per_subject),
        image_size=int(args.synthetic_image_size),
        num_bands=int(config.data.hsi_bands),
        seed=int(config.runtime.seed),
        domain_shift_strength=domain_shift_strength,
        noise_std=noise_std,
        boundary_mix_sigma=boundary_mix_sigma,
        label_noise_prob=label_noise_prob,
    )
    config.path.hsi_dir = synth_paths["hsi_dir"]
    config.path.label_dir = synth_paths["label_dir"]
    config.path.rgb_dir = synth_paths["rgb_dir"]


if __name__ == "__main__":
    main()
