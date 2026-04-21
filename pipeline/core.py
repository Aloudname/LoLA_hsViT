from __future__ import annotations

# end-to-end pipeline orchestration for refactored experiments.
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from munch import Munch

from pipeline.monitor import tprint
from pipeline.visualize import Visualizer
from model import HSIAdapter, RGBViT, UNet
from pipeline.dataset import build_dataloaders
from pipeline.trainer import Trainer, TrainerResult
from pipeline.analyzer import Analyzer, MetricsBundle

@dataclass
class PipelineResult:
    """final summary for one experiment run."""

    model_key: str
    output_dir: str
    best_epoch: int
    best_eval_dice: float
    test_summary: Dict[str, float]
    metrics_json: str
    onnx_path: Optional[str]
    onnx_note_path: Optional[str]


class Pipeline:
    """main execution bus for data -> model -> trainer -> analyzer -> visualize."""

    def __init__(self, config: Mapping[str, Any], model_key: str, experiment_name: Optional[str] = None) -> None:
        self.config = Munch.fromDict(dict(config))
        self.model_key = str(model_key)

        self._seed_everything(int(self.config.runtime.seed))

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = experiment_name or f"{self.model_key}_{stamp}"
        self.output_dir = Path(self.config.path.output_dir) / run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.analyzer = Analyzer(
            class_names=list(self.config.data.class_names),
            ignore_index=None,
            small_target_class=1,
        )
        self.visualizer = Visualizer(str(self.output_dir), class_names=list(self.config.data.class_names))

    def run(self) -> PipelineResult:
        """execute full training/eval/test workflow."""
        tprint(f"pipeline start: model_key={self.model_key} output_dir={self.output_dir}")

        modality = "rgb" if self.model_key == "rgb" else "hsi"

        stage_t0 = time.perf_counter()
        tprint(f"stage[data]: build dataloaders (modality={modality})")
        loaders, prepared = build_dataloaders(self.config, modality=modality)
        tprint(f"stage[data] done in {time.perf_counter() - stage_t0:.2f}s")

        stage_t0 = time.perf_counter()
        in_channels = self._resolve_input_channels(modality)
        tprint(f"stage[model]: build model (in_channels={in_channels})")
        model = self._build_model(in_channels=in_channels)
        tprint(f"stage[model] done in {time.perf_counter() - stage_t0:.2f}s")

        stage_t0 = time.perf_counter()
        tprint("stage[trainer]: init trainer")
        trainer = Trainer(model=model, config=self.config, output_dir=str(self.output_dir))
        tprint(f"stage[trainer] done in {time.perf_counter() - stage_t0:.2f}s")

        stage_t0 = time.perf_counter()
        tprint("stage[train]: fit train/eval")
        trainer_result = trainer.fit(
            train_loader=loaders["train"],
            eval_loader=loaders["eval"],
            epochs=int(self.config.train.epochs),
        )
        tprint(
            f"stage[train] done in {time.perf_counter() - stage_t0:.2f}s "
            f"(best_epoch={trainer_result.best_epoch}, best_eval_dice={trainer_result.best_metric:.4f})"
        )

        stage_t0 = time.perf_counter()
        tprint("stage[test]: predict on test loader")
        pred_pack = trainer.predict(
            dataloader=loaders["test"],
            keep_images=int(self.config.visualization.num_seg_examples),
            keep_features=int(self.config.visualization.tsne_max_points),
        )
        tprint(f"stage[test] done in {time.perf_counter() - stage_t0:.2f}s")

        stage_t0 = time.perf_counter()
        tprint("stage[metrics]: compute analyzer metrics")
        metrics_bundle = self.analyzer.compute_metrics(
            preds=pred_pack["pred_masks"],
            targets=pred_pack["gt_masks"],
            probs=pred_pack["prob_maps"],
        )

        tprint(self.analyzer.summarize(metrics_bundle))
        tprint(f"stage[metrics] done in {time.perf_counter() - stage_t0:.2f}s")

        stage_t0 = time.perf_counter()
        metrics_path = self._save_metrics(metrics_bundle, trainer_result)
        tprint(f"stage[metrics-save] done in {time.perf_counter() - stage_t0:.2f}s")

        stage_t0 = time.perf_counter()
        tprint("stage[viz]: generate visualizations")
        self._run_visualizations(metrics_bundle, trainer_result, pred_pack, prepared.stats, modality)
        tprint(f"stage[viz] done in {time.perf_counter() - stage_t0:.2f}s")

        stage_t0 = time.perf_counter()
        tprint("stage[export]: export onnx")
        onnx_path = self._export_onnx(trainer, in_channels=in_channels)
        tprint(f"stage[export] done in {time.perf_counter() - stage_t0:.2f}s")
        onnx_note_path = str(Path(onnx_path).with_name("best_model_info.txt")) if onnx_path else None

        tprint(f"pipeline done: model_key={self.model_key} output_dir={self.output_dir}")

        return PipelineResult(
            model_key=self.model_key,
            output_dir=str(self.output_dir),
            best_epoch=int(trainer_result.best_epoch),
            best_eval_dice=float(trainer_result.best_metric),
            test_summary={k: float(v) for k, v in metrics_bundle.summary.items()},
            metrics_json=str(metrics_path),
            onnx_path=onnx_path,
            onnx_note_path=onnx_note_path,
        )

    def _build_model(self, in_channels: int) -> torch.nn.Module:
        """build model instance from model_key."""
        common_cfg = {
            "in_channels": in_channels,
            "num_classes": int(self.config.data.num_classes),
            "spectral_dim": int(self.config.model.spectral_dim),
            "embed_dim": int(self.config.model.embed_dim),
            "depth": int(self.config.model.depth),
            "num_heads": int(self.config.model.num_heads),
            "mlp_ratio": float(self.config.model.mlp_ratio),
            "decoder_dim": int(self.config.model.decoder_dim),
            "dropout": float(self.config.model.dropout),
            "freeze_backbone": bool(self.config.model.freeze_backbone),
            "use_pretrained": bool(
                getattr(self.config.model, "use_pretrained", getattr(self.config.model, "pretrained_backbone", False))
            ),
            "backbone_name": str(getattr(self.config.model, "backbone_name", "vit_small_patch16_224")),
            "pretrained_weights": bool(
                getattr(self.config.model, "pretrained_weights", getattr(self.config.model, "backbone_pretrained", True))
            ),
            "pretrained_cache_dir": str(getattr(self.config.path, "pretrained_cache_dir", "")),
            "unfreeze_last_n": int(getattr(self.config.model, "unfreeze_last_n", 0)),
        }

        tprint(
            "model config: "
            f"family={self.config.model.family} in_channels={in_channels} "
            f"num_classes={int(self.config.data.num_classes)} "
            f"embed_dim={int(self.config.model.embed_dim)} depth={int(self.config.model.depth)} "
            f"heads={int(self.config.model.num_heads)} "
            f"freeze_backbone={bool(self.config.model.freeze_backbone)} "
            f"use_pretrained={common_cfg['use_pretrained']} pretrained_weights={common_cfg['pretrained_weights']}"
        )

        if self.model_key.startswith("hsi"):
            model = HSIAdapter(common_cfg)
        elif self.model_key == "rgb":
            model = RGBViT(common_cfg)
        elif self.model_key == "unet":
            unet_cfg = {
                "in_channels": in_channels,
                "num_classes": int(self.config.data.num_classes),
                "base_channels": int(max(16, self.config.model.embed_dim // 4)),
            }
            model = UNet(unet_cfg)
        else:
            raise ValueError(f"unknown model_key: {self.model_key}")

        total_params = int(sum(p.numel() for p in model.parameters()))
        trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        tprint(f"model ready: trainable_params={trainable_params} total_params={total_params}")
        return model

    def _resolve_input_channels(self, modality: str) -> int:
        """resolve input channels according to modality and preprocessing mode."""
        if modality == "rgb":
            tprint("input channels resolved: modality=rgb channels=3")
            return 3

        preprocess_mode = str(self.config.data.preprocess.mode).lower()
        if preprocess_mode == "none":
            channels = int(self.config.data.hsi_bands)
        else:
            channels = int(self.config.data.preprocess.output_dim)

        tprint(
            f"input channels resolved: modality=hsi preprocess_mode={preprocess_mode} channels={channels}"
        )
        return channels

    def _save_metrics(self, bundle: MetricsBundle, trainer_result: TrainerResult) -> Path:
        """save metrics and training history json files."""
        metrics_dir = self.output_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        cm = bundle.confusion_matrix.tolist()
        payload = {
            "summary": bundle.summary,
            "per_class": bundle.per_class,
            "roc_auc": bundle.roc_auc,
            "roc_curves": {
                name: {
                    "fpr": curves["fpr"].tolist(),
                    "tpr": curves["tpr"].tolist(),
                }
                for name, curves in bundle.roc_curves.items()
            },
            "confusion_matrix": cm,
            "best_epoch": int(trainer_result.best_epoch),
            "best_eval_dice": float(trainer_result.best_metric),
            "history": trainer_result.history,
        }

        metrics_path = metrics_dir / "result_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        tprint(f"metrics saved: {metrics_path}")

        return metrics_path

    def _run_visualizations(
        self,
        metrics_bundle: MetricsBundle,
        trainer_result: TrainerResult,
        pred_pack: Mapping[str, Any],
        split_stats: Mapping[str, Any],
        modality: str,
    ) -> None:
        """generate all required plots."""
        tprint("visualization: training curves")
        self.visualizer.plot_training_curves(trainer_result.history)
        tprint("visualization: prf/confusion/roc/distribution")
        self.visualizer.plot_prf(metrics_bundle)
        self.visualizer.plot_confusion_matrix(metrics_bundle)
        self.visualizer.plot_roc(metrics_bundle)
        self.visualizer.plot_distribution(split_stats)

        tprint("visualization: segmentation samples")
        self.visualizer.show_segmentation(
            images=pred_pack["image_samples"],
            preds=pred_pack["pred_masks"],
            gts=pred_pack["gt_masks"],
            max_items=int(self.config.visualization.num_seg_examples),
            prefix="test",
        )

        features = np.asarray(pred_pack.get("features", np.empty((0, 0), dtype=np.float32)))
        labels = np.asarray(pred_pack.get("feature_labels", np.empty((0,), dtype=np.int64)))
        if features.size > 0 and labels.size > 0:
            tprint(f"visualization: tsne points={features.shape[0]}")
            self.visualizer.plot_tsne(features=features, labels=labels, title=f"{self.model_key} feature tsne")

        if modality == "hsi":
            spectra = self._collect_spectral_curves(pred_pack, max_points=int(self.config.visualization.spectral_max_points_per_class))
            tprint(f"visualization: spectral curves classes={len(spectra)}")
            self.visualizer.plot_spectral(spectra)

        attention = pred_pack.get("attention_map")
        tprint(f"visualization: attention map available={attention is not None}")
        self.visualizer.plot_attention_map(attention)

    def _collect_spectral_curves(self, pred_pack: Mapping[str, Any], max_points: int) -> Dict[str, Any]:
        """estimate spectral mean/std from retained image samples and gt masks."""
        images = pred_pack.get("image_samples", [])
        gts = pred_pack.get("gt_masks", [])

        if not images or not gts:
            return {}

        num_classes = int(self.config.data.num_classes)
        class_names = list(self.config.data.class_names)
        by_class: Dict[int, List[np.ndarray]] = {i: [] for i in range(num_classes)}

        for image, gt in zip(images, gts):
            img = np.asarray(image)
            mask = np.asarray(gt)
            c, h, w = img.shape
            flat = img.reshape(c, h * w).T
            labels = mask.reshape(-1)

            for cls_idx in range(num_classes):
                idx = np.where(labels == cls_idx)[0]
                if idx.size == 0:
                    continue
                if idx.size > max_points:
                    idx = np.random.choice(idx, size=max_points, replace=False)
                by_class[cls_idx].append(flat[idx])

        out: Dict[str, Any] = {}
        for cls_idx, chunks in by_class.items():
            if not chunks:
                continue
            arr = np.concatenate(chunks, axis=0)
            if arr.shape[0] > max_points:
                sel = np.random.choice(arr.shape[0], size=max_points, replace=False)
                arr = arr[sel]
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            out[class_names[cls_idx]] = (mean, std)

        return out

    def _export_onnx(self, trainer: Trainer, in_channels: int) -> Optional[str]:
        """export best model to onnx."""
        if not bool(self.config.export.export_onnx):
            tprint("onnx export skipped: export.export_onnx=false")
            return None

        model = trainer.ema.ema_model if trainer.ema is not None else trainer.model
        model.eval()

        onnx_dir = self.output_dir / "models"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = onnx_dir / "best_model.onnx"
        note_path = onnx_dir / "best_model_info.txt"

        tprint(
            "onnx export start: "
            f"path={onnx_path} opset={int(self.config.export.onnx_opset)} in_channels={in_channels}"
        )

        patch_size = int(self.config.data.patch_size)
        dummy = torch.randn(1, in_channels, patch_size, patch_size, dtype=torch.float32)

        prev_fastpath: Optional[bool] = None
        try:
            if hasattr(torch.backends, "mha") and hasattr(torch.backends.mha, "get_fastpath_enabled"):
                prev_fastpath = bool(torch.backends.mha.get_fastpath_enabled())
                torch.backends.mha.set_fastpath_enabled(False)

            model_cpu = model.cpu()
            with torch.no_grad():
                output_sample = model_cpu(dummy)
                torch.onnx.export(
                    model_cpu,
                    dummy,
                    str(onnx_path),
                    input_names=["input"],
                    output_names=["logits"],
                    opset_version=int(self.config.export.onnx_opset),
                    dynamic_axes={
                        "input": {0: "batch"},
                        "logits": {0: "batch"},
                    },
                )

            self._write_onnx_note(
                note_path=note_path,
                onnx_path=onnx_path,
                model=model_cpu,
                input_sample=dummy,
                output_sample=output_sample,
            )
            tprint(f"onnx info note saved: {note_path}")
            return str(onnx_path)
        except Exception as exc:
            tprint(f"onnx export failed: {exc}")
            return None
        finally:
            if prev_fastpath is not None and hasattr(torch.backends, "mha") and hasattr(torch.backends.mha, "set_fastpath_enabled"):
                torch.backends.mha.set_fastpath_enabled(prev_fastpath)

    def _write_onnx_note(
        self,
        note_path: Path,
        onnx_path: Path,
        model: torch.nn.Module,
        input_sample: torch.Tensor,
        output_sample: torch.Tensor,
    ) -> None:
        """Write deployment-friendly model and I/O notes beside ONNX file."""
        input_shape = [int(v) for v in input_sample.shape]
        output_shape = [int(v) for v in output_sample.shape]
        patch_size = int(self.config.data.patch_size)
        stride = int(getattr(self.config.data, "stride", patch_size))
        overlap_ratio = max(0.0, 1.0 - (float(stride) / float(max(1, patch_size))))

        input_c = input_shape[1] if len(input_shape) > 1 else 1
        input_h = input_shape[2] if len(input_shape) > 2 else patch_size
        input_w = input_shape[3] if len(input_shape) > 3 else patch_size
        class_names = [str(name) for name in list(self.config.data.class_names)]
        class_map_lines = [f"  - {idx}: {name}" for idx, name in enumerate(class_names)]

        preprocess_mode = str(getattr(self.config.data.preprocess, "mode", "none"))
        preprocess_out_dim = int(getattr(self.config.data.preprocess, "output_dim", input_shape[1]))
        model_family = str(getattr(self.config.model, "family", "unknown"))
        backbone_name = str(getattr(self.config.model, "backbone_name", "n/a"))

        total_params = int(sum(p.numel() for p in model.parameters()))
        trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

        preprocess_lines = [
            "  - Convert input to float32.",
            "  - Ensure tensor layout is [N, C, H, W].",
        ]
        if self.model_key == "rgb":
            preprocess_lines.extend(
                [
                    "  - Read image as RGB (if using OpenCV, convert BGR -> RGB first).",
                    "  - Scale pixel values to [0, 1] using /255.0.",
                    f"  - Expected channels C={input_shape[1]}.",
                ]
            )
        else:
            preprocess_lines.extend(
                [
                    "  - Load hyperspectral cube with float32 values.",
                    f"  - Reducer mode from training config: {preprocess_mode}.",
                ]
            )
            if preprocess_mode != "none":
                preprocess_lines.append(
                    f"  - Reducer output channels from training config: {preprocess_out_dim}."
                )
            preprocess_lines.append(f"  - Expected channels C={input_shape[1]}.")

        sliding_window_lines = [
            f"  - window_size: {patch_size} x {patch_size}",
            f"  - stride: {stride}",
            f"  - overlap_ratio: ~{overlap_ratio:.2f}",
            "  - For large images, tile windows with tail coverage so the right/bottom borders are always covered.",
            "  - Fuse overlapping windows by averaging logits with a count map, then run argmax.",
            "  - If input is smaller than window_size, pad first and crop back to original size after fusion.",
        ]

        lines = [
            "Model Deployment Note",
            "=====================",
            f"model_key: {self.model_key}",
            f"model_family: {model_family}",
            f"backbone_name: {backbone_name}",
            f"onnx_file: {onnx_path.name}",
            f"onnx_path: {onnx_path}",
            f"export_time: {datetime.now().isoformat(timespec='seconds')}",
            f"parameter_count_total: {total_params}",
            f"parameter_count_trainable: {trainable_params}",
            "",
            "Input",
            "-----",
            "name: input",
            f"dtype: {str(input_sample.dtype).replace('torch.', '')}",
            f"shape_example: {input_shape}",
            "dynamic_axes: batch axis 0",
            "layout: [N, C, H, W]",
            f"recommended_patch_size: {patch_size} x {patch_size}",
            "",
            "Output",
            "------",
            "name: logits",
            f"dtype: {str(output_sample.dtype).replace('torch.', '')}",
            f"shape_example: {output_shape}",
            "dynamic_axes: batch axis 0",
            "layout: [N, num_classes, H, W]",
            "postprocess: argmax(logits, axis=1) -> predicted label map [N, H, W]",
            "",
            "Class Mapping",
            "-------------",
            *class_map_lines,
            "",
            "Preprocess Contract",
            "-------------------",
            *preprocess_lines,
            "",
            "Sliding Window Inference Contract",
            "---------------------------------",
            *sliding_window_lines,
            "",
            "ONNX Runtime Example (Python)",
            "-----------------------------",
            "import numpy as np",
            "import onnxruntime as ort",
            "",
            f"sess = ort.InferenceSession(r\"{onnx_path}\", providers=['CPUExecutionProvider'])",
            f"x = np.random.randn(1, {input_c}, {input_h}, {input_w}).astype(np.float32)",
            "logits = sess.run(['logits'], {'input': x})[0]",
            "pred = logits.argmax(axis=1).astype(np.int64)",
            "",
            "Sliding Window Pseudocode (NumPy)",
            "-------------------------------",
            "# logits_acc: [num_classes, H, W], count_acc: [1, H, W]",
            "# for each window (y, x):",
            "#   patch = image[:, y:y+patch_h, x:x+patch_w]",
            "#   logits_patch = sess.run(['logits'], {'input': patch[None].astype(np.float32)})[0][0]",
            "#   logits_acc[:, y:y+patch_h, x:x+patch_w] += logits_patch",
            "#   count_acc[:, y:y+patch_h, x:x+patch_w] += 1.0",
            "# fused_logits = logits_acc / np.maximum(count_acc, 1e-6)",
            "# pred = np.argmax(fused_logits, axis=0).astype(np.int64)",
            "",
            "Model Structure",
            "---------------",
            str(model),
            "",
            "Config Snapshot",
            "---------------",
            f"num_classes: {int(self.config.data.num_classes)}",
            f"class_names: {class_names}",
            f"preprocess_mode: {preprocess_mode}",
            f"patch_size: {patch_size}",
            f"stride: {stride}",
            f"onnx_opset: {int(self.config.export.onnx_opset)}",
            f"device_for_export: cpu",
        ]

        note_path.parent.mkdir(parents=True, exist_ok=True)
        with note_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    @staticmethod
    def _seed_everything(seed: int) -> None:
        """set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
