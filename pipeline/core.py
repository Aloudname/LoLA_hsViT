# end-to-end pipeline orchestration for refactored experiments.
import json, random, torch, numpy as np

from munch import Munch
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from __future__ import annotations
from typing import Any, Dict, List, Mapping, Optional, Sequence

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
        modality = "rgb" if self.model_key == "rgb_vit" else "hsi"
        loaders, prepared = build_dataloaders(self.config, modality=modality)

        in_channels = self._resolve_input_channels(modality)
        model = self._build_model(in_channels=in_channels)

        trainer = Trainer(model=model, config=self.config, output_dir=str(self.output_dir))
        trainer_result = trainer.fit(
            train_loader=loaders["train"],
            eval_loader=loaders["eval"],
            epochs=int(self.config.train.epochs),
        )

        pred_pack = trainer.predict(
            dataloader=loaders["test"],
            keep_images=int(self.config.visualization.num_seg_examples),
            keep_features=int(self.config.visualization.tsne_max_points),
        )

        metrics_bundle = self.analyzer.compute_metrics(
            preds=pred_pack["pred_masks"],
            targets=pred_pack["gt_masks"],
            probs=pred_pack["prob_maps"],
        )

        tprint(self.analyzer.summarize(metrics_bundle))

        metrics_path = self._save_metrics(metrics_bundle, trainer_result)
        self._run_visualizations(metrics_bundle, trainer_result, pred_pack, prepared.stats, modality)

        onnx_path = self._export_onnx(trainer, in_channels=in_channels)

        return PipelineResult(
            model_key=self.model_key,
            output_dir=str(self.output_dir),
            best_epoch=int(trainer_result.best_epoch),
            best_eval_dice=float(trainer_result.best_metric),
            test_summary={k: float(v) for k, v in metrics_bundle.summary.items()},
            metrics_json=str(metrics_path),
            onnx_path=onnx_path,
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
        }

        if self.model_key.startswith("hsi_adapter"):
            return HSIAdapter(common_cfg)
        if self.model_key == "rgb_vit":
            return RGBViT(common_cfg)
        if self.model_key == "unet":
            unet_cfg = {
                "in_channels": in_channels,
                "num_classes": int(self.config.data.num_classes),
                "base_channels": int(max(16, self.config.model.embed_dim // 4)),
            }
            return UNet(unet_cfg)

        raise ValueError(f"unknown model_key: {self.model_key}")

    def _resolve_input_channels(self, modality: str) -> int:
        """resolve input channels according to modality and preprocessing mode."""
        if modality == "rgb":
            return 3

        preprocess_mode = str(self.config.data.preprocess.mode).lower()
        if preprocess_mode == "none":
            return int(self.config.data.hsi_bands)
        return int(self.config.data.preprocess.output_dim)

    def _save_metrics(self, bundle: MetricsBundle, trainer_result: TrainerResult) -> Path:
        """save metrics and training history json files."""
        metrics_dir = self.output_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        cm = bundle.confusion_matrix.tolist()
        payload = {
            "summary": bundle.summary,
            "per_class": bundle.per_class,
            "roc_auc": bundle.roc_auc,
            "confusion_matrix": cm,
            "best_epoch": int(trainer_result.best_epoch),
            "best_eval_dice": float(trainer_result.best_metric),
            "history": trainer_result.history,
        }

        metrics_path = metrics_dir / "result_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

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
        self.visualizer.plot_training_curves(trainer_result.history)
        self.visualizer.plot_prf(metrics_bundle)
        self.visualizer.plot_confusion_matrix(metrics_bundle)
        self.visualizer.plot_roc(metrics_bundle)
        self.visualizer.plot_distribution(split_stats)

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
            self.visualizer.plot_tsne(features=features, labels=labels, title=f"{self.model_key} feature tsne")

        if modality == "hsi":
            spectra = self._collect_spectral_curves(pred_pack, max_points=int(self.config.visualization.spectral_max_points_per_class))
            self.visualizer.plot_spectral(spectra)

        attention = pred_pack.get("attention_map")
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
            return None

        model = trainer.ema.ema_model if trainer.ema is not None else trainer.model
        model.eval()

        onnx_dir = self.output_dir / "models"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = onnx_dir / "best_model.onnx"

        patch_size = int(self.config.data.patch_size)
        dummy = torch.randn(1, in_channels, patch_size, patch_size, dtype=torch.float32)

        try:
            torch.onnx.export(
                model.cpu(),
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
            return str(onnx_path)
        except Exception as exc:
            tprint(f"onnx export failed: {exc}")
            return None

    @staticmethod
    def _seed_everything(seed: int) -> None:
        """set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
