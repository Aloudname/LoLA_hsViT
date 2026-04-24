from __future__ import annotations

# end-to-end pipeline orchestration for refactored experiments.
import json
import random
import time
import copy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from munch import Munch
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from config import _to_munch
from pipeline.monitor import tprint
from pipeline.visualize import Visualizer
from model import HSIAdapter, RGBViT, UNet
from pipeline.dataset import build_dataloaders
from pipeline.trainer import Trainer, TrainerResult
from pipeline.analyzer import Analyzer, MetricsBundle
from pipeline.stats_utils import bootstrap_cosine_similarity_summary, format_anonymous_patient_id, pairwise_cosine_matrix

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
    """
    main execution bus for
    ``data -> model -> trainer -> analyzer -> visualize``.
    """

    def __init__(self, config: Munch, model_key: str, experiment_name: Optional[str] = None) -> None:
        self.config = config
        self.model_key = model_key

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
        self.visualizer = Visualizer(
            str(self.output_dir),
            class_names=list(self.config.data.class_names),
            anonymize_patients=bool(getattr(self.config.visualization, "anonymize_patients", False)),
        )

    def run(self) -> PipelineResult:
        """execute full training/eval/test workflow."""
        tprint(f"pipeline start: \n"
               f"\tmodel_key={self.model_key}, output_dir={self.output_dir}\n")

        modality = "rgb" if self.model_key == "rgb" else "hsi"

        # build dataloaders
        stage_t0 = time.perf_counter()
        tprint(f"stage[data]: build dataloaders (modality={modality})")
        loaders, prepared = build_dataloaders(self.config, modality=modality)
        tprint(f"stage[data] done in {time.perf_counter() - stage_t0:.2f}s\n")

        # dataset visualizations
        stage_t0 = time.perf_counter()
        tprint("stage[data-viz]: dataset visualizations before training")
        self._run_dataset_visualizations(prepared=prepared)
        tprint(f"stage[data-viz] done in {time.perf_counter() - stage_t0:.2f}s\n")

        # build model, resolve shape
        stage_t0 = time.perf_counter()
        in_channels = self._resolve_input_channels(modality)
        tprint(f"stage[model]: build model (in_channels={in_channels})")
        model = self._build_model(in_channels=in_channels)
        tprint(f"stage[model] done in {time.perf_counter() - stage_t0:.2f}s\n")

        # init trainer
        stage_t0 = time.perf_counter()
        tprint("stage[trainer]: init trainer")
        trainer = Trainer(model=model, config=self.config, output_dir=str(self.output_dir))
        tprint(f"stage[trainer] \033[92m{trainer.device}\033[0m trainer done in {time.perf_counter() - stage_t0:.2f}s\n")

        # plot t-SNE of initial features
        if self.config.visualization.tsne_enabled:
                stage_t0 = time.perf_counter()
                tprint("stage[pre-tsne]: plot t-SNE of pre features")
                init_features, init_labels = trainer.extract_features(loaders["train"], max_points=int(self.config.visualization.tsne_max_points))
                self.visualizer.plot_tsne(features=init_features, labels=init_labels, title=f"{self.model_key} initial feature tsne")
                tprint(f"stage[pre-tsne] done in {time.perf_counter() - stage_t0:.2f}s\n")
        
        # train/eval loop
        stage_t0 = time.perf_counter()
        tprint("stage[train]: fit train/eval")
        trainer_result = trainer.fit(
            train_loader=loaders["train"],
            val_loader=loaders["eval"],
            num_epochs=int(self.config.train.epochs),
        )
        tprint(
            f"stage[train] done in {time.perf_counter() - stage_t0:.2f}s\n"
            f"\t(best_epoch={trainer_result.best_epoch}, best_eval_dice={trainer_result.best_metric:.4f}\n)"
        )

        # test predictions
        stage_t0 = time.perf_counter()
        tprint("stage[test]: predict on test loader")
        pred_pack = trainer.predict(
            data_loader=loaders["test"],
            keep_images=int(self.config.visualization.num_seg_examples),
            keep_features=int(self.config.visualization.tsne_max_points),
        )
        tprint(f"stage[test] done in {time.perf_counter() - stage_t0:.2f}s\n")

        # analyze metrics
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

        # post-visualizations
        stage_t0 = time.perf_counter()
        tprint("stage[viz]: generate visualizations")
        self._run_visualizations(metrics_bundle, trainer_result, pred_pack, prepared, modality)
        tprint(f"stage[viz] done in {time.perf_counter() - stage_t0:.2f}s")

        # model export
        stage_t0 = time.perf_counter()
        tprint("stage[export]: export onnx")
        onnx_path = self._export_onnx(trainer, in_channels=in_channels)
        tprint(f"stage[export] done in {time.perf_counter() - stage_t0:.2f}s\n")
        trainer.cleanup_checkpoints()
        onnx_note_path = str(Path(onnx_path).with_name("best_model_info.txt")) if onnx_path else None

        tprint(f"pipeline done: model_key={self.model_key}, output_dir={self.output_dir}\n")

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

        tprint(
            "model config: \n"
            f"\tfamily={self.config.model.family}, in_channels={in_channels}\n"
            f"\tnum_classes={int(self.config.data.num_classes)}\n"
            f"\tembed_dim={int(self.config.model.embed_dim)}, depth={int(self.config.model.depth)}\n"
            f"\theads={int(self.config.model.num_heads)}\n"
            f"\tfreeze_backbone={bool(self.config.model.freeze_backbone)}\n"
            f"\tpretrained_weights={bool(getattr(self.config.model, 'pretrained_weights', True))}\n"
        )

        if self.model_key.startswith("hsi"):
            model = HSIAdapter(self.config)
        elif self.model_key == "rgb":
            model = RGBViT(self.config)
        elif self.model_key == "unet":
            model = UNet(self.config)
        elif self.model_key == "light":
            model = HSIAdapter(self.config)
        else:
            raise ValueError(f"unknown model_key: {self.model_key}")

        total_params = int(sum(p.numel() for p in model.parameters()))
        trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        tprint(f"model ready: trainable_params={trainable_params} total_params={total_params}")
        return model

    def _resolve_input_channels(self, modality: str) -> int:
        """resolve input channels according to modality and preprocessing mode."""
        if modality == "rgb":
            tprint("input channels resolved:\n"
                   "\tmodality = rgb, C = 3")
            return 3

        preprocess_mode = str(self.config.data.preprocess.mode).lower()
        if preprocess_mode == "none":
            channels = int(self.config.data.hsi_bands)
        else:
            channels = int(self.config.data.preprocess.output_dim)

        tprint(
            f"input channels resolved:\n"
            f"\tmodality = hsi, preprocess_mode = {preprocess_mode}, reduced C = {channels}"
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
        prepared: Any,
        modality: str,
    ) -> None:
        """generate all required plots."""
        
        tprint("visualization: training curves")
        self.visualizer.plot_training_curves(trainer_result.history)
        tprint("visualization: prf/confusion/roc")
        self.visualizer.plot_prf(metrics_bundle)
        self.visualizer.plot_confusion_matrix(metrics_bundle)
        self.visualizer.plot_roc(metrics_bundle)

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

    def _run_dataset_visualizations(self, prepared: Any) -> None:
        """generate dataset-only plots that do not depend on trained predictions."""
        tprint("visualization: dataset class distribution")
        self.visualizer.plot_distribution(prepared.stats)
        tprint("visualization: train sampling distribution")
        self.visualizer.plot_sampling_stats(prepared.stats)
        data_dir = self.output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        stats_for_json = copy.deepcopy(prepared.stats)

        if prepared.modality == "hsi":
            global_patient_ids_sorted = sorted(
                {str(s.patient_id) for split in prepared.samples_by_split.values() for s in split}
            )
            tprint("visualization: spectral reducer comparison")
            reducer_payload = self._collect_reducer_points(
                prepared=prepared,
                max_points=int(self.config.visualization.tsne_max_points),
            )
            if reducer_payload is not None:
                ref_features, reduced_features, reducer_labels = reducer_payload
                self.visualizer.plot_pca_lda_comparison(
                    pca_features=ref_features,
                    lda_features=reduced_features,
                    labels=reducer_labels,
                    explained_variance_ratio=getattr(prepared.reducer.pca, "explained_variance_ratio_", None),
                    pca_dim=getattr(prepared.reducer, "pca_dim", None),
                    lda_dim=getattr(prepared.reducer, "final_dim", None),
                    title=f"{self.model_key} spectral reducer comparison",
                    pca_label=getattr(prepared.reducer, "reference_projection_name", "PCA"),
                    lda_label=getattr(prepared.reducer, "reduced_projection_name", "Reducer"),
                )
            tprint("visualization: patient domain shift")
            patient_shift_payload = self._collect_patient_shift_points(prepared=prepared)
            if patient_shift_payload is not None:
                (
                    raw_points_2d,
                    norm_points_2d,
                    point_labels,
                    point_patient_ids,
                    selected_patients,
                    raw_features_hd,
                    norm_features_hd,
                ) = patient_shift_payload
                self.visualizer.plot_patient_shift(
                    raw_points_2d=raw_points_2d,
                    norm_points_2d=norm_points_2d,
                    labels=point_labels,
                    patient_ids=point_patient_ids,
                    selected_patients=selected_patients,
                    title=f"{self.model_key} patient spectral shift",
                    patient_anon_order=global_patient_ids_sorted,
                )
                self.visualizer.plot_patient_distance_distribution(
                    features_hd=norm_features_hd,
                    labels=point_labels,
                    patient_ids=point_patient_ids,
                    selected_patients=selected_patients,
                    title=f"{self.model_key} intra/inter patient distance",
                )
                self.visualizer.plot_patient_divergence_heatmaps(
                    features_hd=norm_features_hd,
                    labels=point_labels,
                    patient_ids=point_patient_ids,
                    selected_patients=selected_patients,
                    title=f"{self.model_key} patient MMD/Bhattacharyya",
                    patient_anon_order=global_patient_ids_sorted,
                )
                shift_stats = self._compute_patient_shift_stats(
                    raw_features_hd=raw_features_hd,
                    norm_features_hd=norm_features_hd,
                    labels=point_labels,
                    patient_ids=point_patient_ids,
                    selected_patients=selected_patients,
                )
                self.visualizer.plot_patient_feature_shift(
                    feature_shift_by_class=shift_stats.get("feature_shift_top_dims", {}),
                    title=f"{self.model_key} patient feature-wise shift (top dims)",
                )
                band_shift_payload = self._collect_patient_band_shift(prepared=prepared)
                if band_shift_payload is not None:
                    (
                        band_stats_all,
                        all_patient_ids,
                        band_stats_viz,
                        viz_patient_ids,
                    ) = band_shift_payload
                    self.visualizer.plot_patient_wise_band_shift(
                        band_stats_by_class=band_stats_viz,
                        patient_order=viz_patient_ids,
                        title=f"{self.model_key} patient-wise band shift",
                        patient_anon_order=global_patient_ids_sorted,
                    )
                    band_data_path = data_dir / "patient_wise_band_shift_data.npz"
                    self._save_patient_band_shift_npz(
                        path=band_data_path,
                        band_stats_all=band_stats_all,
                        all_patient_ids=all_patient_ids,
                    )
                    loaded_band_data = self._load_patient_band_shift_npz(band_data_path)
                    cluster_payload = self._cluster_patients_from_band_shift_data(loaded_band_data)
                    if cluster_payload is not None:
                        sampling_payload = {
                            "patient_clusters": cluster_payload["clusters"],
                            "cluster_metrics": cluster_payload["metrics"],
                            "model_key": self.model_key,
                        }
                        with (data_dir / "sampling.json").open("w", encoding="utf-8") as f:
                            json.dump(sampling_payload, f, indent=2, ensure_ascii=False)
                        self.visualizer.plot_patient_clustering(
                            embedding_2d=cluster_payload["embedding_2d"],
                            patient_ids=cluster_payload["patient_ids"],
                            cluster_ids=cluster_payload["cluster_ids"],
                            title=f"{self.model_key} patient clustering",
                            patient_anon_order=global_patient_ids_sorted,
                        )
                stats_for_json["patient_shift_analysis"] = shift_stats

            sa_m = getattr(self.config.data, "spectral_alignment", Munch())
            if bool(getattr(sa_m, "enabled", False)):
                tprint("visualization: Stage A (on-disk diff [+ SNV]) band curves and similarity")
                tgt = int(getattr(sa_m, "viz_random_patients", 16))
                diff_payload = self._collect_patient_band_diff_shift(prepared=prepared, target_patients=tgt)
                if diff_payload is not None:
                    band_diff_all, all_diff_patient_ids, band_diff_viz, viz_diff_patient_ids = diff_payload
                    self.visualizer.plot_patient_wise_band_diff_shift(
                        band_stats_by_class=band_diff_viz,
                        patient_order=viz_diff_patient_ids,
                        title=f"{self.model_key} Stage A patient-wise band statistics",
                        patient_anon_order=global_patient_ids_sorted,
                    )
                    diff_npz = data_dir / "patient_wise_band_diff_shift_data.npz"
                    self._save_patient_band_diff_shift_npz(
                        path=diff_npz,
                        band_stats_all=band_diff_all,
                        all_patient_ids=all_diff_patient_ids,
                    )
                    mean_arr = np.asarray(band_diff_all["mean"], dtype=np.float64)
                    pid_to_idx = {p: i for i, p in enumerate(all_diff_patient_ids)}
                    plabs = [p for p in viz_diff_patient_ids if p in pid_to_idx]
                    n_boot = int(getattr(sa_m, "similarity_bootstrap_iters", 2000))
                    seed_sim = int(self.config.data.split.seed) + int(getattr(sa_m, "similarity_random_seed", 137))
                    sim_json: Dict[str, Any] = {"classes": {}, "n_bootstrap": n_boot}
                    if len(plabs) < 2:
                        sim_json["note"] = "insufficient viz patients for pairwise Stage A similarity"
                    else:
                        for ci, cname in enumerate(band_diff_all["class_names"]):
                            curves = np.stack([mean_arr[ci, pid_to_idx[p], :] for p in plabs], axis=0)
                            sim_mat = pairwise_cosine_matrix(curves)
                            safe_name = "".join(
                                ch if ch.isalnum() or ch in "-_" else "_" for ch in str(cname)
                            )
                            self.visualizer.plot_band_diff_similarity_summary(
                                similarity_matrix=sim_mat,
                                patient_labels=plabs,
                                title=f"{self.model_key} Stage A cosine similarity ({cname})",
                                patient_anon_order=global_patient_ids_sorted,
                                rel_path=f"data/patient_band_diff_similarity_{safe_name}.png",
                            )
                            boot = bootstrap_cosine_similarity_summary(
                                curves,
                                n_bootstrap=n_boot,
                                seed=seed_sim + ci,
                                patient_labels=plabs,
                            )
                            if bool(getattr(self.config.visualization, "anonymize_patients", False)):
                                for row in boot.get("pairwise", []):
                                    pi = row.get("patient_id_i")
                                    pj = row.get("patient_id_j")
                                    if isinstance(pi, str) and isinstance(pj, str):
                                        row["patient_display_i"] = format_anonymous_patient_id(
                                            pi, global_patient_ids_sorted
                                        )
                                        row["patient_display_j"] = format_anonymous_patient_id(
                                            pj, global_patient_ids_sorted
                                        )
                            sim_json["classes"][str(cname)] = boot
                    with (data_dir / "patient_band_diff_similarity.json").open("w", encoding="utf-8") as f:
                        json.dump(sim_json, f, indent=2, ensure_ascii=False, default=str)

        # keep runtime JSON compact and remove huge cache path mapping
        stats_for_json.pop("cached_hsi_paths", None)
        _sam = stats_for_json.get("spectral_alignment")
        if isinstance(_sam, dict) and "per_sample" in _sam:
            _sam = dict(_sam)
            _sam.pop("per_sample", None)
            stats_for_json["spectral_alignment"] = _sam
        with (data_dir / "sampling_stats.json").open("w", encoding="utf-8") as f:
            json.dump(stats_for_json, f, indent=2, ensure_ascii=False)

    def _collect_reducer_points(self, prepared: Any, max_points: int) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """collect balanced pixels for spectral reducer comparison plots."""
        reducer = getattr(prepared, "reducer", None)
        if reducer is None or not hasattr(reducer, "project_pixels"):
            return None

        samples = list(prepared.samples_by_split.get("train", []))
        if not samples:
            return None

        num_classes = int(self.config.data.num_classes)
        rng = np.random.default_rng(int(self.config.runtime.seed))
        per_class_budget = max(32, int(max_points // max(1, num_classes)))
        per_sample_budget = max(32, int(per_class_budget // max(1, len(samples) // 3 + 1)))

        x_chunks: List[np.ndarray] = []
        y_chunks: List[np.ndarray] = []

        for sample in samples:
            cube = np.load(sample.hsi_path).astype(np.float32)
            mask = np.load(sample.mask_path).astype(np.int64)
            if cube.ndim != 3 or mask.ndim != 2:
                continue

            h, w, c = cube.shape
            flat = cube.reshape(-1, c)
            labels = mask.reshape(-1)

            # spectral projection excludes background class (0)
            for cls_idx in range(1, num_classes):
                idx = np.where(labels == cls_idx)[0]
                if idx.size == 0:
                    continue
                take = min(idx.size, per_sample_budget)
                chosen = rng.choice(idx, size=take, replace=False)
                x_chunks.append(flat[chosen])
                y_chunks.append(np.full(take, cls_idx, dtype=np.int64))

        if not x_chunks:
            return None

        x_raw = np.concatenate(x_chunks, axis=0).astype(np.float32)
        y_raw = np.concatenate(y_chunks, axis=0).astype(np.int64)

        if x_raw.shape[0] > max_points:
            chosen = rng.choice(x_raw.shape[0], size=max_points, replace=False)
            x_raw = x_raw[chosen]
            y_raw = y_raw[chosen]

        ref_features, reduced_features = reducer.project_pixels(x_raw)
        if ref_features.size == 0 or reduced_features.size == 0:
            return None
        return ref_features, reduced_features, y_raw

    def _collect_spectral_curves(self, pred_pack: Mapping[str, Any], max_points: int) -> Dict[str, Any]:
        """estimate spectral mean/std from retained image samples and gt masks."""
        images = pred_pack.get("image_samples", [])
        gts = pred_pack.get("gt_masks", [])

        if images is None or gts is None:
            return {}
        if len(images) == 0 or len(gts) == 0:
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

    def _collect_patient_shift_points(
        self,
        prepared: Any,
        target_patients: int = 10,
        max_pixels_per_patient_class: int = 2000,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str], np.ndarray, np.ndarray]]:
        """
        Collect per-pixel projected features for cross-patient shift visualization.
        Uses reducer projection + discriminative 2D selection (same family as spectral projection).
        """
        all_samples: List[Any] = []
        for split_name in ("train", "eval", "test"):
            all_samples.extend(list(prepared.samples_by_split.get(split_name, [])))
        if not all_samples:
            return None

        rng = np.random.default_rng(int(self.config.runtime.seed))
        target_patients = int(np.clip(target_patients, 5, 10))

        by_patient: Dict[str, List[Any]] = {}
        for sample in all_samples:
            by_patient.setdefault(str(sample.patient_id), []).append(sample)

        patient_pool = sorted(by_patient.keys())
        if not patient_pool:
            return None
        if len(patient_pool) > target_patients:
            chosen_idx = rng.choice(len(patient_pool), size=target_patients, replace=False)
            selected_patients = [patient_pool[int(i)] for i in sorted(chosen_idx)]
        else:
            selected_patients = patient_pool

        reducer = getattr(prepared, "reducer", None)
        if reducer is None or not hasattr(reducer, "project_pixels"):
            return None

        raw_chunks: List[np.ndarray] = []
        norm_chunks: List[np.ndarray] = []
        label_chunks: List[np.ndarray] = []
        patient_chunks: List[np.ndarray] = []
        num_classes = int(self.config.data.num_classes)

        for pid in selected_patients:
            patient_samples = by_patient.get(pid, [])
            for sample in patient_samples:
                cube = np.load(sample.hsi_path).astype(np.float32)
                mask = np.load(sample.mask_path).astype(np.int64)
                if cube.ndim != 3 or mask.ndim != 2:
                    continue
                flat = cube.reshape(-1, cube.shape[-1])
                labels = mask.reshape(-1)
                for cls_idx in range(1, num_classes):
                    idx = np.where(labels == cls_idx)[0]
                    if idx.size == 0:
                        continue
                    take = min(int(max_pixels_per_patient_class), int(idx.size))
                    chosen = rng.choice(idx, size=take, replace=False)
                    raw_sel = flat[chosen]
                    norm_sel = reducer._prepare_features(raw_sel, fit=False) if hasattr(reducer, "_prepare_features") else raw_sel
                    raw_chunks.append(raw_sel)
                    norm_chunks.append(norm_sel)
                    label_chunks.append(np.full(take, cls_idx, dtype=np.int64))
                    patient_chunks.append(np.full(take, pid, dtype=object))

        if not raw_chunks:
            return None

        x_raw = np.concatenate(raw_chunks, axis=0).astype(np.float32)
        x_norm = np.concatenate(norm_chunks, axis=0).astype(np.float32)
        y = np.concatenate(label_chunks, axis=0).astype(np.int64)
        p = np.concatenate(patient_chunks, axis=0).astype(object)

        if x_raw.shape[0] < 20 or x_norm.shape[0] < 20:
            return None

        # Keep same projection style: discriminative 2D selected from feature space.
        def _best2d(points: np.ndarray) -> np.ndarray:
            if points.shape[1] == 1:
                return np.concatenate([points, np.zeros((points.shape[0], 1), dtype=np.float32)], axis=1)
            if points.shape[1] == 2:
                return points.astype(np.float32)
            global_mean = points.mean(axis=0)
            scores = np.zeros(points.shape[1], dtype=np.float64)
            for cls_idx in range(1, num_classes):
                m = y == cls_idx
                if not np.any(m):
                    continue
                cls = points[m]
                cls_mean = cls.mean(axis=0)
                cls_var = cls.var(axis=0)
                scores += cls.shape[0] * (cls_mean - global_mean) ** 2 / np.maximum(cls_var, 1e-8)
            best = np.argsort(-scores)[:2]
            best = np.sort(best)
            return points[:, best].astype(np.float32)

        raw_points_2d = _best2d(x_raw)
        norm_points_2d = _best2d(x_norm)
        return raw_points_2d, norm_points_2d, y, [str(v) for v in p.tolist()], selected_patients, x_raw, x_norm

    def _compute_patient_shift_stats(
        self,
        raw_features_hd: np.ndarray,
        norm_features_hd: np.ndarray,
        labels: np.ndarray,
        patient_ids: List[str],
        selected_patients: List[str],
    ) -> Dict[str, Any]:
        """Compute quantitative patient shift statistics for JSON/reporting."""
        x_raw = np.asarray(raw_features_hd, dtype=np.float64)
        x_norm = np.asarray(norm_features_hd, dtype=np.float64)
        y = np.asarray(labels, dtype=np.int64)
        p = np.asarray([str(v) for v in patient_ids], dtype=object)
        fg_labels = [idx for idx in range(1, int(self.config.data.num_classes))]
        rng = np.random.default_rng(int(self.config.runtime.seed) + 73)

        def _sample_pair_dist(a: np.ndarray, b: np.ndarray, n_pairs: int = 1200) -> np.ndarray:
            if a.shape[0] == 0 or b.shape[0] == 0:
                return np.empty((0,), dtype=np.float64)
            ia = rng.integers(0, a.shape[0], size=n_pairs)
            ib = rng.integers(0, b.shape[0], size=n_pairs)
            return np.linalg.norm(a[ia] - b[ib], axis=1)

        def _rbf_mmd2(a: np.ndarray, b: np.ndarray) -> float:
            na = min(256, a.shape[0]); nb = min(256, b.shape[0])
            if na < 4 or nb < 4:
                return float("nan")
            a = a[rng.choice(a.shape[0], size=na, replace=False)]
            b = b[rng.choice(b.shape[0], size=nb, replace=False)]
            z = np.concatenate([a, b], axis=0)
            d2 = np.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=-1)
            sigma2 = float(np.median(d2[np.triu_indices_from(d2, k=1)]))
            sigma2 = max(sigma2, 1e-6)
            kaa = np.exp(-np.sum((a[:, None, :] - a[None, :, :]) ** 2, axis=-1) / (2.0 * sigma2))
            kbb = np.exp(-np.sum((b[:, None, :] - b[None, :, :]) ** 2, axis=-1) / (2.0 * sigma2))
            kab = np.exp(-np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1) / (2.0 * sigma2))
            return float(kaa.mean() + kbb.mean() - 2.0 * kab.mean())

        def _bhattacharyya(a: np.ndarray, b: np.ndarray) -> float:
            if a.shape[0] < 4 or b.shape[0] < 4:
                return float("nan")
            mu1 = a.mean(axis=0); mu2 = b.mean(axis=0)
            c1 = np.cov(a, rowvar=False); c2 = np.cov(b, rowvar=False)
            dim = c1.shape[0]
            reg = 1e-4 * np.eye(dim)
            c1 = c1 + reg; c2 = c2 + reg
            c = 0.5 * (c1 + c2)
            invc = np.linalg.pinv(c)
            diff = (mu1 - mu2).reshape(-1, 1)
            t1 = 0.125 * float(diff.T @ invc @ diff)
            det_c = max(np.linalg.det(c), 1e-12)
            det_c1 = max(np.linalg.det(c1), 1e-12)
            det_c2 = max(np.linalg.det(c2), 1e-12)
            t2 = 0.5 * np.log(det_c / np.sqrt(det_c1 * det_c2))
            return float(t1 + t2)

        summary: Dict[str, Any] = {
            "selected_patients": list(selected_patients),
            "num_selected_patients": len(selected_patients),
            "samples": {
                "num_points": int(x_norm.shape[0]),
                "num_features_raw": int(x_raw.shape[1]) if x_raw.ndim == 2 else 0,
                "num_features_norm": int(x_norm.shape[1]) if x_norm.ndim == 2 else 0,
            },
            "by_class": {},
            "feature_shift_top_dims": {},
        }

        for cls_idx in fg_labels:
            cls_name = self.config.data.class_names[cls_idx] if cls_idx < len(self.config.data.class_names) else str(cls_idx)
            intra_list: List[np.ndarray] = []
            inter_list: List[np.ndarray] = []
            mmd_vals: List[float] = []
            bha_vals: List[float] = []
            patient_means: List[np.ndarray] = []

            for pid in selected_patients:
                cur = x_norm[(y == cls_idx) & (p == pid)]
                if cur.shape[0] < 5:
                    continue
                intra_list.append(_sample_pair_dist(cur, cur))
                patient_means.append(cur.mean(axis=0))

            for i in range(len(selected_patients)):
                for j in range(i + 1, len(selected_patients)):
                    ai = x_norm[(y == cls_idx) & (p == selected_patients[i])]
                    bj = x_norm[(y == cls_idx) & (p == selected_patients[j])]
                    if ai.shape[0] < 5 or bj.shape[0] < 5:
                        continue
                    inter_list.append(_sample_pair_dist(ai, bj))
                    mmd_vals.append(_rbf_mmd2(ai, bj))
                    bha_vals.append(_bhattacharyya(ai, bj))

            intra = np.concatenate(intra_list, axis=0) if intra_list else np.empty((0,), dtype=np.float64)
            inter = np.concatenate(inter_list, axis=0) if inter_list else np.empty((0,), dtype=np.float64)
            class_stat = {
                "intra_mean": float(np.mean(intra)) if intra.size > 0 else None,
                "inter_mean": float(np.mean(inter)) if inter.size > 0 else None,
                "intra_median": float(np.median(intra)) if intra.size > 0 else None,
                "inter_median": float(np.median(inter)) if inter.size > 0 else None,
                "inter_over_intra_ratio": float(np.mean(inter) / max(np.mean(intra), 1e-8)) if intra.size > 0 and inter.size > 0 else None,
                "mmd_mean": float(np.nanmean(mmd_vals)) if mmd_vals else None,
                "mmd_std": float(np.nanstd(mmd_vals)) if mmd_vals else None,
                "bhattacharyya_mean": float(np.nanmean(bha_vals)) if bha_vals else None,
                "bhattacharyya_std": float(np.nanstd(bha_vals)) if bha_vals else None,
                "num_valid_patient_pairs": int(len(mmd_vals)),
            }
            summary["by_class"][cls_name] = class_stat

            if patient_means:
                means = np.stack(patient_means, axis=0)
                # feature-wise shift strength: std across patient means
                shift_strength = means.std(axis=0)
                top_k = min(12, shift_strength.shape[0])
                top_idx = np.argsort(-shift_strength)[:top_k]
                summary["feature_shift_top_dims"][cls_name] = [
                    {"feature_idx": int(i), "shift_strength": float(shift_strength[i])}
                    for i in top_idx
                ]
            else:
                summary["feature_shift_top_dims"][cls_name] = []

        return summary

    def _save_patient_band_shift_npz(
        self,
        path: Path,
        band_stats_all: Mapping[str, Any],
        all_patient_ids: Sequence[str],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(path),
            class_names=np.asarray(band_stats_all["class_names"], dtype="U32"),
            patient_ids=np.asarray(list(all_patient_ids), dtype="U64"),
            mean=np.asarray(band_stats_all["mean"], dtype=np.float32),
            median=np.asarray(band_stats_all["median"], dtype=np.float32),
            std=np.asarray(band_stats_all["std"], dtype=np.float32),
            count=np.asarray(band_stats_all["count"], dtype=np.int32),
        )

    def _load_patient_band_shift_npz(self, path: Path) -> Dict[str, Any]:
        with np.load(str(path), allow_pickle=False) as f:
            return {
                "class_names": [str(v) for v in f["class_names"].tolist()],
                "patient_ids": [str(v) for v in f["patient_ids"].tolist()],
                "mean": np.asarray(f["mean"], dtype=np.float64),
                "median": np.asarray(f["median"], dtype=np.float64),
                "std": np.asarray(f["std"], dtype=np.float64),
                "count": np.asarray(f["count"], dtype=np.int64),
            }

    def _cluster_patients_from_band_shift_data(self, band_data: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        class_names = list(band_data.get("class_names", []))
        patient_ids = list(band_data.get("patient_ids", []))
        mean_arr = np.asarray(band_data.get("mean", []), dtype=np.float64)      # [C, P, B]
        std_arr = np.asarray(band_data.get("std", []), dtype=np.float64)        # [C, P, B]
        count_arr = np.asarray(band_data.get("count", []), dtype=np.int64)      # [C, P]
        if mean_arr.ndim != 3 or std_arr.ndim != 3 or count_arr.ndim != 2 or len(patient_ids) < 2:
            return None
        c, p, b = mean_arr.shape
        if p != len(patient_ids):
            return None
        # Build patient feature using all patients; missing class stats are imputed later.
        x = np.concatenate([mean_arr, std_arr], axis=2)  # [C, P, 2B]
        x = np.transpose(x, (1, 0, 2)).reshape(p, c * 2 * b)  # [P, C*2B]
        # mark missing class stats as NaN using count==0
        missing_mask = (count_arr <= 0)  # [C, P]
        if np.any(missing_mask):
            for cls_idx in range(c):
                miss_p = np.where(missing_mask[cls_idx])[0]
                if miss_p.size == 0:
                    continue
                start = cls_idx * (2 * b)
                end = start + (2 * b)
                x[miss_p, start:end] = np.nan
        # impute NaN with feature-wise mean
        feat_mean = np.nanmean(x, axis=0)
        feat_mean = np.where(np.isfinite(feat_mean), feat_mean, 0.0)
        nan_idx = np.where(~np.isfinite(x))
        if nan_idx[0].size > 0:
            x[nan_idx] = feat_mean[nan_idx[1]]

        mu = x.mean(axis=0, keepdims=True)
        sd = x.std(axis=0, keepdims=True)
        xz = (x - mu) / np.maximum(sd, 1e-8)

        n = xz.shape[0]
        if n == 2:
            labels = np.array([0, 1], dtype=np.int32)
        else:
            k_min = 2
            k_max = min(6, n - 1)
            best_k = 2
            best_score = -1.0
            best_labels = None
            for k in range(k_min, k_max + 1):
                model = AgglomerativeClustering(n_clusters=k, linkage="ward")
                cur = model.fit_predict(xz)
                if len(np.unique(cur)) < 2:
                    continue
                score = float(silhouette_score(xz, cur))
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_labels = cur
            labels = best_labels if best_labels is not None else AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit_predict(xz)

        clusters: Dict[str, Any] = {}
        dist_mat = np.linalg.norm(xz[:, None, :] - xz[None, :, :], axis=-1)
        for cid in sorted(np.unique(labels).tolist()):
            member_indices = [i for i in range(len(patient_ids)) if int(labels[i]) == int(cid)]
            members = [patient_ids[i] for i in member_indices]
            cluster_entry: Dict[str, Any] = {"patients": members, "size": len(members)}

            if len(member_indices) >= 1:
                # Medoid: patient with minimum sum distance to others in the cluster.
                sub = dist_mat[np.ix_(member_indices, member_indices)]
                sums = sub.sum(axis=1)
                medoid_local_idx = int(np.argmin(sums))
                medoid_global_idx = member_indices[medoid_local_idx]
                cluster_entry["medoid_patient_id"] = patient_ids[medoid_global_idx]
            else:
                cluster_entry["medoid_patient_id"] = None

            if len(member_indices) >= 2:
                nearest_pair = None
                farthest_pair = None
                nearest_dist = float("inf")
                farthest_dist = -float("inf")
                for i_pos in range(len(member_indices)):
                    for j_pos in range(i_pos + 1, len(member_indices)):
                        gi = member_indices[i_pos]
                        gj = member_indices[j_pos]
                        d = float(dist_mat[gi, gj])
                        if d < nearest_dist:
                            nearest_dist = d
                            nearest_pair = (patient_ids[gi], patient_ids[gj])
                        if d > farthest_dist:
                            farthest_dist = d
                            farthest_pair = (patient_ids[gi], patient_ids[gj])
                cluster_entry["nearest_patient_pair"] = {
                    "patients": list(nearest_pair) if nearest_pair is not None else [],
                    "distance": float(nearest_dist) if nearest_pair is not None else None,
                }
                cluster_entry["farthest_patient_pair"] = {
                    "patients": list(farthest_pair) if farthest_pair is not None else [],
                    "distance": float(farthest_dist) if farthest_pair is not None else None,
                }
            else:
                cluster_entry["nearest_patient_pair"] = {"patients": [], "distance": None}
                cluster_entry["farthest_patient_pair"] = {"patients": [], "distance": None}

            clusters[f"cluster_{int(cid)}"] = cluster_entry

        # Compute class-wise intra/inter distance metrics.
        metrics: Dict[str, Any] = {"global": {}, "by_class": {}}
        for cls_idx, cls in enumerate(class_names):
            class_vecs = []
            valid_patient_indices = []
            for pid_idx in range(len(patient_ids)):
                if count_arr[cls_idx, pid_idx] <= 0:
                    continue
                class_vecs.append(mean_arr[cls_idx, pid_idx])
                valid_patient_indices.append(pid_idx)
            if len(class_vecs) < 2:
                metrics["by_class"][cls] = {
                    "intra_mean_distance": None,
                    "intra_std_distance": None,
                    "inter_mean_distance": None,
                    "inter_std_distance": None,
                    "inter_over_intra_ratio": None,
                }
                continue
            arr = np.stack(class_vecs, axis=0)
            intra_dists: List[float] = []
            inter_dists: List[float] = []
            for i in range(arr.shape[0]):
                for j in range(i + 1, arr.shape[0]):
                    d = float(np.linalg.norm(arr[i] - arr[j]))
                    if labels[valid_patient_indices[i]] == labels[valid_patient_indices[j]]:
                        intra_dists.append(d)
                    else:
                        inter_dists.append(d)
            metrics["by_class"][cls] = {
                "intra_mean_distance": float(np.mean(intra_dists)) if intra_dists else None,
                "intra_std_distance": float(np.std(intra_dists)) if intra_dists else None,
                "inter_mean_distance": float(np.mean(inter_dists)) if inter_dists else None,
                "inter_std_distance": float(np.std(inter_dists)) if inter_dists else None,
                "inter_over_intra_ratio": (
                    float(np.mean(inter_dists) / max(np.mean(intra_dists), 1e-8))
                    if intra_dists and inter_dists
                    else None
                ),
            }

        # global metrics over concatenated patient vectors
        global_intra: List[float] = []
        global_inter: List[float] = []
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(xz[i] - xz[j]))
                if labels[i] == labels[j]:
                    global_intra.append(d)
                else:
                    global_inter.append(d)
        metrics["global"] = {
            "num_patients": int(n),
            "num_clusters": int(len(np.unique(labels))),
            "intra_mean_distance": float(np.mean(global_intra)) if global_intra else None,
            "intra_std_distance": float(np.std(global_intra)) if global_intra else None,
            "inter_mean_distance": float(np.mean(global_inter)) if global_inter else None,
            "inter_std_distance": float(np.std(global_inter)) if global_inter else None,
            "inter_over_intra_ratio": (
                float(np.mean(global_inter) / max(np.mean(global_intra), 1e-8))
                if global_intra and global_inter
                else None
            ),
        }

        pca = PCA(n_components=2, random_state=int(self.config.runtime.seed))
        emb = pca.fit_transform(xz).astype(np.float32)
        return {
            "clusters": clusters,
            "metrics": metrics,
            "patient_ids": patient_ids,
            "cluster_ids": [int(v) for v in labels.tolist()],
            "embedding_2d": emb,
        }

    def _collect_patient_band_shift(
        self,
        prepared: Any,
        target_patients: int = 16,
    ) -> Optional[Tuple[Dict[str, Any], List[str], Dict[str, Dict[str, Dict[str, np.ndarray]]], List[str]]]:
        """
        Build per-band patient statistics (mean/median/std) on raw reflectance
        for each foreground class before training.
        """
        all_samples: List[Any] = []
        for split_name in ("train", "eval", "test"):
            all_samples.extend(list(prepared.samples_by_split.get(split_name, [])))
        if not all_samples:
            return None

        by_patient: Dict[str, List[Any]] = {}
        for sample in all_samples:
            by_patient.setdefault(str(sample.patient_id), []).append(sample)
        patient_pool = sorted(by_patient.keys())
        if not patient_pool:
            return None

        rng = np.random.default_rng(int(self.config.runtime.seed) + 137)
        target_patients = int(np.clip(target_patients, 7, 16))
        if len(patient_pool) > target_patients:
            chosen_idx = rng.choice(len(patient_pool), size=target_patients, replace=False)
            selected_patients_viz = [patient_pool[int(i)] for i in sorted(chosen_idx)]
        else:
            selected_patients_viz = patient_pool

        num_classes = int(self.config.data.num_classes)
        class_names = [self.config.data.class_names[i] if i < len(self.config.data.class_names) else str(i) for i in range(num_classes)]
        fg_class_names = [class_names[i] for i in range(1, num_classes)]
        n_cls = len(fg_class_names)
        n_pat = len(patient_pool)
        mean_arr: List[np.ndarray] = []
        median_arr: List[np.ndarray] = []
        std_arr: List[np.ndarray] = []
        count_arr = np.zeros((n_cls, n_pat), dtype=np.int32)
        band_stats_by_class_viz: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
        n_bands = None
        for cls_idx in range(1, num_classes):
            cls_name = class_names[cls_idx]
            by_patient_stats: Dict[str, Dict[str, np.ndarray]] = {}
            cls_means: List[np.ndarray] = []
            cls_medians: List[np.ndarray] = []
            cls_stds: List[np.ndarray] = []
            for pid_idx, pid in enumerate(patient_pool):
                rows: List[np.ndarray] = []
                for sample in by_patient.get(pid, []):
                    cube = np.load(sample.hsi_path).astype(np.float32)
                    mask = np.load(sample.mask_path).astype(np.int64)
                    if cube.ndim != 3 or mask.ndim != 2:
                        continue
                    # Support both HWC and CHW cube layouts.
                    if cube.shape[:2] == mask.shape:
                        cube_hwc = cube
                    elif cube.shape[1:] == mask.shape:
                        cube_hwc = np.moveaxis(cube, 0, -1)
                    else:
                        continue
                    m = mask == cls_idx
                    if not np.any(m):
                        continue
                    rows.append(cube_hwc[m])  # [N, bands]
                if not rows:
                    if n_bands is None:
                        # infer n_bands from first valid sample across all patients/classes later
                        cls_means.append(np.array([], dtype=np.float32))
                        cls_medians.append(np.array([], dtype=np.float32))
                        cls_stds.append(np.array([], dtype=np.float32))
                    else:
                        cls_means.append(np.full((n_bands,), np.nan, dtype=np.float32))
                        cls_medians.append(np.full((n_bands,), np.nan, dtype=np.float32))
                        cls_stds.append(np.full((n_bands,), np.nan, dtype=np.float32))
                    continue
                arr = np.concatenate(rows, axis=0).astype(np.float32)
                if n_bands is None:
                    n_bands = int(arr.shape[1])
                mean_v = arr.mean(axis=0)
                med_v = np.median(arr, axis=0)
                std_v = arr.std(axis=0)
                count_arr[cls_idx - 1, pid_idx] = int(arr.shape[0])
                cls_means.append(mean_v)
                cls_medians.append(med_v)
                cls_stds.append(std_v)
                by_patient_stats[pid] = {"mean": mean_v, "median": med_v, "std": std_v}
            if n_bands is None:
                return None
            # backfill any empty placeholders created before n_bands was known
            for i in range(len(cls_means)):
                if cls_means[i].size == 0:
                    cls_means[i] = np.full((n_bands,), np.nan, dtype=np.float32)
                    cls_medians[i] = np.full((n_bands,), np.nan, dtype=np.float32)
                    cls_stds[i] = np.full((n_bands,), np.nan, dtype=np.float32)
            mean_arr.append(np.stack(cls_means, axis=0))
            median_arr.append(np.stack(cls_medians, axis=0))
            std_arr.append(np.stack(cls_stds, axis=0))

            # viz-only subset keeps a manageable number of patients
            band_stats_by_class_viz[cls_name] = {pid: by_patient_stats[pid] for pid in selected_patients_viz if pid in by_patient_stats}

        if not mean_arr:
            return None
        band_stats_all = {
            "class_names": fg_class_names,
            "mean": np.stack(mean_arr, axis=0),      # [C, P, B]
            "median": np.stack(median_arr, axis=0),  # [C, P, B]
            "std": np.stack(std_arr, axis=0),        # [C, P, B]
            "count": count_arr,                      # [C, P]
        }
        return band_stats_all, patient_pool, band_stats_by_class_viz, selected_patients_viz

    def _collect_patient_band_diff_shift(
        self,
        prepared: Any,
        target_patients: int = 16,
    ) -> Optional[Tuple[Dict[str, Any], List[str], Dict[str, Dict[str, Dict[str, np.ndarray]]], List[str]]]:
        """
        Per-patient band mean/median/std on **on-disk** cubes (Stage A: diff [+ SNV]).
        Same layout as _collect_patient_band_shift; uses a different RNG stream for viz subset.
        """
        all_samples: List[Any] = []
        for split_name in ("train", "eval", "test"):
            all_samples.extend(list(prepared.samples_by_split.get(split_name, [])))
        if not all_samples:
            return None

        by_patient: Dict[str, List[Any]] = {}
        for sample in all_samples:
            by_patient.setdefault(str(sample.patient_id), []).append(sample)
        patient_pool = sorted(by_patient.keys())
        if not patient_pool:
            return None

        rng = np.random.default_rng(int(self.config.data.split.seed) + 139)
        target_patients = int(np.clip(target_patients, 7, 16))
        if len(patient_pool) > target_patients:
            chosen_idx = rng.choice(len(patient_pool), size=target_patients, replace=False)
            selected_patients_viz = [patient_pool[int(i)] for i in sorted(chosen_idx)]
        else:
            selected_patients_viz = patient_pool

        num_classes = int(self.config.data.num_classes)
        class_names = [
            self.config.data.class_names[i] if i < len(self.config.data.class_names) else str(i)
            for i in range(num_classes)
        ]
        fg_class_names = [class_names[i] for i in range(1, num_classes)]
        n_cls = len(fg_class_names)
        n_pat = len(patient_pool)
        mean_arr: List[np.ndarray] = []
        median_arr: List[np.ndarray] = []
        std_arr: List[np.ndarray] = []
        count_arr = np.zeros((n_cls, n_pat), dtype=np.int32)
        band_stats_by_class_viz: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
        n_bands = None
        for cls_idx in range(1, num_classes):
            cls_name = class_names[cls_idx]
            by_patient_stats: Dict[str, Dict[str, np.ndarray]] = {}
            cls_means: List[np.ndarray] = []
            cls_medians: List[np.ndarray] = []
            cls_stds: List[np.ndarray] = []
            for pid_idx, pid in enumerate(patient_pool):
                rows: List[np.ndarray] = []
                for sample in by_patient.get(pid, []):
                    cube = np.load(sample.hsi_path).astype(np.float32)
                    mask = np.load(sample.mask_path).astype(np.int64)
                    if cube.ndim != 3 or mask.ndim != 2:
                        continue
                    if cube.shape[:2] == mask.shape:
                        cube_hwc = cube
                    elif cube.shape[1:] == mask.shape:
                        cube_hwc = np.moveaxis(cube, 0, -1)
                    else:
                        continue
                    m = mask == cls_idx
                    if not np.any(m):
                        continue
                    rows.append(cube_hwc[m])
                if not rows:
                    if n_bands is None:
                        cls_means.append(np.array([], dtype=np.float32))
                        cls_medians.append(np.array([], dtype=np.float32))
                        cls_stds.append(np.array([], dtype=np.float32))
                    else:
                        cls_means.append(np.full((n_bands,), np.nan, dtype=np.float32))
                        cls_medians.append(np.full((n_bands,), np.nan, dtype=np.float32))
                        cls_stds.append(np.full((n_bands,), np.nan, dtype=np.float32))
                    continue
                arr = np.concatenate(rows, axis=0).astype(np.float32)
                if n_bands is None:
                    n_bands = int(arr.shape[1])
                mean_v = arr.mean(axis=0)
                med_v = np.median(arr, axis=0)
                std_v = arr.std(axis=0)
                count_arr[cls_idx - 1, pid_idx] = int(arr.shape[0])
                cls_means.append(mean_v)
                cls_medians.append(med_v)
                cls_stds.append(std_v)
                by_patient_stats[pid] = {"mean": mean_v, "median": med_v, "std": std_v}
            if n_bands is None:
                return None
            for i in range(len(cls_means)):
                if cls_means[i].size == 0:
                    cls_means[i] = np.full((n_bands,), np.nan, dtype=np.float32)
                    cls_medians[i] = np.full((n_bands,), np.nan, dtype=np.float32)
                    cls_stds[i] = np.full((n_bands,), np.nan, dtype=np.float32)
            mean_arr.append(np.stack(cls_means, axis=0))
            median_arr.append(np.stack(cls_medians, axis=0))
            std_arr.append(np.stack(cls_stds, axis=0))
            band_stats_by_class_viz[cls_name] = {
                pid: by_patient_stats[pid] for pid in selected_patients_viz if pid in by_patient_stats
            }

        if not mean_arr:
            return None
        band_stats_all = {
            "class_names": fg_class_names,
            "mean": np.stack(mean_arr, axis=0),
            "median": np.stack(median_arr, axis=0),
            "std": np.stack(std_arr, axis=0),
            "count": count_arr,
        }
        return band_stats_all, patient_pool, band_stats_by_class_viz, selected_patients_viz

    def _save_patient_band_diff_shift_npz(
        self,
        path: Path,
        band_stats_all: Mapping[str, Any],
        all_patient_ids: Sequence[str],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(path),
            class_names=np.asarray(band_stats_all["class_names"], dtype="U32"),
            patient_ids=np.asarray(list(all_patient_ids), dtype="U64"),
            mean=np.asarray(band_stats_all["mean"], dtype=np.float32),
            median=np.asarray(band_stats_all["median"], dtype=np.float32),
            std=np.asarray(band_stats_all["std"], dtype=np.float32),
            count=np.asarray(band_stats_all["count"], dtype=np.int32),
        )

    def _load_patient_band_diff_shift_npz(self, path: Path) -> Dict[str, Any]:
        with np.load(str(path), allow_pickle=False) as f:
            return {
                "class_names": [str(v) for v in f["class_names"].tolist()],
                "patient_ids": [str(v) for v in f["patient_ids"].tolist()],
                "mean": np.asarray(f["mean"], dtype=np.float64),
                "median": np.asarray(f["median"], dtype=np.float64),
                "std": np.asarray(f["std"], dtype=np.float64),
                "count": np.asarray(f["count"], dtype=np.int64),
            }

    def _export_onnx(self, trainer: Trainer, in_channels: int) -> Optional[str]:
        """export best model to onnx."""
        if not bool(self.config.export.export_onnx):
            tprint("onnx export skipped: export.export_onnx=false")
            return None

        model = trainer.ema.ema if trainer.ema is not None else trainer.model
        model.eval()

        onnx_dir = self.output_dir / "models"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = onnx_dir / "best_model.onnx"
        note_path = onnx_dir / "best_model_info.txt"

        tprint(
            ".onnx export start: "
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
            "_____________________",
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
            "_____",
            "name: input",
            f"dtype: {str(input_sample.dtype).replace('torch.', '')}",
            f"shape_example: {input_shape}",
            "dynamic_axes: batch axis 0",
            "layout: [N, C, H, W]",
            f"recommended_patch_size: {patch_size} x {patch_size}",
            "",
            "Output",
            "______",
            "name: logits",
            f"dtype: {str(output_sample.dtype).replace('torch.', '')}",
            f"shape_example: {output_shape}",
            "dynamic_axes: batch axis 0",
            "layout: [N, num_classes, H, W]",
            "postprocess: argmax(logits, axis=1) -> predicted label map [N, H, W]",
            "",
            "Class Mapping",
            "_____________",
            *class_map_lines,
            "",
            "Preprocess Contract",
            "___________________",
            *preprocess_lines,
            "",
            "Sliding Window Inference Contract",
            "_________________________________",
            *sliding_window_lines,
            "",
            "ONNX Runtime Example (Python)",
            "_____________________________",
            "import numpy as np",
            "import onnxruntime as ort",
            "",
            f"""sess = ort.InferenceSession(r\"{onnx_path}\", providers=['CPUExecutionProvider'])""",
            f"x = np.random.randn(1, {input_c}, {input_h}, {input_w}).astype(np.float32)",
            "logits = sess.run(['logits'], {'input': x})[0]",
            "pred = logits.argmax(axis=1).astype(np.int64)",
            "",
            "Sliding Window Pseudocode (NumPy)",
            "_________________________________",
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
            "_______________",
            str(model),
            "",
            "Config Snapshot",
            "_______________",
            f"num_classes: {int(self.config.data.num_classes)}",
            f"class_names: {class_names}",
            f"preprocess_mode: {preprocess_mode}",
            f"patch_size: {patch_size}",
            f"stride: {stride}",
            f"onnx_opset: {int(self.config.export.onnx_opset)}",
            "device_for_export: cpu",
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
