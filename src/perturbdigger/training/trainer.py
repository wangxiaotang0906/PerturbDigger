from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from perturbdigger.config import ensure_output_dirs
from perturbdigger.data.dataset import DatasetBundle, PerturbationDataset
from perturbdigger.explain.subgraph import aggregate_explanations, build_sample_explanations
from perturbdigger.graph.calibration import GraphCalibrationModel, compute_tf_relevance_prior
from perturbdigger.graph.hetero_graph import GraphSpecification, LearnableEdgeWeights, build_graph_specification
from perturbdigger.model.perturbdigger import PerturbationResponseModel
from perturbdigger.utils import dump_json, dump_jsonl, get_device_summary, resolve_device, set_seed


@dataclass
class EpochResult:
    loss: float
    metrics: Dict[str, float]


def _collate_batch(batch: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    return {
        key: torch.as_tensor(np.stack([item[key] for item in batch]), dtype=torch.float32)
        if key != "index"
        else torch.as_tensor(np.stack([item[key] for item in batch]), dtype=torch.long)
        for key in batch[0]
    }


def _build_loader(
    bundle: DatasetBundle,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    pin_memory: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        PerturbationDataset(bundle, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        collate_fn=_collate_batch,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )


def _random_mask_like(x: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    mask = torch.rand_like(x) < mask_ratio
    empty_rows = ~mask.any(dim=1)
    if bool(empty_rows.any()):
        random_cols = torch.randint(0, x.shape[1], size=(int(empty_rows.sum().item()),), device=x.device)
        row_indices = torch.nonzero(empty_rows, as_tuple=False).squeeze(-1)
        mask[row_indices, random_cols] = True
    return mask


def _maybe_subsample_indices(
    indices: np.ndarray,
    max_samples: int | None,
    seed: int,
) -> np.ndarray:
    if max_samples is None or max_samples <= 0 or len(indices) <= max_samples:
        return indices
    rng = np.random.default_rng(seed)
    sampled = rng.choice(indices, size=max_samples, replace=False)
    sampled.sort()
    return sampled.astype(np.int64)


class ExperimentRunner:
    def __init__(self, config: Dict[str, object]):
        self.config = config
        set_seed(int(config["training"]["seed"]))
        self.device = resolve_device(str(config["training"].get("device", "auto")))
        self.show_progress = bool(config["training"].get("show_progress", True))
        self.pin_memory = bool(config["training"].get("pin_memory", self.device.type == "cuda"))
        self.num_workers = int(config["training"].get("num_workers", 0))
        self.output_dirs = ensure_output_dirs(config)
        self.device_summary = get_device_summary(self.device)

    def run(self, bundle: DatasetBundle) -> Dict[str, Dict[str, float]]:
        graph = build_graph_specification(bundle, self.device)
        calibrated_weights, calibration_metrics = self._run_calibration(bundle, graph)
        predictor, training_metrics = self._run_perturbation_training(bundle, graph, calibrated_weights)
        explanation_metrics = self._export_explanations(bundle, predictor)

        summary = {
            "system": self.device_summary,
            "calibration": calibration_metrics,
            "training": training_metrics,
            "explanations": explanation_metrics,
        }
        dump_json(summary, self.output_dirs["logs"] / "summary.json")
        return summary

    def _run_calibration(
        self,
        bundle: DatasetBundle,
        graph: GraphSpecification,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        calibration_cfg = self.config["training"]["calibration"]
        edge_weights = LearnableEdgeWeights(graph).to(self.device)
        q_prior_np = compute_tf_relevance_prior(
            control_x=bundle.x_rows(bundle.control_indices()),
            tg_edge_index=bundle.edge_indices["tg"],
            num_genes=bundle.num_genes,
            alpha=float(calibration_cfg["lasso_alpha"]),
            show_progress=self.show_progress,
        )
        q_prior = torch.as_tensor(q_prior_np, dtype=torch.float32, device=self.device)
        model = GraphCalibrationModel(graph, edge_weights, q_prior, self.config["model"]).to(self.device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(calibration_cfg["lr"]),
            weight_decay=float(calibration_cfg.get("weight_decay", 0.0)),
        )

        loader = _build_loader(
            bundle,
            _maybe_subsample_indices(
                bundle.control_indices(),
                calibration_cfg.get("max_control_samples"),
                seed=int(self.config["training"]["seed"]),
            ),
            batch_size=int(calibration_cfg["batch_size"]),
            shuffle=True,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )
        last_loss = 0.0
        mask_ratio = float(calibration_cfg["mask_ratio"])
        lambda_cal = float(calibration_cfg["lambda_cal"])

        model.train()
        num_epochs = int(calibration_cfg["epochs"])
        epoch_iterator = range(num_epochs)
        if self.show_progress:
            epoch_iterator = tqdm(epoch_iterator, desc="Calibration", unit="epoch")

        for epoch_idx in epoch_iterator:
            batch_iterator = loader
            if self.show_progress:
                batch_iterator = tqdm(loader, desc=f"Calib {epoch_idx + 1}/{num_epochs}", leave=False, unit="batch")
            for batch in batch_iterator:
                x = batch["x"].to(self.device, non_blocking=self.pin_memory)
                mask = _random_mask_like(x, mask_ratio)
                output = model(x, mask=mask)
                rec_loss = ((output["x_hat"] - x) ** 2)[mask].mean()
                reg_loss = model.graph_regularization()
                loss = rec_loss + lambda_cal * reg_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                last_loss = float(loss.item())
                if self.show_progress:
                    batch_iterator.set_postfix(loss=f"{loss.item():.4f}", rec=f"{rec_loss.item():.4f}", reg=f"{reg_loss.item():.4f}")

        metrics = {
            "graph_loss": last_loss,
            "num_control_samples": float(bundle.control_indices().shape[0]),
            "num_control_samples_used": float(len(loader.dataset)),
        }
        calibrated_weights = edge_weights.detached()
        torch.save(
            {
                "edge_weights": {name: value.cpu() for name, value in calibrated_weights.items()},
                "q_tg_prior": q_prior.cpu(),
            },
            self.output_dirs["checkpoints"] / "graph_calibration.pt",
        )
        return calibrated_weights, metrics

    def _run_perturbation_training(
        self,
        bundle: DatasetBundle,
        graph: GraphSpecification,
        calibrated_weights: Dict[str, torch.Tensor],
    ) -> Tuple[PerturbationResponseModel, Dict[str, float]]:
        training_cfg = self.config["training"]["perturbation"]
        model = PerturbationResponseModel(graph, self.config["model"], calibrated_weights).to(self.device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(training_cfg["lr"]),
            weight_decay=float(training_cfg.get("weight_decay", 0.0)),
        )

        train_indices = _maybe_subsample_indices(
            bundle.perturbation_indices("train"),
            training_cfg.get("max_train_samples"),
            seed=int(self.config["training"]["seed"]),
        )
        val_indices = _maybe_subsample_indices(
            bundle.perturbation_indices("val"),
            training_cfg.get("max_val_samples"),
            seed=int(self.config["training"]["seed"]) + 1,
        )
        test_indices = _maybe_subsample_indices(
            bundle.perturbation_indices("test"),
            training_cfg.get("max_test_samples"),
            seed=int(self.config["training"]["seed"]) + 2,
        )

        train_loader = _build_loader(
            bundle,
            train_indices,
            batch_size=int(training_cfg["batch_size"]),
            shuffle=True,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )
        val_loader = _build_loader(
            bundle,
            val_indices,
            batch_size=int(training_cfg["batch_size"]),
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

        best_state = None
        best_val = float("inf")
        patience = 0
        lambda_nec = float(training_cfg["lambda_nec"])
        gamma = float(training_cfg["gamma"])
        max_patience = int(training_cfg["early_stop_patience"])

        num_epochs = int(training_cfg["epochs"])
        epoch_iterator = range(num_epochs)
        if self.show_progress:
            epoch_iterator = tqdm(epoch_iterator, desc="Training", unit="epoch")

        for epoch_idx in epoch_iterator:
            model.train()
            batch_iterator = train_loader
            if self.show_progress:
                batch_iterator = tqdm(train_loader, desc=f"Train {epoch_idx + 1}/{num_epochs}", leave=False, unit="batch")
            for batch in batch_iterator:
                x = batch["x"].to(self.device, non_blocking=self.pin_memory)
                y = batch["y"].to(self.device, non_blocking=self.pin_memory)
                z = batch["z"].to(self.device, non_blocking=self.pin_memory)
                delta = y - x

                output = model(x, z, return_subgraph=True)
                suf_loss = ((output["delta_hat"] - delta) ** 2).mean()

                blocked = {
                    "tg": (output["tg_selected"] > 0).float().detach(),
                    "gp": (output["gp_selected"] > 0).float().detach(),
                }
                ablated = model(x, z, blocked_mediator_masks=blocked, return_subgraph=False)
                nec_term = gamma - torch.norm(ablated["delta_hat"] - delta, dim=1) + torch.norm(output["delta_hat"] - delta, dim=1)
                nec_loss = torch.relu(nec_term).mean()
                loss = suf_loss + lambda_nec * nec_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if self.show_progress:
                    batch_iterator.set_postfix(loss=f"{loss.item():.4f}", suf=f"{suf_loss.item():.4f}", nec=f"{nec_loss.item():.4f}")

            val_metrics = self._evaluate_loader(model, val_loader, desc=f"Val {epoch_idx + 1}/{num_epochs}")
            val_score = val_metrics["mse"]
            if np.isnan(val_score):
                val_score = self._evaluate_loader(model, train_loader, desc="Train Eval")["mse"]
            if val_score < best_val:
                best_val = val_score
                best_state = {
                    "model": model.state_dict(),
                    "metrics": val_metrics,
                }
                patience = 0
            else:
                patience += 1
                if patience >= max_patience:
                    break
            if self.show_progress:
                epoch_iterator.set_postfix(val_mse=f"{val_score:.4f}", patience=patience)

        if best_state is not None:
            model.load_state_dict(best_state["model"])
        else:
            best_val = self._evaluate_loader(model, train_loader, desc="Train Eval")["mse"]

        test_loader = _build_loader(
            bundle,
            test_indices,
            batch_size=int(training_cfg["batch_size"]),
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )
        test_metrics = self._evaluate_loader(model, test_loader, desc="Test")
        train_metrics = self._evaluate_loader(model, train_loader, desc="Train Eval")
        metrics = {
            "train_mse": train_metrics["mse"],
            "val_mse": best_val,
            "test_mse": test_metrics["mse"],
            "test_mae": test_metrics["mae"],
            "num_train_samples_used": float(len(train_indices)),
            "num_val_samples_used": float(len(val_indices)),
            "num_test_samples_used": float(len(test_indices)),
        }

        torch.save(
            {
                "model_state": model.state_dict(),
                "graph_weights": {name: value.cpu() for name, value in calibrated_weights.items()},
                "config": self.config,
                "metrics": metrics,
            },
            self.output_dirs["checkpoints"] / "perturbdigger.pt",
        )
        return model, metrics

    def _evaluate_loader(self, model: PerturbationResponseModel, loader: DataLoader, desc: str | None = None) -> Dict[str, float]:
        model.eval()
        preds = []
        truths = []
        iterator = loader
        if self.show_progress and desc is not None:
            iterator = tqdm(loader, desc=desc, leave=False, unit="batch")
        with torch.no_grad():
            for batch in iterator:
                x = batch["x"].to(self.device, non_blocking=self.pin_memory)
                y = batch["y"].to(self.device, non_blocking=self.pin_memory)
                z = batch["z"].to(self.device, non_blocking=self.pin_memory)
                delta = y - x
                output = model(x, z, return_subgraph=False)
                preds.append(output["delta_hat"])
                truths.append(delta)
        if not preds:
            return {"mse": float("nan"), "mae": float("nan")}
        pred = torch.cat(preds, dim=0)
        truth = torch.cat(truths, dim=0)
        mse = torch.mean((pred - truth) ** 2).item()
        mae = torch.mean(torch.abs(pred - truth)).item()
        return {"mse": float(mse), "mae": float(mae)}

    def _export_explanations(
        self,
        bundle: DatasetBundle,
        model: PerturbationResponseModel,
    ) -> Dict[str, float]:
        explain_cfg = self.config["explanations"]
        indices = _maybe_subsample_indices(
            bundle.perturbation_indices(explain_cfg["split"]),
            explain_cfg.get("max_samples"),
            seed=int(self.config["training"]["seed"]) + 3,
        )
        loader = _build_loader(
            bundle,
            indices,
            batch_size=int(explain_cfg["batch_size"]),
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )
        model.eval()

        sample_records = []
        iterator = loader
        if self.show_progress:
            iterator = tqdm(loader, desc="Explain", leave=False, unit="batch")
        with torch.no_grad():
            for batch in iterator:
                sample_indices = batch["index"].to(self.device, non_blocking=self.pin_memory)
                x = batch["x"].to(self.device, non_blocking=self.pin_memory)
                z = batch["z"].to(self.device, non_blocking=self.pin_memory)
                output = model(x, z, return_subgraph=True)
                sample_records.extend(
                    build_sample_explanations(
                        bundle=bundle,
                        graph=model.graph,
                        sample_indices=sample_indices.cpu(),
                        tg_selected=output["tg_selected"].cpu(),
                        gp_selected=output["gp_selected"].cpu(),
                        topn=int(explain_cfg["top_edges_per_sample"]),
                    )
                )

        aggregated = aggregate_explanations(sample_records, topn=int(explain_cfg["top_edges_per_condition"]))
        dump_jsonl(sample_records, self.output_dirs["explanations"] / "sample_level.jsonl")
        dump_json({"conditions": aggregated}, self.output_dirs["explanations"] / "perturbation_level.json")
        return {
            "num_sample_explanations": float(len(sample_records)),
            "num_condition_explanations": float(len(aggregated)),
            "num_explanation_samples_used": float(len(indices)),
        }
