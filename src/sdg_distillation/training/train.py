from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Any, Literal

import hydra
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

try:
    import lightning as L
except ImportError:  # pragma: no cover - compatibility fallback
    import pytorch_lightning as L  # type: ignore[no-redef]

from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint


@dataclass(frozen=True)
class DataConfig:
    train_path: str
    test_path: str
    tokenizer_name: str
    val_path: str | None = None
    max_length: int = 512
    batch_size: int = 16
    num_workers: int = 4
    persistent_workers: bool = True
    pin_memory: bool = True
    drop_last: bool = False
    val_fraction: float = 0.1


@dataclass(frozen=True)
class RuntimeConfig:
    print_config: bool = True
    inspect_batch: bool = True
    inspect_batches: int = 1
    probe_batches: int = 50


@dataclass(frozen=True)
class ModelConfig:
    model_name: str = "microsoft/deberta-v3-base"
    num_labels: int = 17
    dropout: float = 0.1


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 2.0e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1


@dataclass(frozen=True)
class TrainerConfig:
    max_epochs: int = 3
    accelerator: str = "auto"
    devices: int = 1
    strategy: str | None = None
    precision: int | str = 32
    accumulate_grad_batches: int = 1
    gradient_clip_val: float | None = None
    gradient_clip_algorithm: str = "norm"
    log_every_n_steps: int = 50
    default_root_dir: str = "./outputs/train_runs"
    monitor: str = "val_loss"
    mode: str = "min"
    save_top_k: int = 1
    every_n_epochs: int = 1
    save_weights_only: bool = False
    run_test_after_fit: bool = True
    fast_dev_run: bool = False


@dataclass(frozen=True)
class DistillationConfig:
    loss_mode: str = "plain_bce"
    confidence_threshold: float = 0.0
    confidence_gamma: float = 1.0
    teacher_temperature: float = 1.0


@dataclass(frozen=True)
class InitializationConfig:
    checkpoint_path: str | None = None
    checkpoint_run_id: str | None = None


@dataclass(frozen=True)
class EvaluationConfig:
    checkpoint_path: str | None = None
    checkpoint_run_id: str | None = None
    split: str = "test"
    reference_train_path: str | None = None
    reference_val_path: str | None = None
    reference_test_path: str | None = None
    prediction_threshold: float = 0.5
    hard_label_threshold: float = 0.5
    calibration_bins: int = 10
    output_dir: str = "./outputs/evaluation"
    report_name: str | None = None
    save_predictions: bool = False
    sdgs_path: str = "./configs/sdgs.yaml"


def _ensure_config_sections(cfg: DictConfig) -> None:
    with open_dict(cfg):
        if "initialization" not in cfg:
            cfg.initialization = OmegaConf.structured(InitializationConfig())
        if "evaluation" not in cfg:
            cfg.evaluation = OmegaConf.structured(EvaluationConfig())


def _load_env() -> None:
    # Optional .env loading for local workflows (e.g., WANDB_API_KEY).
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        # Keep training entrypoint usable even if python-dotenv is absent.
        pass


def _build_datamodule(cfg: DictConfig):
    # Import lazily so config inspection still works in environments
    # where Lightning dependencies are not installed yet.
    from .datamodule import DistillationDataModule

    data_cfg = cfg.data
    return DistillationDataModule(
        train_path=Path(str(data_cfg.train_path)),
        test_path=Path(str(data_cfg.test_path)),
        val_path=Path(str(data_cfg.val_path)) if data_cfg.val_path is not None else None,
        reference_train_path=Path(str(cfg.evaluation.reference_train_path)) if cfg.evaluation.reference_train_path not in (None, "", "null") else None,
        reference_val_path=Path(str(cfg.evaluation.reference_val_path)) if cfg.evaluation.reference_val_path not in (None, "", "null") else None,
        reference_test_path=Path(str(cfg.evaluation.reference_test_path)) if cfg.evaluation.reference_test_path not in (None, "", "null") else None,
        tokenizer_name=str(data_cfg.tokenizer_name),
        target_mode=str(cfg.target_mode),
        max_length=int(data_cfg.max_length),
        batch_size=int(data_cfg.batch_size),
        num_workers=int(data_cfg.num_workers),
        persistent_workers=bool(data_cfg.persistent_workers),
        pin_memory=bool(data_cfg.pin_memory),
        drop_last=bool(data_cfg.drop_last),
        val_fraction=float(data_cfg.val_fraction),
        seed=int(cfg.seed),
    )


def _resolve_initialization_checkpoint_path(cfg: DictConfig) -> Path | None:
    checkpoint_path = cfg.initialization.checkpoint_path
    checkpoint_run_id = cfg.initialization.checkpoint_run_id

    if checkpoint_path not in (None, "", "null"):
        ckpt_path = Path(str(checkpoint_path))
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt_path)
        return ckpt_path

    if checkpoint_run_id not in (None, "", "null"):
        run_id = str(checkpoint_run_id)
        ckpt_dir = Path("./outputs/wandb/sdg-distillation") / run_id / "checkpoints"
        candidates = sorted(ckpt_dir.glob("*.ckpt"))
        if not candidates:
            raise FileNotFoundError(f"No checkpoints found under {ckpt_dir}")
        if len(candidates) > 1:
            raise RuntimeError(
                f"Multiple checkpoints found for run_id={run_id!r}; set initialization.checkpoint_path explicitly."
            )
        return candidates[0]

    return None


def _build_model(cfg: DictConfig):
    from .model import DistillationLitModule

    model_kwargs = dict(
        model_name=str(cfg.model.model_name),
        target_mode=str(cfg.target_mode),
        learning_rate=float(cfg.optimizer.learning_rate),
        weight_decay=float(cfg.optimizer.weight_decay),
        warmup_ratio=float(cfg.optimizer.warmup_ratio),
        dropout=float(cfg.model.dropout),
        distillation_loss_mode=str(cfg.distillation.loss_mode),
        confidence_threshold=float(cfg.distillation.confidence_threshold),
        confidence_gamma=float(cfg.distillation.confidence_gamma),
        teacher_temperature=float(cfg.distillation.teacher_temperature),
        validation_prediction_threshold=float(cfg.evaluation.prediction_threshold),
        validation_hard_label_threshold=float(cfg.evaluation.hard_label_threshold),
        num_labels=int(cfg.model.num_labels),
    )

    ckpt_path = _resolve_initialization_checkpoint_path(cfg)
    if ckpt_path is not None:
        return DistillationLitModule.load_from_checkpoint(
            str(ckpt_path),
            map_location="cpu",
            **model_kwargs,
        )

    return DistillationLitModule(**model_kwargs)


def _build_wandb_logger(cfg: DictConfig) -> Any | None:
    log_cfg = cfg.logging
    if not bool(log_cfg.enabled):
        return None

    # Import lazily to keep non-W&B workflows lightweight.
    try:
        from lightning.pytorch.loggers import WandbLogger
    except Exception:
        from pytorch_lightning.loggers import WandbLogger  # type: ignore[no-redef]

    project = str(log_cfg.project) if log_cfg.project is not None else None
    if project is None or project.strip() == "":
        raise ValueError("logging.project must be set when logging.enabled=true")

    tags = [str(t) for t in list(log_cfg.tags)] if log_cfg.tags is not None else []
    run_type = str(log_cfg.run_type).strip() if log_cfg.run_type is not None else ""
    if run_type and run_type not in tags:
        tags.append(run_type)

    name = None if log_cfg.name in (None, "null", "") else str(log_cfg.name)
    if name is None:
        model_short = str(cfg.model.model_name).split("/")[-1]
        if str(cfg.mode) == "evaluate":
            checkpoint_path = cfg.evaluation.checkpoint_path
            if checkpoint_path not in (None, "", "null"):
                ckpt_short = Path(str(checkpoint_path)).stem
            elif cfg.evaluation.checkpoint_run_id not in (None, "", "null"):
                ckpt_short = str(cfg.evaluation.checkpoint_run_id)
            else:
                ckpt_short = "checkpoint"
            name = f"eval__{ckpt_short}__{cfg.evaluation.split}"
        else:
            name = (
                f"{model_short}"
                f"__{cfg.target_mode}"
                f"__len{int(cfg.data.max_length)}"
                f"__bs{int(cfg.data.batch_size)}"
                f"__s{int(cfg.seed)}"
            )

    return WandbLogger(
        project=project,
        entity=None if log_cfg.entity in (None, "null", "") else str(log_cfg.entity),
        name=name,
        group=None if log_cfg.group in (None, "null", "") else str(log_cfg.group),
        tags=tags if tags else None,
        save_dir=str(log_cfg.save_dir),
        offline=bool(log_cfg.offline),
        log_model=log_cfg.log_model,
    )


def _build_callbacks(cfg: DictConfig) -> list[Any]:
    callbacks: list[Any] = []
    callbacks.append(LearningRateMonitor(logging_interval="step"))
    monitor_key = str(cfg.trainer.monitor)
    callbacks.append(
        ModelCheckpoint(
            monitor=monitor_key,
            mode=str(cfg.trainer.mode),
            save_top_k=int(cfg.trainer.save_top_k),
            every_n_epochs=int(cfg.trainer.every_n_epochs),
            save_weights_only=bool(cfg.trainer.save_weights_only),
            filename=f"epoch{{epoch:02d}}-{monitor_key}{{{monitor_key}:.4f}}",
        )
    )
    return callbacks


def _build_trainer(cfg: DictConfig, logger: Any | None) -> L.Trainer:
    trainer_kwargs: dict[str, Any] = {}

    strategy = cfg.trainer.strategy
    if strategy not in (None, "null", ""):
        trainer_kwargs["strategy"] = str(strategy)

    clip_val = cfg.trainer.gradient_clip_val
    if clip_val not in (None, "null"):
        trainer_kwargs["gradient_clip_val"] = float(clip_val)
        trainer_kwargs["gradient_clip_algorithm"] = str(cfg.trainer.gradient_clip_algorithm)

    return L.Trainer(
        max_epochs=int(cfg.trainer.max_epochs),
        accelerator=str(cfg.trainer.accelerator),
        devices=int(cfg.trainer.devices),
        precision=cfg.trainer.precision,
        accumulate_grad_batches=int(cfg.trainer.accumulate_grad_batches),
        log_every_n_steps=int(cfg.trainer.log_every_n_steps),
        default_root_dir=str(cfg.trainer.default_root_dir),
        callbacks=_build_callbacks(cfg),
        logger=logger,
        fast_dev_run=bool(cfg.trainer.fast_dev_run),
        **trainer_kwargs,
    )


def _get_git_metadata() -> dict[str, Any]:
    def _run(args: list[str]) -> str | None:
        try:
            out = subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True)
            return out.strip()
        except Exception:
            return None

    commit = _run(["git", "rev-parse", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])

    dirty: bool | None
    try:
        rc = subprocess.call(
            ["git", "diff", "--quiet"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        dirty = rc != 0
    except Exception:
        dirty = None

    return {
        "git_commit": commit,
        "git_branch": branch,
        "git_dirty": dirty,
    }


def _log_run_metadata(cfg: DictConfig, logger: Any | None) -> None:
    if logger is None:
        return

    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    git_meta = _get_git_metadata()

    # Log as hyperparams so this is queryable at run level in W&B.
    logger.log_hyperparams(
        {
            "resolved_config": resolved_cfg,
            **git_meta,
        }
    )


def _inspect_data(cfg: DictConfig, logger: Any | None = None) -> None:
    _log_run_metadata(cfg, logger)

    dm = _build_datamodule(cfg)
    dm.setup("fit")

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    n_train = len(dm.train_dataset) if dm.train_dataset is not None else 0
    n_val = len(dm.val_dataset) if dm.val_dataset is not None else 0

    print(f"Data summary: train_rows={n_train} val_rows={n_val}")
    print(
        "Loader summary: "
        f"train_batches={len(train_loader)} "
        + (f"val_batches={len(val_loader)}" if val_loader is not None else "val_batches=0")
    )

    if logger is not None and bool(cfg.logging.log_data_inspection):
        logger.log_hyperparams(
            {
                "target_mode": str(cfg.target_mode),
                "tokenizer_name": str(cfg.data.tokenizer_name),
                "batch_size": int(cfg.data.batch_size),
                "num_workers": int(cfg.data.num_workers),
                "persistent_workers": bool(cfg.data.persistent_workers),
                "drop_last": bool(cfg.data.drop_last),
                "max_length": int(cfg.data.max_length),
                "val_fraction": float(cfg.data.val_fraction),
            }
        )
        logger.log_metrics(
            {
                "inspect/train_rows": float(n_train),
                "inspect/val_rows": float(n_val),
                "inspect/train_batches": float(len(train_loader)),
                "inspect/val_batches": float(len(val_loader)) if val_loader is not None else 0.0,
            },
            step=0,
        )

    if bool(cfg.runtime.inspect_batch):
        k = max(1, int(cfg.runtime.inspect_batches))
        for i, batch in enumerate(train_loader):
            print(
                f"train_batch[{i}] "
                f"input_ids={tuple(batch['input_ids'].shape)} "
                f"attention_mask={tuple(batch['attention_mask'].shape)} "
                f"targets={tuple(batch['targets'].shape)}"
            )
            if i + 1 >= k:
                break

    # In inspect mode we do not create a Trainer, so close W&B run explicitly.
    if logger is not None:
        exp = getattr(logger, "experiment", None)
        if exp is not None and hasattr(exp, "finish"):
            exp.finish()


def _throughput_probe(cfg: DictConfig, logger: Any | None = None) -> None:
    import time

    dm = _build_datamodule(cfg)
    dm.setup("fit")

    model = _build_model(cfg)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    loader = dm.train_dataloader()
    max_batches = max(1, int(cfg.runtime.probe_batches))

    n_samples = 0
    n_tokens = 0
    n_batches = 0

    t0 = time.perf_counter()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            _ = model(input_ids, attention_mask)

            n_samples += int(input_ids.shape[0])
            n_tokens += int(attention_mask.sum().item())
            n_batches += 1

            if i + 1 >= max_batches:
                break

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = max(1e-9, time.perf_counter() - t0)

    samples_per_sec = n_samples / elapsed
    tokens_per_sec = n_tokens / elapsed
    peak_mem_mb = (
        float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
        if device.type == "cuda"
        else 0.0
    )

    print(
        "Throughput probe: "
        f"device={device.type} batches={n_batches} samples={n_samples} "
        f"elapsed_s={elapsed:.3f} samples_per_sec={samples_per_sec:.2f} "
        f"tokens_per_sec={tokens_per_sec:.2f} peak_mem_mb={peak_mem_mb:.1f}"
    )

    if logger is not None:
        _log_run_metadata(cfg, logger)
        logger.log_metrics(
            {
                "probe/batches": float(n_batches),
                "probe/samples": float(n_samples),
                "probe/elapsed_s": float(elapsed),
                "probe/samples_per_sec": float(samples_per_sec),
                "probe/tokens_per_sec": float(tokens_per_sec),
                "probe/peak_mem_mb": float(peak_mem_mb),
            },
            step=0,
        )
        exp = getattr(logger, "experiment", None)
        if exp is not None and hasattr(exp, "finish"):
            exp.finish()


def _train(cfg: DictConfig, logger: Any | None = None) -> None:
    _log_run_metadata(cfg, logger)

    dm = _build_datamodule(cfg)
    model = _build_model(cfg)
    trainer = _build_trainer(cfg, logger=logger)

    trainer.fit(model=model, datamodule=dm)
    if bool(cfg.trainer.run_test_after_fit):
        trainer.test(model=model, datamodule=dm)


def _evaluate(cfg: DictConfig, logger: Any | None = None) -> None:
    from .evaluation import evaluate_checkpoint, flatten_metrics_for_logging

    result = evaluate_checkpoint(cfg)

    if logger is not None:
        _log_run_metadata(cfg, logger)
        logger.log_hyperparams(
            {
                "evaluation/checkpoint_path": str(cfg.evaluation.checkpoint_path),
                "evaluation/requested_checkpoint_run_id": str(cfg.evaluation.checkpoint_run_id),
                "evaluation/checkpoint_run_id": result["checkpoint_run_id"],
                "evaluation/checkpoint_target_mode": result["checkpoint_target_mode"],
                "evaluation/checkpoint_model_name": result["checkpoint_model_name"],
                "evaluation/split": result["split"],
                "evaluation/reference_path": result["reference_path"],
                "evaluation/summary_path": str(result["summary_path"]),
                "evaluation/per_label_path": str(result["per_label_path"]),
            }
        )
        logger.log_metrics(
            flatten_metrics_for_logging(
                result["summary"],
                result["per_label_df"],
                split=str(result["split"]),
            ),
            step=0,
        )
        exp = getattr(logger, "experiment", None)
        if exp is not None and hasattr(exp, "finish"):
            exp.finish()


@hydra.main(version_base=None, config_path="../../../configs/train", config_name="config")
def main(cfg: DictConfig) -> None:
    _load_env()
    _ensure_config_sections(cfg)

    if bool(cfg.runtime.print_config):
        print(OmegaConf.to_yaml(cfg))

    L.seed_everything(int(cfg.seed), workers=True)
    logger = _build_wandb_logger(cfg)

    mode: Literal["inspect_data", "throughput_probe", "train", "evaluate"] = str(cfg.mode)  # type: ignore[assignment]
    if mode == "inspect_data":
        _inspect_data(cfg, logger=logger)
        return

    if mode == "throughput_probe":
        _throughput_probe(cfg, logger=logger)
        return

    if mode == "train":
        _train(cfg, logger=logger)
        return

    if mode == "evaluate":
        _evaluate(cfg, logger=logger)
        return

    raise ValueError(f"Unknown mode={cfg.mode!r}")


if __name__ == "__main__":
    main()
