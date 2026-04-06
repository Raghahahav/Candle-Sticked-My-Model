from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

try:
    from transformers import AutoImageProcessor, AutoModelForImageClassification
except ImportError as exc:  # pragma: no cover - only triggered before notebook installs deps
    AutoImageProcessor = None
    AutoModelForImageClassification = None
    TRANSFORMERS_IMPORT_ERROR = exc
else:
    TRANSFORMERS_IMPORT_ERROR = None


DEFAULT_CLASS_ORDER = ("down", "neutral", "up")


@dataclass
class TrainConfig:
    data_root: str | None = None
    metadata_csv: str | None = None
    checkpoint_dir: str = "/kaggle/working/candlestick_runs"
    model_family: str = "resnet50"  # "resnet50" or "deit"
    vit_checkpoint: str = "facebook/deit-base-patch16-224"
    class_names: tuple[str, ...] | None = None
    image_size: int = 320
    batch_size: int = 32
    epochs: int = 14
    patience: int = 4
    lr_backbone: float = 5e-5
    lr_head: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    num_workers: int = 2
    seed: int = 42
    window_size: int = 20
    purge_gap: int = 19
    crop_chart: bool = True
    border_ignore_px: int = 4
    background_threshold: int = 245
    crop_margin_px: int = 8
    weighted_sampler: bool = True
    amp: bool = True
    respect_existing_split: bool = True


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def detect_data_root(config: TrainConfig) -> Path:
    candidates: list[Path] = []
    if config.data_root:
        candidates.append(Path(config.data_root))

    repo_root = Path(__file__).resolve().parents[1]
    candidates.extend(
        [
            Path("/kaggle/input/datasets/romromkankane/stock-dataset/stock_dataset"),
            Path("/kaggle/input/stock-dataset/stock_dataset"),
            repo_root / "stock_dataset",
            Path.cwd() / "stock_dataset",
            Path.cwd().parent / "stock_dataset",
        ]
    )

    for root in candidates:
        if (root / "labels.csv").exists() and (root / "raw_images").exists():
            return root.resolve()

    joined = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(f"Could not locate stock_dataset. Tried:\n{joined}")


def infer_class_names(labels: pd.Series, preferred_order: tuple[str, ...] | None = None) -> list[str]:
    preferred = preferred_order or DEFAULT_CLASS_ORDER
    unique = [str(label) for label in pd.Index(labels).dropna().unique().tolist()]
    ordered = [label for label in preferred if label in unique]
    ordered.extend(sorted(label for label in unique if label not in ordered))
    return ordered


def _resolve_image_path(data_root: Path, image_path: str, stock: str) -> Path:
    filename = Path(image_path).name
    candidates = [
        Path(image_path),
        data_root / "raw_images" / stock / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Missing image for {stock}: {filename}")


def load_metadata(config: TrainConfig) -> pd.DataFrame:
    data_root = detect_data_root(config)
    csv_path = Path(config.metadata_csv) if config.metadata_csv else data_root / "labels.csv"
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["label"] = df["label"].str.lower()
    class_names = list(config.class_names) if config.class_names else infer_class_names(df["label"])
    label_to_id = {label: idx for idx, label in enumerate(class_names)}
    df = df[df["label"].isin(class_names)].copy()
    df["label_id"] = df["label"].map(label_to_id)
    df["image_idx"] = df["image_path"].str.extract(r"_(\d+)\.png$").astype(int)

    if "resolved_path" in df.columns:
        resolved_paths = []
        for image_path, resolved_path, stock in zip(df["image_path"], df["resolved_path"], df["stock"]):
            candidate = Path(str(resolved_path))
            if str(resolved_path).strip() and candidate.exists():
                resolved_paths.append(str(candidate.resolve()))
            else:
                resolved_paths.append(str(_resolve_image_path(data_root, image_path, stock)))
        df["resolved_path"] = resolved_paths
    else:
        df["resolved_path"] = [
            str(_resolve_image_path(data_root, image_path, stock))
            for image_path, stock in zip(df["image_path"], df["stock"])
        ]
    return df.sort_values(["stock", "timestamp", "image_idx"]).reset_index(drop=True)


def prepare_split_df(df: pd.DataFrame, config: TrainConfig) -> pd.DataFrame:
    if (
        config.respect_existing_split
        and "split" in df.columns
        and {"train", "val", "test"}.issubset(set(df["split"].dropna().unique()))
    ):
        return df[df["split"].isin(["train", "val", "test"])].copy().reset_index(drop=True)

    return build_purged_time_split(df, gap=config.purge_gap)


def build_purged_time_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    gap: int = 19,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []

    for stock, stock_df in df.groupby("stock", sort=False):
        stock_df = stock_df.sort_values(["timestamp", "image_idx"]).reset_index(drop=True).copy()
        n = len(stock_df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        val_start = min(train_end + gap, n)
        test_start = min(val_end + gap, n)

        stock_df["split"] = "gap"
        stock_df.loc[: train_end - 1, "split"] = "train"
        stock_df.loc[val_start : val_end - 1, "split"] = "val"
        stock_df.loc[test_start:, "split"] = "test"
        parts.append(stock_df)

    out = pd.concat(parts, ignore_index=True)
    return out[out["split"] != "gap"].reset_index(drop=True)


def dataset_diagnostics(df: pd.DataFrame, config: TrainConfig, class_names: list[str]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    report["rows"] = int(len(df))
    report["stocks"] = int(df["stock"].nunique())
    report["label_counts"] = df["label"].value_counts().to_dict()
    report["purge_gap"] = int(config.purge_gap)
    report["window_size"] = int(config.window_size)
    report["boundary_overlap_if_unpurged"] = max(config.window_size - 1, 0)

    stock_day = (
        df.assign(date=df["timestamp"].dt.date)
        .groupby(["stock", "date"])["label"]
        .nunique()
    )
    report["stock_days_with_multiple_labels_pct"] = round(float((stock_day > 1).mean() * 100), 2)

    flip_rates: list[float] = []
    for _, stock_df in df.groupby("stock", sort=False):
        stock_df = stock_df.sort_values("image_idx")
        if len(stock_df) < 2:
            continue
        flips = (stock_df["label"] != stock_df["label"].shift(1)).iloc[1:]
        flip_rates.append(float(flips.mean()))
    report["adjacent_window_flip_rate_pct"] = round(float(np.mean(flip_rates) * 100), 2)

    split_counts = df.groupby(["split", "label"]).size().unstack(fill_value=0).reindex(class_names, axis=1)
    report["split_counts"] = split_counts.to_dict(orient="index")
    return report


def print_diagnostics(report: dict[str, Any]) -> None:
    print(json.dumps(report, indent=2))


def crop_chart_region(
    image: Image.Image,
    border_ignore_px: int = 4,
    background_threshold: int = 245,
    crop_margin_px: int = 8,
) -> Image.Image:
    width, height = image.size
    if width <= 2 * border_ignore_px or height <= 2 * border_ignore_px:
        return image

    arr = np.array(image)
    inner = arr[border_ignore_px : height - border_ignore_px, border_ignore_px : width - border_ignore_px]
    if inner.size == 0:
        return image

    mask = ~(inner >= background_threshold).all(axis=2)
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return image.crop(
            (
                border_ignore_px,
                border_ignore_px,
                width - border_ignore_px,
                height - border_ignore_px,
            )
        )

    left = max(int(xs.min()) + border_ignore_px - crop_margin_px, 0)
    top = max(int(ys.min()) + border_ignore_px - crop_margin_px, 0)
    right = min(int(xs.max()) + border_ignore_px + crop_margin_px + 1, width)
    bottom = min(int(ys.max()) + border_ignore_px + crop_margin_px + 1, height)

    if right - left < 32 or bottom - top < 32:
        return image
    return image.crop((left, top, right, bottom))


class CandlestickDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform: transforms.Compose) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.frame.iloc[index]
        image = Image.open(row["resolved_path"]).convert("RGB")
        image = self.transform(image)
        return image, int(row["label_id"])


def get_image_stats(frame: pd.DataFrame, sample_size: int = 64) -> dict[str, float]:
    if frame.empty:
        return {"white_fraction_median": float("nan")}

    sampled = frame.sample(n=min(sample_size, len(frame)), random_state=42)
    white_fracs: list[float] = []
    for path in sampled["resolved_path"]:
        arr = np.array(Image.open(path).convert("RGB"))
        white_fracs.append(float((arr > 245).all(axis=2).mean()))

    return {
        "white_fraction_min": round(float(np.min(white_fracs)), 4),
        "white_fraction_median": round(float(np.median(white_fracs)), 4),
        "white_fraction_max": round(float(np.max(white_fracs)), 4),
    }


def get_normalization(config: TrainConfig) -> tuple[list[float], list[float]]:
    if config.model_family == "resnet50":
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    if AutoImageProcessor is None:
        raise ImportError(
            "transformers is required for DeiT training. Install it in Kaggle first."
        ) from TRANSFORMERS_IMPORT_ERROR

    processor = AutoImageProcessor.from_pretrained(config.vit_checkpoint)
    return list(processor.image_mean), list(processor.image_std)


def build_transforms(config: TrainConfig) -> tuple[transforms.Compose, transforms.Compose]:
    mean, std = get_normalization(config)
    crop = transforms.Lambda(
        lambda image: crop_chart_region(
            image,
            border_ignore_px=config.border_ignore_px,
            background_threshold=config.background_threshold,
            crop_margin_px=config.crop_margin_px,
        )
        if config.crop_chart
        else image
    )

    common = [
        crop,
        transforms.Resize(
            (config.image_size, config.image_size),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    train_transform = transforms.Compose(
        [
            crop,
            transforms.Resize(
                (config.image_size, config.image_size),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ColorJitter(brightness=0.04, contrast=0.04),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_transform = transforms.Compose(common)
    return train_transform, eval_transform


def build_dataloaders(
    split_df: pd.DataFrame,
    config: TrainConfig,
) -> tuple[DataLoader, DataLoader, DataLoader, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = split_df[split_df["split"] == "train"].copy()
    val_df = split_df[split_df["split"] == "val"].copy()
    test_df = split_df[split_df["split"] == "test"].copy()

    train_tf, eval_tf = build_transforms(config)
    train_ds = CandlestickDataset(train_df, train_tf)
    val_ds = CandlestickDataset(val_df, eval_tf)
    test_ds = CandlestickDataset(test_df, eval_tf)

    train_sampler = None
    shuffle = True
    if config.weighted_sampler:
        counts = train_df["label"].value_counts()
        sample_weights = train_df["label"].map(lambda label: 1.0 / counts[label]).to_numpy()
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False

    loader_kwargs = dict(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )
    train_loader = DataLoader(
        train_ds,
        shuffle=shuffle,
        sampler=train_sampler,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader, train_df, val_df, test_df


def _set_trainable(module: nn.Module, enabled: bool) -> None:
    for param in module.parameters():
        param.requires_grad = enabled


def build_model(config: TrainConfig, device: torch.device, num_classes: int, class_names: list[str]) -> nn.Module:
    if config.model_family == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        _set_trainable(model, False)
        _set_trainable(model.layer3, True)
        _set_trainable(model.layer4, True)

        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.35),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )
        return model.to(device)

    if config.model_family == "deit":
        if AutoModelForImageClassification is None:
            raise ImportError(
                "transformers is required for DeiT training. Install it in Kaggle first."
            ) from TRANSFORMERS_IMPORT_ERROR

        model = AutoModelForImageClassification.from_pretrained(
            config.vit_checkpoint,
            num_labels=num_classes,
            id2label={idx: label for idx, label in enumerate(class_names)},
            label2id={label: idx for idx, label in enumerate(class_names)},
            ignore_mismatched_sizes=True,
        )
        _set_trainable(model, False)

        encoder = getattr(model, "deit", None) or getattr(model, "vit", None)
        if encoder is None:
            raise ValueError(f"Unsupported DeiT-style model: {config.vit_checkpoint}")

        blocks = encoder.encoder.layer
        for block in blocks[-6:]:
            _set_trainable(block, True)

        layernorm = getattr(encoder, "layernorm", None)
        if layernorm is not None:
            _set_trainable(layernorm, True)

        _set_trainable(model.classifier, True)
        return model.to(device)

    raise ValueError(f"Unknown model_family: {config.model_family}")


def build_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    head_names = ("fc.", "classifier.")
    head_params: list[nn.Parameter] = []
    backbone_params: list[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(head_names):
            head_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": config.lr_backbone})
    if head_params:
        param_groups.append({"params": head_params, "lr": config.lr_head})

    return torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)


def forward_logits(model: nn.Module, images: torch.Tensor, config: TrainConfig) -> torch.Tensor:
    if config.model_family == "deit":
        return model(pixel_values=images).logits
    return model(images)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    scaler: GradScaler,
    config: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    preds: list[int] = []
    targets: list[int] = []
    amp_enabled = bool(config.amp and device.type == "cuda")

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            logits = forward_logits(model, images, config)
            loss = criterion(logits, labels)

        if train_mode and optimizer is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item() * images.size(0)
        preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())

    return {
        "loss": total_loss / max(len(loader.dataset), 1),
        "accuracy": accuracy_score(targets, preds),
        "f1_macro": f1_score(targets, preds, average="macro"),
    }


@torch.no_grad()
def evaluate_with_outputs(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    config: TrainConfig,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    preds: list[int] = []
    targets: list[int] = []
    amp_enabled = bool(config.amp and device.type == "cuda")

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            logits = forward_logits(model, images, config)
            loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds.extend(logits.argmax(dim=1).cpu().tolist())
        targets.extend(labels.cpu().tolist())

    metrics = {
        "loss": total_loss / max(len(loader.dataset), 1),
        "accuracy": accuracy_score(targets, preds),
        "f1_macro": f1_score(targets, preds, average="macro"),
    }
    return metrics, np.asarray(preds), np.asarray(targets)


def train_model(config: TrainConfig) -> dict[str, Any]:
    seed_everything(config.seed)
    device = get_device()
    metadata_df = load_metadata(config)
    class_names = list(config.class_names) if config.class_names else infer_class_names(metadata_df["label"])
    split_df = prepare_split_df(metadata_df, config)
    diagnostics = dataset_diagnostics(split_df, config, class_names)
    diagnostics["image_stats"] = get_image_stats(split_df)

    train_loader, val_loader, test_loader, train_df, val_df, test_df = build_dataloaders(split_df, config)
    model = build_model(config, device, num_classes=len(class_names), class_names=class_names)
    optimizer = build_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config.amp and device.type == "cuda"))

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    label_tag = "-".join(class_names)
    run_name = f"{config.model_family}_{config.image_size}px_gap{config.purge_gap}_{label_tag}"
    checkpoint_path = checkpoint_dir / f"{run_name}.pt"

    history: list[dict[str, float]] = []
    best_state = None
    best_val_acc = -math.inf
    patience_left = config.patience

    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, criterion, scaler, config, device)
        val_metrics = run_epoch(model, val_loader, None, criterion, scaler, config, device)
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "train_f1": train_metrics["f1_macro"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_f1": val_metrics["f1_macro"],
        }
        history.append(row)
        print(
            f"[{epoch:02d}/{config.epochs}] "
            f"train_acc={row['train_acc']:.4f} val_acc={row['val_acc']:.4f} "
            f"train_f1={row['train_f1']:.4f} val_f1={row['val_f1']:.4f} lr={row['lr']:.2e}"
        )

        if row["val_acc"] > best_val_acc:
            best_val_acc = row["val_acc"]
            patience_left = config.patience
            best_state = {
                "epoch": epoch,
                "model_state": copy.deepcopy(model.state_dict()),
                "config": asdict(config),
                "diagnostics": diagnostics,
                "class_names": class_names,
            }
            torch.save(best_state, checkpoint_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    model.load_state_dict(best_state["model_state"])
    val_metrics, val_preds, val_targets = evaluate_with_outputs(model, val_loader, criterion, config, device)
    test_metrics, test_preds, test_targets = evaluate_with_outputs(model, test_loader, criterion, config, device)

    report = {
        "run_name": run_name,
        "checkpoint_path": str(checkpoint_path),
        "device": str(device),
        "config": asdict(config),
        "diagnostics": diagnostics,
        "splits": {
            "train": train_df["label"].value_counts().to_dict(),
            "val": val_df["label"].value_counts().to_dict(),
            "test": test_df["label"].value_counts().to_dict(),
        },
        "history": history,
        "val": {
            **val_metrics,
            "classification_report": classification_report(
                val_targets,
                val_preds,
                labels=list(range(len(class_names))),
                target_names=class_names,
                digits=4,
                output_dict=True,
            ),
        },
        "test": {
            **test_metrics,
            "classification_report": classification_report(
                test_targets,
                test_preds,
                labels=list(range(len(class_names))),
                target_names=class_names,
                digits=4,
                output_dict=True,
            ),
            "confusion_matrix": confusion_matrix(
                test_targets,
                test_preds,
                labels=list(range(len(class_names))),
            ).tolist(),
        },
    }

    summary_path = checkpoint_dir / f"{run_name}_summary.json"
    summary_path.write_text(json.dumps(report, indent=2))
    print(f"Saved summary to {summary_path}")
    return report


def plot_history(report: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    history = pd.DataFrame(report["history"])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(history["train_loss"], label="Train", lw=2)
    axes[0].plot(history["val_loss"], label="Val", lw=2)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["train_acc"], label="Train", lw=2)
    axes[1].plot(history["val_acc"], label="Val", lw=2)
    axes[1].axhline(1 / 3, color="red", linestyle="--", alpha=0.5, label="Random")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(history["lr"], color="orange", lw=2)
    axes[2].set_title("Learning Rate")
    axes[2].set_xlabel("Epoch")
    axes[2].set_yscale("log")
    axes[2].grid(alpha=0.3)

    fig.suptitle(report["run_name"], fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_confusion(report: dict[str, Any], split: str = "test") -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    class_names = list(report["config"].get("class_names") or infer_class_names(pd.Series(report["splits"]["train"].keys())))
    cm = np.asarray(report[split]["confusion_matrix"])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title(f"{split.title()} Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1],
    )
    axes[1].set_title(f"{split.title()} Confusion Matrix (normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    plt.tight_layout()
    plt.show()


def compare_reports(reports: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for report in reports:
        rows.append(
            {
                "run_name": report["run_name"],
                "val_acc": report["val"]["accuracy"],
                "val_f1": report["val"]["f1_macro"],
                "test_acc": report["test"]["accuracy"],
                "test_f1": report["test"]["f1_macro"],
            }
        )
    return pd.DataFrame(rows).sort_values(["test_acc", "test_f1"], ascending=False).reset_index(drop=True)
