from pathlib import Path
import json
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"
REPORTS_DIAGNOSTICS_DIR = PROJECT_ROOT / "reports" / "diagnostics"
MODELS_DIR = PROJECT_ROOT / "models"

X_PATH = PROCESSED_DIR / "target_class_l3_2_2048_X.npy"
Y_PATH = PROCESSED_DIR / "target_class_l3_2_2048_y.npy"
METADATA_PATH = PROCESSED_DIR / "target_class_l3_2_2048_metadata.csv"

VERSION = "v2_scaffold"
RUN_NAME = f"mlp_l3_2_2048_{VERSION}"

MODEL_PATH = MODELS_DIR / f"{RUN_NAME}_best_model.pt"
CONFIG_PATH = REPORTS_METRICS_DIR / f"{RUN_NAME}_config.json"
HISTORY_PATH = REPORTS_METRICS_DIR / f"{RUN_NAME}_training_history.csv"
FINAL_METRICS_PATH = REPORTS_METRICS_DIR / f"{RUN_NAME}_test_metrics.csv"
CLASSIFICATION_REPORT_PATH = REPORTS_METRICS_DIR / f"{RUN_NAME}_classification_report.csv"
CONFUSION_MATRIX_PATH = REPORTS_METRICS_DIR / f"{RUN_NAME}_confusion_matrix.csv"
CONFUSION_MATRIX_NORMALIZED_PATH = REPORTS_METRICS_DIR / f"{RUN_NAME}_confusion_matrix_normalized.csv"
PREDICTIONS_PATH = REPORTS_DIAGNOSTICS_DIR / f"{RUN_NAME}_test_predictions.csv"
SPLIT_PATH = PROCESSED_DIR / f"{RUN_NAME}_split_assignments.csv"
SCAFFOLD_STATS_PATH = REPORTS_METRICS_DIR / f"{RUN_NAME}_scaffold_stats.csv"
SPLIT_SUMMARY_PATH = REPORTS_METRICS_DIR / f"{RUN_NAME}_split_summary.csv"

TARGET_CLASS_COLUMN = "target_class_l3"
LABEL_COLUMN = "label_id"

RANDOM_STATE = 27

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15

INPUT_SIZE = 2048
HIDDEN_1 = 512
HIDDEN_2 = 256
HIDDEN_3 = 128

NEGATIVE_SLOPE = 0.01
DROPOUT_1 = 0.30
DROPOUT_2 = 0.30
DROPOUT_3 = 0.20

BATCH_SIZE = 256
MAX_EPOCHS = 150
EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_MIN_DELTA = 1e-6

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 5
SCHEDULER_MIN_LR = 1e-6

USE_CUDA = True
CUDA_DEVICE_ID = 0

# None means no artificial memory limit.
# PyTorch will be allowed to use all available GPU memory if needed.
CUDA_MEMORY_FRACTION = None

NUM_WORKERS = 0


class FingerprintDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, indices: np.ndarray):
        self.x = x
        self.y = y
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = self.indices[item]
        features = self.x[index]
        label = self.y[index]
        return features, label


class MLPClassifier(nn.Module):
    def __init__(self, input_size: int, n_classes: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, HIDDEN_1),
            nn.BatchNorm1d(HIDDEN_1),
            nn.LeakyReLU(negative_slope=NEGATIVE_SLOPE),
            nn.Dropout(DROPOUT_1),

            nn.Linear(HIDDEN_1, HIDDEN_2),
            nn.BatchNorm1d(HIDDEN_2),
            nn.LeakyReLU(negative_slope=NEGATIVE_SLOPE),
            nn.Dropout(DROPOUT_2),

            nn.Linear(HIDDEN_2, HIDDEN_3),
            nn.BatchNorm1d(HIDDEN_3),
            nn.LeakyReLU(negative_slope=NEGATIVE_SLOPE),
            nn.Dropout(DROPOUT_3),

            nn.Linear(HIDDEN_3, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def prepare_directories() -> None:
    REPORTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    if USE_CUDA and torch.cuda.is_available():
        device = torch.device(f"cuda:{CUDA_DEVICE_ID}")

        if CUDA_MEMORY_FRACTION is not None:
            try:
                torch.cuda.set_per_process_memory_fraction(
                    CUDA_MEMORY_FRACTION,
                    device=CUDA_DEVICE_ID,
                )
            except RuntimeError as error:
                print(f"CUDA memory fraction was not set: {error}")

        return device

    return torch.device("cpu")


def get_gpu_memory_mb(device: torch.device) -> tuple[float, float]:
    if device.type != "cuda":
        return 0.0, 0.0

    allocated = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2

    return float(allocated), float(reserved)


def load_data() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if not X_PATH.exists():
        raise FileNotFoundError(f"X file not found: {X_PATH}")

    if not Y_PATH.exists():
        raise FileNotFoundError(f"y file not found: {Y_PATH}")

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")

    x = np.load(X_PATH)
    y = np.load(Y_PATH)
    metadata = pd.read_csv(METADATA_PATH)

    if len(x) != len(y) or len(y) != len(metadata):
        raise ValueError(
            f"Length mismatch: X={len(x)}, y={len(y)}, metadata={len(metadata)}"
        )

    if x.shape[1] != INPUT_SIZE:
        raise ValueError(f"Expected input size {INPUT_SIZE}, got {x.shape[1]}")

    y = y.astype(np.int64)

    return x, y, metadata


def get_class_names(metadata: pd.DataFrame, n_classes: int) -> list[str]:
    label_map = (
        metadata[[LABEL_COLUMN, TARGET_CLASS_COLUMN]]
        .drop_duplicates()
        .sort_values(LABEL_COLUMN)
    )

    if len(label_map) != n_classes:
        raise ValueError(
            f"Expected {n_classes} class names, got {len(label_map)}"
        )

    return label_map[TARGET_CLASS_COLUMN].tolist()


def smiles_to_murcko_scaffold(smiles: str, row_index: int) -> str:
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return f"INVALID_SMILES_{row_index}"

    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol,
        includeChirality=False,
    )

    if scaffold is None or scaffold == "":
        return f"NO_SCAFFOLD_{row_index}"

    return scaffold


def build_scaffold_groups(
    metadata: pd.DataFrame,
    y: np.ndarray,
    n_classes: int,
) -> pd.DataFrame:
    scaffold_rows = []

    for row_index, smiles in enumerate(metadata["canonical_smiles"].tolist()):
        scaffold = smiles_to_murcko_scaffold(smiles, row_index)
        scaffold_rows.append(
            {
                "row_index": row_index,
                "scaffold": scaffold,
                "label_id": int(y[row_index]),
            }
        )

        if (row_index + 1) % 10000 == 0:
            print(f"Computed scaffolds for {row_index + 1}/{len(metadata)} molecules")

    scaffold_df = pd.DataFrame(scaffold_rows)

    group_rows = []

    for scaffold, group in scaffold_df.groupby("scaffold", sort=False):
        indices = group["row_index"].to_numpy(dtype=np.int64)
        labels = group["label_id"].to_numpy(dtype=np.int64)
        class_counts = np.bincount(labels, minlength=n_classes)

        group_row = {
            "scaffold": scaffold,
            "indices": indices,
            "size": len(indices),
            "majority_label_id": int(np.argmax(class_counts)),
            "n_classes_in_scaffold": int(np.count_nonzero(class_counts)),
        }

        for class_id in range(n_classes):
            group_row[f"class_{class_id}_count"] = int(class_counts[class_id])

        group_rows.append(group_row)

    scaffold_groups = pd.DataFrame(group_rows)

    scaffold_groups = scaffold_groups.sort_values(
        by=["size", "n_classes_in_scaffold", "scaffold"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    return scaffold_groups


def assign_scaffold_groups_to_splits(
    scaffold_groups: pd.DataFrame,
    y: np.ndarray,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    split_names = ["train", "validation", "test"]
    split_fractions = {
        "train": TRAIN_SIZE,
        "validation": VALIDATION_SIZE,
        "test": TEST_SIZE,
    }

    total_samples = len(y)
    total_class_counts = np.bincount(y, minlength=n_classes).astype(np.float64)

    target_total_by_split = {
        split: split_fractions[split] * total_samples
        for split in split_names
    }

    target_class_counts_by_split = {
        split: split_fractions[split] * total_class_counts
        for split in split_names
    }

    current_total_by_split = {
        split: 0.0
        for split in split_names
    }

    current_class_counts_by_split = {
        split: np.zeros(n_classes, dtype=np.float64)
        for split in split_names
    }

    assigned_indices = {
        split: []
        for split in split_names
    }

    scaffold_assignment_rows = []

    class_columns = [f"class_{class_id}_count" for class_id in range(n_classes)]

    for _, row in scaffold_groups.iterrows():
        group_indices = row["indices"]
        group_size = float(row["size"])
        group_class_counts = row[class_columns].to_numpy(dtype=np.float64)

        best_split = None
        best_score = np.inf

        for split in split_names:
            new_total = current_total_by_split[split] + group_size
            new_class_counts = (
                current_class_counts_by_split[split]
                + group_class_counts
            )

            total_target = target_total_by_split[split]
            class_target = target_class_counts_by_split[split]

            total_score = ((new_total - total_target) / max(total_target, 1.0)) ** 2

            class_score = np.mean(
                ((new_class_counts - class_target) / np.maximum(class_target, 1.0)) ** 2
            )

            overflow = max(0.0, new_total - total_target)
            overflow_score = (overflow / max(total_target, 1.0)) ** 2

            score = total_score + class_score + 3.0 * overflow_score

            if score < best_score:
                best_score = score
                best_split = split

        assigned_indices[best_split].extend(group_indices.tolist())
        current_total_by_split[best_split] += group_size
        current_class_counts_by_split[best_split] += group_class_counts

        scaffold_assignment_rows.append(
            {
                "scaffold": row["scaffold"],
                "split": best_split,
                "size": int(row["size"]),
                "majority_label_id": int(row["majority_label_id"]),
                "n_classes_in_scaffold": int(row["n_classes_in_scaffold"]),
                "assignment_score": float(best_score),
            }
        )

    train_indices = np.asarray(assigned_indices["train"], dtype=np.int64)
    validation_indices = np.asarray(assigned_indices["validation"], dtype=np.int64)
    test_indices = np.asarray(assigned_indices["test"], dtype=np.int64)

    scaffold_assignments = pd.DataFrame(scaffold_assignment_rows)

    return train_indices, validation_indices, test_indices, scaffold_assignments


def make_scaffold_splits(
    metadata: pd.DataFrame,
    y: np.ndarray,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    print("Computing Bemis-Murcko scaffolds...")
    scaffold_groups = build_scaffold_groups(metadata, y, n_classes)

    print()
    print(f"Unique scaffold groups: {len(scaffold_groups)}")
    print(f"Largest scaffold group size: {scaffold_groups['size'].max()}")
    print(f"Median scaffold group size: {scaffold_groups['size'].median():.1f}")
    print()

    train_indices, validation_indices, test_indices, scaffold_assignments = (
        assign_scaffold_groups_to_splits(
            scaffold_groups=scaffold_groups,
            y=y,
            n_classes=n_classes,
        )
    )

    scaffold_stats_export = scaffold_groups.drop(columns=["indices"]).merge(
        scaffold_assignments[["scaffold", "split", "assignment_score"]],
        on="scaffold",
        how="left",
    )

    scaffold_stats_export.to_csv(
        SCAFFOLD_STATS_PATH,
        index=False,
        encoding="utf-8",
    )

    return train_indices, validation_indices, test_indices, scaffold_stats_export


def save_split_assignments(
    metadata: pd.DataFrame,
    y: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
    class_names: list[str],
) -> None:
    split_data = metadata.copy()
    split_data["split"] = ""

    split_data.loc[train_indices, "split"] = "train"
    split_data.loc[validation_indices, "split"] = "validation"
    split_data.loc[test_indices, "split"] = "test"

    split_data.to_csv(SPLIT_PATH, index=False, encoding="utf-8")

    summary_rows = []

    for split_name, indices in [
        ("train", train_indices),
        ("validation", validation_indices),
        ("test", test_indices),
    ]:
        split_y = y[indices]
        class_counts = np.bincount(split_y, minlength=len(class_names))

        summary_rows.append(
            {
                "split": split_name,
                "samples": len(indices),
                "fraction": len(indices) / len(y),
                "unique_classes": int(np.count_nonzero(class_counts)),
            }
        )

        for class_id, class_name in enumerate(class_names):
            summary_rows.append(
                {
                    "split": split_name,
                    "class_id": class_id,
                    "class_name": class_name,
                    "samples": int(class_counts[class_id]),
                    "fraction_within_split": (
                        class_counts[class_id] / len(indices)
                        if len(indices) > 0
                        else 0.0
                    ),
                }
            )

    split_summary = pd.DataFrame(summary_rows)
    split_summary.to_csv(SPLIT_SUMMARY_PATH, index=False, encoding="utf-8")


def make_dataloaders(
    x: np.ndarray,
    y: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    pin_memory = device.type == "cuda"

    train_dataset = FingerprintDataset(x, y, train_indices)
    validation_dataset = FingerprintDataset(x, y, validation_indices)
    test_dataset = FingerprintDataset(x, y, test_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    return train_loader, train_eval_loader, validation_loader, test_loader


def compute_class_weights(y_train: np.ndarray, n_classes: int) -> np.ndarray:
    class_counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)

    if np.any(class_counts == 0):
        raise ValueError(f"Some classes are missing in train split: {class_counts}")

    total_samples = class_counts.sum()
    class_weights = total_samples / (n_classes * class_counts)

    return class_weights.astype(np.float32)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()

    total_loss = 0.0
    total_samples = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=True).float()
        y_batch = y_batch.to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)

        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        loss.backward()
        optimizer.step()

        batch_size = y_batch.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    return_predictions: bool = False,
) -> dict:
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_true = []
    all_pred = []
    all_prob = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=True).float()
        y_batch = y_batch.to(device, non_blocking=True).long()

        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)

        batch_size = y_batch.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_true.append(y_batch.detach().cpu().numpy())
        all_pred.append(predictions.detach().cpu().numpy())

        if return_predictions:
            all_prob.append(probabilities.detach().cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    metrics = {
        "loss": total_loss / total_samples,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_precision": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
    }

    if return_predictions:
        y_prob = np.vstack(all_prob)
        metrics["y_true"] = y_true
        metrics["y_pred"] = y_pred
        metrics["y_prob"] = y_prob

    return metrics


def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def save_config(
    device: torch.device,
    n_classes: int,
    class_names: list[str],
    class_weights: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
) -> None:
    config = {
        "run_name": RUN_NAME,
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "x_path": str(X_PATH),
        "y_path": str(Y_PATH),
        "metadata_path": str(METADATA_PATH),
        "input_size": INPUT_SIZE,
        "n_classes": n_classes,
        "class_names": class_names,
        "architecture": {
            "hidden_layers": [HIDDEN_1, HIDDEN_2, HIDDEN_3],
            "activation": "LeakyReLU",
            "negative_slope": NEGATIVE_SLOPE,
            "batch_norm_after_each_hidden_layer": True,
            "dropout": [DROPOUT_1, DROPOUT_2, DROPOUT_3],
            "output_layer": n_classes,
        },
        "training": {
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
            "loss": "CrossEntropyLoss with class weights",
            "primary_metric": "validation_macro_f1",
            "scheduler": "ReduceLROnPlateau",
            "scheduler_factor": SCHEDULER_FACTOR,
            "scheduler_patience": SCHEDULER_PATIENCE,
            "scheduler_min_lr": SCHEDULER_MIN_LR,
        },
        "split": {
            "split_type": "greedy_scaffold_split",
            "train_size_target": TRAIN_SIZE,
            "validation_size_target": VALIDATION_SIZE,
            "test_size_target": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "train_samples": len(train_indices),
            "validation_samples": len(validation_indices),
            "test_samples": len(test_indices),
        },
        "cuda": {
            "use_cuda": USE_CUDA,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_id": CUDA_DEVICE_ID,
            "cuda_memory_fraction": CUDA_MEMORY_FRACTION,
            "gpu_name": torch.cuda.get_device_name(CUDA_DEVICE_ID)
            if torch.cuda.is_available()
            else None,
        },
        "class_weights": class_weights.tolist(),
        "output_files": {
            "model_path": str(MODEL_PATH),
            "history_path": str(HISTORY_PATH),
            "final_metrics_path": str(FINAL_METRICS_PATH),
            "classification_report_path": str(CLASSIFICATION_REPORT_PATH),
            "confusion_matrix_path": str(CONFUSION_MATRIX_PATH),
            "predictions_path": str(PREDICTIONS_PATH),
            "split_path": str(SPLIT_PATH),
            "scaffold_stats_path": str(SCAFFOLD_STATS_PATH),
            "split_summary_path": str(SPLIT_SUMMARY_PATH),
        },
    }

    with open(CONFIG_PATH, "w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=4)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_macro_f1: float,
    class_names: list[str],
    config_path: Path,
) -> None:
    checkpoint = {
        "run_name": RUN_NAME,
        "version": VERSION,
        "epoch": epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "class_names": class_names,
        "input_size": INPUT_SIZE,
        "n_classes": len(class_names),
        "architecture": {
            "hidden_layers": [HIDDEN_1, HIDDEN_2, HIDDEN_3],
            "activation": "LeakyReLU",
            "negative_slope": NEGATIVE_SLOPE,
            "dropout": [DROPOUT_1, DROPOUT_2, DROPOUT_3],
        },
        "config_path": str(config_path),
    }

    torch.save(checkpoint, MODEL_PATH)


def save_final_outputs(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    metadata: pd.DataFrame,
    test_indices: np.ndarray,
    class_names: list[str],
    history: list[dict],
    best_epoch: int,
    best_val_macro_f1: float,
    total_training_time_minutes: float,
) -> None:
    history_df = pd.DataFrame(history)
    history_df.to_csv(HISTORY_PATH, index=False, encoding="utf-8")

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        return_predictions=True,
    )

    y_true = test_metrics.pop("y_true")
    y_pred = test_metrics.pop("y_pred")
    y_prob = test_metrics.pop("y_prob")

    try:
        test_log_loss = log_loss(
            y_true,
            y_prob,
            labels=list(range(len(class_names))),
        )
    except ValueError:
        test_log_loss = np.nan

    final_metrics = {
        "run_name": RUN_NAME,
        "version": VERSION,
        "split_type": "greedy_scaffold_split",
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "total_training_time_minutes": total_training_time_minutes,
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "test_macro_precision": test_metrics["macro_precision"],
        "test_macro_recall": test_metrics["macro_recall"],
        "test_log_loss": test_log_loss,
        "test_samples": len(y_true),
    }

    pd.DataFrame([final_metrics]).to_csv(
        FINAL_METRICS_PATH,
        index=False,
        encoding="utf-8",
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(CLASSIFICATION_REPORT_PATH, index=True, encoding="utf-8")

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=class_names,
        columns=class_names,
    )

    matrix_df.to_csv(CONFUSION_MATRIX_PATH, index=True, encoding="utf-8")

    with np.errstate(divide="ignore", invalid="ignore"):
        normalized_matrix = matrix.astype(np.float64) / matrix.sum(axis=1, keepdims=True)
        normalized_matrix = np.nan_to_num(normalized_matrix)

    normalized_matrix_df = pd.DataFrame(
        normalized_matrix,
        index=class_names,
        columns=class_names,
    )

    normalized_matrix_df.to_csv(
        CONFUSION_MATRIX_NORMALIZED_PATH,
        index=True,
        encoding="utf-8",
    )

    predictions = metadata.iloc[test_indices].copy().reset_index(drop=True)
    predictions["true_label_id"] = y_true
    predictions["predicted_label_id"] = y_pred
    predictions["predicted_target_class_l3"] = [class_names[label] for label in y_pred]
    predictions["predicted_probability"] = y_prob.max(axis=1)
    predictions["correct"] = y_true == y_pred

    for class_id, class_name in enumerate(class_names):
        safe_class_name = (
            class_name
            .replace(" ", "_")
            .replace("/", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("-", "_")
            .replace(",", "")
        )

        predictions[f"prob_{class_id}_{safe_class_name}"] = y_prob[:, class_id]

    predictions.to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8")

    print()
    print("Final test metrics")
    print("=" * 80)
    print(pd.DataFrame([final_metrics]).to_string(index=False))
    print("=" * 80)
    print()
    print(f"Training history saved to: {HISTORY_PATH}")
    print(f"Final test metrics saved to: {FINAL_METRICS_PATH}")
    print(f"Classification report saved to: {CLASSIFICATION_REPORT_PATH}")
    print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")
    print(f"Normalized confusion matrix saved to: {CONFUSION_MATRIX_NORMALIZED_PATH}")
    print(f"Test predictions saved to: {PREDICTIONS_PATH}")


def main() -> None:
    RDLogger.DisableLog("rdApp.*")

    prepare_directories()
    set_seed(RANDOM_STATE)

    device = get_device()

    print("MLP target-class training")
    print("=" * 80)
    print(f"Run name: {RUN_NAME}")
    print(f"Split type: greedy scaffold split")
    print(f"X path: {X_PATH}")
    print(f"y path: {Y_PATH}")
    print(f"Metadata path: {METADATA_PATH}")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU name: {torch.cuda.get_device_name(CUDA_DEVICE_ID)}")
        print("CUDA memory fraction: no artificial limit")

    print("=" * 80)
    print()

    x, y, metadata = load_data()

    n_classes = int(np.max(y)) + 1
    class_names = get_class_names(metadata, n_classes)

    print(f"X shape: {x.shape}")
    print(f"y shape: {y.shape}")
    print(f"Classes: {n_classes}")
    print()

    train_indices, validation_indices, test_indices, scaffold_stats = make_scaffold_splits(
        metadata=metadata,
        y=y,
        n_classes=n_classes,
    )

    save_split_assignments(
        metadata=metadata,
        y=y,
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
        class_names=class_names,
    )

    print("Split sizes")
    print("=" * 80)
    print(f"Train: {len(train_indices)} ({len(train_indices) / len(y):.3f})")
    print(f"Validation: {len(validation_indices)} ({len(validation_indices) / len(y):.3f})")
    print(f"Test: {len(test_indices)} ({len(test_indices) / len(y):.3f})")
    print(f"Scaffold stats saved to: {SCAFFOLD_STATS_PATH}")
    print(f"Split summary saved to: {SPLIT_SUMMARY_PATH}")
    print("=" * 80)
    print()

    train_loader, train_eval_loader, validation_loader, test_loader = make_dataloaders(
        x=x,
        y=y,
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
        device=device,
    )

    class_weights = compute_class_weights(y[train_indices], n_classes)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    print("Class weights")
    print("=" * 80)
    for class_id, class_name in enumerate(class_names):
        count = int(np.sum(y[train_indices] == class_id))
        print(
            f"{class_id:02d} | {class_name} | train_count={count} | "
            f"weight={class_weights[class_id]:.4f}"
        )
    print("=" * 80)
    print()

    model = MLPClassifier(
        input_size=INPUT_SIZE,
        n_classes=n_classes,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=SCHEDULER_MIN_LR,
    )

    save_config(
        device=device,
        n_classes=n_classes,
        class_names=class_names,
        class_weights=class_weights,
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
    )

    print("Model architecture")
    print("=" * 80)
    print(model)
    print("=" * 80)
    print()

    history = []

    best_val_macro_f1 = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0

    training_start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start_time = time.time()

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        train_epoch_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        train_metrics = evaluate(
            model=model,
            loader=train_eval_loader,
            criterion=criterion,
            device=device,
            return_predictions=False,
        )

        validation_metrics = evaluate(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
            return_predictions=False,
        )

        scheduler.step(validation_metrics["loss"])

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        epoch_time_seconds = time.time() - epoch_start_time
        total_elapsed_minutes = (time.time() - training_start_time) / 60
        gpu_allocated_mb, gpu_reserved_mb = get_gpu_memory_mb(device)

        current_lr = get_current_lr(optimizer)

        improved = (
            validation_metrics["macro_f1"]
            > best_val_macro_f1 + EARLY_STOPPING_MIN_DELTA
        )

        if improved:
            best_val_macro_f1 = validation_metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_macro_f1=best_val_macro_f1,
                class_names=class_names,
                config_path=CONFIG_PATH,
            )

            best_marker = " *best*"
        else:
            epochs_without_improvement += 1
            best_marker = ""

        epoch_row = {
            "epoch": epoch,
            "train_epoch_loss": train_epoch_loss,
            "train_loss": train_metrics["loss"],
            "val_loss": validation_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": validation_metrics["accuracy"],
            "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "val_balanced_accuracy": validation_metrics["balanced_accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_macro_f1": validation_metrics["macro_f1"],
            "train_weighted_f1": train_metrics["weighted_f1"],
            "val_weighted_f1": validation_metrics["weighted_f1"],
            "train_macro_precision": train_metrics["macro_precision"],
            "val_macro_precision": validation_metrics["macro_precision"],
            "train_macro_recall": train_metrics["macro_recall"],
            "val_macro_recall": validation_metrics["macro_recall"],
            "learning_rate": current_lr,
            "epoch_time_seconds": epoch_time_seconds,
            "total_elapsed_minutes": total_elapsed_minutes,
            "gpu_memory_allocated_mb": gpu_allocated_mb,
            "gpu_memory_reserved_mb": gpu_reserved_mb,
            "best_val_macro_f1": best_val_macro_f1,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
        }

        history.append(epoch_row)

        pd.DataFrame(history).to_csv(
            HISTORY_PATH,
            index=False,
            encoding="utf-8",
        )

        print(
            f"Epoch {epoch:03d}/{MAX_EPOCHS} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={validation_metrics['loss']:.4f} | "
            f"train_macro_f1={train_metrics['macro_f1']:.4f} | "
            f"val_macro_f1={validation_metrics['macro_f1']:.4f} | "
            f"val_acc={validation_metrics['accuracy']:.4f} | "
            f"val_bal_acc={validation_metrics['balanced_accuracy']:.4f} | "
            f"lr={current_lr:.6f} | "
            f"epoch_time={epoch_time_seconds:.1f}s | "
            f"elapsed={total_elapsed_minutes:.1f}min | "
            f"gpu_alloc={gpu_allocated_mb:.0f}MB | "
            f"gpu_reserved={gpu_reserved_mb:.0f}MB"
            f"{best_marker}"
        )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print()
            print(
                f"Early stopping triggered at epoch {epoch}. "
                f"Best epoch: {best_epoch}, "
                f"best validation macro F1: {best_val_macro_f1:.6f}"
            )
            break

    total_training_time_minutes = (time.time() - training_start_time) / 60

    print()
    print("Loading best model checkpoint...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    save_final_outputs(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        metadata=metadata,
        test_indices=test_indices,
        class_names=class_names,
        history=history,
        best_epoch=best_epoch,
        best_val_macro_f1=best_val_macro_f1,
        total_training_time_minutes=total_training_time_minutes,
    )

    print()
    print("Done.")
    print(f"Best model saved to: {MODEL_PATH}")
    print(f"Config saved to: {CONFIG_PATH}")
    print(f"Split assignments saved to: {SPLIT_PATH}")

    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()