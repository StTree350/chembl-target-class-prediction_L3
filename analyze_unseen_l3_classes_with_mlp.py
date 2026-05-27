from pathlib import Path
import math
import time
import json

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs


PROJECT_ROOT = Path(__file__).resolve().parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"
REPORTS_DIAGNOSTICS_DIR = PROJECT_ROOT / "reports" / "diagnostics"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_CSV_PATH = RAW_DIR / "chembl_target_class_activities_large.csv"

KNOWN_DATASET_PATH = PROCESSED_DIR / "target_class_l3_single_label_dataset.csv"
KNOWN_X_PATH = PROCESSED_DIR / "target_class_l3_2_2048_X.npy"
KNOWN_Y_PATH = PROCESSED_DIR / "target_class_l3_2_2048_y.npy"
KNOWN_METADATA_PATH = PROCESSED_DIR / "target_class_l3_2_2048_metadata.csv"

MIN_UNIQUE_MOLECULES = 100
EXCLUDE_MOLECULES_FROM_KNOWN_DATASET = True

RANDOM_STATE = 27

FINGERPRINT_RADIUS = 2
FINGERPRINT_SIZE = 2048

BATCH_SIZE = 512
USE_CUDA = True
CUDA_DEVICE_ID = 0

INPUT_SIZE = 2048
HIDDEN_1 = 512
HIDDEN_2 = 256
HIDDEN_3 = 128
NEGATIVE_SLOPE = 0.01
DROPOUT_1 = 0.30
DROPOUT_2 = 0.30
DROPOUT_3 = 0.20

RUNS = [
    {
        "model_name": "v1_random",
        "checkpoint_path": MODELS_DIR / "mlp_l3_2_2048_v1_best_model.pt",
        "split_path": PROCESSED_DIR / "mlp_l3_2_2048_v1_split_assignments.csv",
    },
    {
        "model_name": "v2_scaffold",
        "checkpoint_path": MODELS_DIR / "mlp_l3_2_2048_v2_scaffold_best_model.pt",
        "split_path": PROCESSED_DIR / "mlp_l3_2_2048_v2_scaffold_split_assignments.csv",
    },
    {
        "model_name": "v3_cluster",
        "checkpoint_path": MODELS_DIR / "mlp_l3_2_2048_v3_cluster_best_model.pt",
        "split_path": PROCESSED_DIR / "mlp_l3_2_2048_v3_cluster_split_assignments.csv",
    },
    {
        "model_name": "v4_controlled_similarity",
        "checkpoint_path": MODELS_DIR / "mlp_l3_2_2048_v4_controlled_similarity_best_model.pt",
        "split_path": PROCESSED_DIR / "mlp_l3_2_2048_v4_controlled_similarity_split_assignments.csv",
    },
]

UNSEEN_DATASET_PATH = (
    PROCESSED_DIR
    / f"unseen_l3_min{MIN_UNIQUE_MOLECULES}_dataset.csv"
)

UNSEEN_X_PATH = (
    PROCESSED_DIR
    / f"unseen_l3_min{MIN_UNIQUE_MOLECULES}_2_2048_X.npy"
)

UNSEEN_METADATA_PATH = (
    PROCESSED_DIR
    / f"unseen_l3_min{MIN_UNIQUE_MOLECULES}_2_2048_metadata.csv"
)

UNSEEN_CLASS_STATS_PATH = (
    REPORTS_METRICS_DIR
    / f"unseen_l3_min{MIN_UNIQUE_MOLECULES}_class_stats.csv"
)

INVALID_SMILES_PATH = (
    REPORTS_DIAGNOSTICS_DIR
    / f"unseen_l3_min{MIN_UNIQUE_MOLECULES}_invalid_smiles.csv"
)

PER_MOLECULE_PREDICTIONS_PATH = (
    REPORTS_DIAGNOSTICS_DIR
    / f"unseen_l3_min{MIN_UNIQUE_MOLECULES}_mlp_per_molecule_predictions.csv"
)

CLASS_SUMMARY_PATH = (
    REPORTS_METRICS_DIR
    / f"unseen_l3_min{MIN_UNIQUE_MOLECULES}_mlp_class_summary.csv"
)

MODEL_COMPARISON_PATH = (
    REPORTS_METRICS_DIR
    / f"unseen_l3_min{MIN_UNIQUE_MOLECULES}_mlp_model_comparison_summary.csv"
)

CONFIG_OUTPUT_PATH = (
    REPORTS_METRICS_DIR
    / f"unseen_l3_min{MIN_UNIQUE_MOLECULES}_mlp_analysis_config.json"
)


class FingerprintDataset(Dataset):
    def __init__(self, x: np.ndarray):
        self.x = x

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, item: int):
        return self.x[item]


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

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.network[:-1](x)
        return embedding


def prepare_directories() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    if USE_CUDA and torch.cuda.is_available():
        return torch.device(f"cuda:{CUDA_DEVICE_ID}")

    return torch.device("cpu")


def load_known_data() -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    if not KNOWN_X_PATH.exists():
        raise FileNotFoundError(f"Known X file not found: {KNOWN_X_PATH}")

    if not KNOWN_Y_PATH.exists():
        raise FileNotFoundError(f"Known y file not found: {KNOWN_Y_PATH}")

    if not KNOWN_METADATA_PATH.exists():
        raise FileNotFoundError(f"Known metadata file not found: {KNOWN_METADATA_PATH}")

    x_known = np.load(KNOWN_X_PATH)
    y_known = np.load(KNOWN_Y_PATH).astype(np.int64)
    metadata_known = pd.read_csv(KNOWN_METADATA_PATH)

    if len(x_known) != len(y_known) or len(y_known) != len(metadata_known):
        raise ValueError(
            f"Known data length mismatch: "
            f"X={len(x_known)}, y={len(y_known)}, metadata={len(metadata_known)}"
        )

    label_map = (
        metadata_known[["label_id", "target_class_l3"]]
        .drop_duplicates()
        .sort_values("label_id")
    )

    class_names = label_map["target_class_l3"].tolist()

    return x_known, y_known, metadata_known, class_names


def load_known_molecule_ids() -> set[str]:
    if not KNOWN_DATASET_PATH.exists():
        raise FileNotFoundError(f"Known dataset file not found: {KNOWN_DATASET_PATH}")

    known_dataset = pd.read_csv(
        KNOWN_DATASET_PATH,
        usecols=["molecule_chembl_id"],
    )

    known_ids = (
        known_dataset["molecule_chembl_id"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    return set(known_ids)


def make_unseen_dataset(known_class_names: list[str]) -> pd.DataFrame:
    if not RAW_CSV_PATH.exists():
        raise FileNotFoundError(f"Raw CSV not found: {RAW_CSV_PATH}")

    print("Loading raw activity CSV...")
    print(f"Path: {RAW_CSV_PATH}")

    usecols = [
        "molregno",
        "molecule_chembl_id",
        "canonical_smiles",
        "pchembl_value",
        "target_class_l3",
        "target_chembl_id",
        "target_name",
        "target_organism",
        "standard_type",
    ]

    raw = pd.read_csv(RAW_CSV_PATH, usecols=usecols)

    print(f"Raw rows loaded: {len(raw)}")

    raw = raw.dropna(
        subset=[
            "molecule_chembl_id",
            "canonical_smiles",
            "pchembl_value",
            "target_class_l3",
        ]
    ).copy()

    raw["molecule_chembl_id"] = raw["molecule_chembl_id"].astype(str)
    raw["target_class_l3"] = raw["target_class_l3"].astype(str)

    before_exclusion = len(raw)

    raw = raw[~raw["target_class_l3"].isin(known_class_names)].copy()

    print(
        f"Rows after excluding known 11 classes: "
        f"{len(raw)} from {before_exclusion}"
    )

    if EXCLUDE_MOLECULES_FROM_KNOWN_DATASET:
        known_molecule_ids = load_known_molecule_ids()

        before_known_molecule_exclusion = len(raw)

        raw = raw[
            ~raw["molecule_chembl_id"].isin(known_molecule_ids)
        ].copy()

        print(
            "Rows after excluding molecules already present in known 11-class dataset: "
            f"{len(raw)} from {before_known_molecule_exclusion}"
        )

    class_stats_all = (
        raw
        .groupby("target_class_l3")
        .agg(
            activity_records=("molecule_chembl_id", "size"),
            unique_molecules=("molecule_chembl_id", "nunique"),
            unique_targets=("target_chembl_id", "nunique"),
            mean_pchembl=("pchembl_value", "mean"),
            median_pchembl=("pchembl_value", "median"),
            min_pchembl=("pchembl_value", "min"),
            max_pchembl=("pchembl_value", "max"),
        )
        .reset_index()
        .sort_values(
            ["unique_molecules", "activity_records"],
            ascending=[False, False],
        )
    )

    selected_classes = class_stats_all.loc[
        class_stats_all["unique_molecules"] >= MIN_UNIQUE_MOLECULES,
        "target_class_l3",
    ].tolist()

    print()
    print(f"All excluded L3 classes: {class_stats_all['target_class_l3'].nunique()}")
    print(
        f"Selected excluded L3 classes with unique_molecules >= "
        f"{MIN_UNIQUE_MOLECULES}: {len(selected_classes)}"
    )

    selected_stats = class_stats_all[
        class_stats_all["target_class_l3"].isin(selected_classes)
    ].copy()

    selected_stats.to_csv(
        UNSEEN_CLASS_STATS_PATH,
        index=False,
        encoding="utf-8",
    )

    print()
    print("Selected unseen classes:")
    print(selected_stats.to_string(index=False))
    print()
    print(f"Class stats saved to: {UNSEEN_CLASS_STATS_PATH}")

    unseen = raw[
        raw["target_class_l3"].isin(selected_classes)
    ].copy()

    if len(unseen) == 0:
        raise ValueError("No unseen data left after filtering.")

    grouped = (
        unseen
        .groupby(
            [
                "target_class_l3",
                "molregno",
                "molecule_chembl_id",
                "canonical_smiles",
            ],
            dropna=False,
        )
        .agg(
            max_pchembl_value=("pchembl_value", "max"),
            mean_pchembl_value=("pchembl_value", "mean"),
            activity_records=("pchembl_value", "size"),
            unique_targets=("target_chembl_id", "nunique"),
            target_names=("target_name", lambda values: "; ".join(sorted(set(values.dropna().astype(str)))[:10])),
            target_chembl_ids=("target_chembl_id", lambda values: "; ".join(sorted(set(values.dropna().astype(str)))[:10])),
            target_organisms=("target_organism", lambda values: "; ".join(sorted(set(values.dropna().astype(str)))[:10])),
            standard_types=("standard_type", lambda values: "; ".join(sorted(set(values.dropna().astype(str)))[:10])),
        )
        .reset_index()
        .sort_values(
            ["target_class_l3", "max_pchembl_value"],
            ascending=[True, False],
        )
    )

    grouped["unseen_class_id"] = (
        grouped["target_class_l3"]
        .astype("category")
        .cat.codes
        .astype(int)
    )

    grouped.to_csv(
        UNSEEN_DATASET_PATH,
        index=False,
        encoding="utf-8",
    )

    print()
    print("Unseen dataset created.")
    print(f"Rows molecule-class pairs: {len(grouped)}")
    print(f"Unique molecules: {grouped['molecule_chembl_id'].nunique()}")
    print(f"Unseen classes: {grouped['target_class_l3'].nunique()}")
    print(f"Saved to: {UNSEEN_DATASET_PATH}")

    return grouped


def smiles_to_morgan_fp(smiles: str, generator) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    fingerprint = generator.GetFingerprint(mol)
    array = np.zeros((FINGERPRINT_SIZE,), dtype=np.uint8)

    DataStructs.ConvertToNumpyArray(fingerprint, array)

    return array


def make_unseen_fingerprints(unseen_dataset: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    print()
    print("Generating Morgan fingerprints for unseen classes...")
    print("=" * 80)
    print(f"Radius: {FINGERPRINT_RADIUS}")
    print(f"Fingerprint size: {FINGERPRINT_SIZE}")
    print("=" * 80)

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=FINGERPRINT_RADIUS,
        fpSize=FINGERPRINT_SIZE,
    )

    fingerprints = []
    valid_rows = []
    invalid_rows = []

    start_time = time.time()

    for row_index, row in unseen_dataset.iterrows():
        smiles = row["canonical_smiles"]

        fingerprint = smiles_to_morgan_fp(smiles, generator)

        if fingerprint is None:
            invalid_rows.append(row.to_dict())
            continue

        fingerprints.append(fingerprint)
        valid_rows.append(row.to_dict())

        if len(valid_rows) % 5000 == 0:
            elapsed_minutes = (time.time() - start_time) / 60
            print(
                f"Valid fingerprints: {len(valid_rows)}, "
                f"processed rows: {row_index + 1}/{len(unseen_dataset)}, "
                f"elapsed={elapsed_minutes:.1f} min"
            )

    if len(fingerprints) == 0:
        raise ValueError("No valid fingerprints were generated.")

    x_unseen = np.asarray(fingerprints, dtype=np.float32)
    metadata_unseen = pd.DataFrame(valid_rows).reset_index(drop=True)

    invalid_df = pd.DataFrame(invalid_rows)

    x_unseen = x_unseen.astype(np.float32)

    np.save(UNSEEN_X_PATH, x_unseen)

    metadata_unseen.to_csv(
        UNSEEN_METADATA_PATH,
        index=False,
        encoding="utf-8",
    )

    if len(invalid_df) > 0:
        invalid_df.to_csv(
            INVALID_SMILES_PATH,
            index=False,
            encoding="utf-8",
        )

    elapsed_minutes = (time.time() - start_time) / 60

    print()
    print("Fingerprint generation completed.")
    print(f"Input rows: {len(unseen_dataset)}")
    print(f"Valid fingerprints: {len(metadata_unseen)}")
    print(f"Invalid SMILES: {len(invalid_df)}")
    print(f"X shape: {x_unseen.shape}")
    print(f"Elapsed time: {elapsed_minutes:.1f} min")
    print(f"X saved to: {UNSEEN_X_PATH}")
    print(f"Metadata saved to: {UNSEEN_METADATA_PATH}")

    if len(invalid_df) > 0:
        print(f"Invalid SMILES saved to: {INVALID_SMILES_PATH}")

    return x_unseen, metadata_unseen


def load_or_create_unseen_data(
    known_class_names: list[str],
) -> tuple[np.ndarray, pd.DataFrame]:
    if UNSEEN_X_PATH.exists() and UNSEEN_METADATA_PATH.exists():
        print("Loading cached unseen fingerprints and metadata...")
        print(f"X path: {UNSEEN_X_PATH}")
        print(f"Metadata path: {UNSEEN_METADATA_PATH}")

        x_unseen = np.load(UNSEEN_X_PATH).astype(np.float32)
        metadata_unseen = pd.read_csv(UNSEEN_METADATA_PATH)

        if len(x_unseen) != len(metadata_unseen):
            raise ValueError(
                f"Cached unseen length mismatch: "
                f"X={len(x_unseen)}, metadata={len(metadata_unseen)}"
            )

        return x_unseen, metadata_unseen

    unseen_dataset = make_unseen_dataset(known_class_names)
    x_unseen, metadata_unseen = make_unseen_fingerprints(unseen_dataset)

    return x_unseen, metadata_unseen


def load_model(checkpoint_path: Path, n_classes: int, device: torch.device) -> tuple[MLPClassifier, dict]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = MLPClassifier(
        input_size=INPUT_SIZE,
        n_classes=n_classes,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


def get_train_indices_for_run(
    split_path: Path,
    n_known_samples: int,
) -> np.ndarray:
    if not split_path.exists():
        print(f"Split file not found, using all known samples for centroids: {split_path}")
        return np.arange(n_known_samples, dtype=np.int64)

    split_data = pd.read_csv(split_path)

    if len(split_data) != n_known_samples:
        print(
            f"Split file length mismatch, using all known samples for centroids. "
            f"split={len(split_data)}, known={n_known_samples}"
        )
        return np.arange(n_known_samples, dtype=np.int64)

    if "split" not in split_data.columns:
        print(
            f"Split column not found, using all known samples for centroids: {split_path}"
        )
        return np.arange(n_known_samples, dtype=np.int64)

    train_indices = split_data.index[
        split_data["split"] == "train"
    ].to_numpy(dtype=np.int64)

    if len(train_indices) == 0:
        print(
            f"No train rows in split file, using all known samples for centroids: {split_path}"
        )
        return np.arange(n_known_samples, dtype=np.int64)

    return train_indices


@torch.no_grad()
def predict_logits_probabilities_embeddings(
    model: MLPClassifier,
    x: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = FingerprintDataset(x)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    all_probabilities = []
    all_embeddings = []

    for x_batch in loader:
        x_batch = x_batch.to(device, non_blocking=True).float()

        embeddings = model.extract_embedding(x_batch)
        logits = model.network[-1](embeddings)
        probabilities = torch.softmax(logits, dim=1)

        all_probabilities.append(probabilities.cpu().numpy())
        all_embeddings.append(embeddings.cpu().numpy())

    probabilities = np.vstack(all_probabilities)
    embeddings = np.vstack(all_embeddings)

    return probabilities, embeddings


def compute_class_centroids(
    embeddings: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    centroids = []

    for class_id in range(n_classes):
        class_indices = indices[y[indices] == class_id]

        if len(class_indices) == 0:
            raise ValueError(
                f"Cannot compute centroid: class {class_id} missing in selected indices."
            )

        centroid = embeddings[class_indices].mean(axis=0)
        centroids.append(centroid)

    return np.vstack(centroids).astype(np.float32)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


def entropy_from_probabilities(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1)
    return entropy


def top_k_from_vector(values: np.ndarray, names: list[str], k: int = 3) -> list[tuple[str, float]]:
    order = np.argsort(values)[::-1][:k]
    return [(names[index], float(values[index])) for index in order]


def make_per_molecule_prediction_rows(
    model_name: str,
    metadata_unseen: pd.DataFrame,
    probabilities: np.ndarray,
    unseen_embeddings: np.ndarray,
    known_centroids: np.ndarray,
    class_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_classes = len(class_names)

    max_probability = probabilities.max(axis=1)
    predicted_label_ids = probabilities.argmax(axis=1)
    entropy = entropy_from_probabilities(probabilities)

    known_centroids_normalized = normalize_rows(known_centroids)
    unseen_embeddings_normalized = normalize_rows(unseen_embeddings)

    cosine_similarities = unseen_embeddings_normalized @ known_centroids_normalized.T

    nearest_centroid_ids = cosine_similarities.argmax(axis=1)
    nearest_centroid_similarity = cosine_similarities.max(axis=1)

    euclidean_distances = np.sqrt(
        np.maximum(
            (
                np.sum(unseen_embeddings ** 2, axis=1, keepdims=True)
                + np.sum(known_centroids ** 2, axis=1)[None, :]
                - 2.0 * unseen_embeddings @ known_centroids.T
            ),
            0.0,
        )
    )

    nearest_euclidean_ids = euclidean_distances.argmin(axis=1)
    nearest_euclidean_distance = euclidean_distances.min(axis=1)

    rows = metadata_unseen.copy()
    rows["model_name"] = model_name
    rows["top1_softmax_label_id"] = predicted_label_ids
    rows["top1_softmax_known_class"] = [
        class_names[label_id]
        for label_id in predicted_label_ids
    ]
    rows["top1_softmax_probability"] = max_probability
    rows["softmax_entropy"] = entropy

    rows["nearest_centroid_cosine_label_id"] = nearest_centroid_ids
    rows["nearest_centroid_cosine_known_class"] = [
        class_names[label_id]
        for label_id in nearest_centroid_ids
    ]
    rows["nearest_centroid_cosine_similarity"] = nearest_centroid_similarity

    rows["nearest_centroid_euclidean_label_id"] = nearest_euclidean_ids
    rows["nearest_centroid_euclidean_known_class"] = [
        class_names[label_id]
        for label_id in nearest_euclidean_ids
    ]
    rows["nearest_centroid_euclidean_distance"] = nearest_euclidean_distance

    for class_id, class_name in enumerate(class_names):
        safe_name = (
            class_name
            .replace(" ", "_")
            .replace("/", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("-", "_")
            .replace(",", "")
        )

        rows[f"prob_{class_id}_{safe_name}"] = probabilities[:, class_id]
        rows[f"centroid_cosine_{class_id}_{safe_name}"] = cosine_similarities[:, class_id]

    class_summary_rows = []

    for unseen_class, group in rows.groupby("target_class_l3"):
        group_indices = group.index.to_numpy(dtype=np.int64)

        group_probabilities = probabilities[group_indices]
        group_embeddings = unseen_embeddings[group_indices]

        mean_probabilities = group_probabilities.mean(axis=0)

        top3_softmax = top_k_from_vector(
            mean_probabilities,
            class_names,
            k=3,
        )

        unseen_class_centroid = group_embeddings.mean(axis=0, keepdims=True)
        unseen_class_centroid_normalized = normalize_rows(unseen_class_centroid)

        class_centroid_cosines = (
            unseen_class_centroid_normalized
            @ known_centroids_normalized.T
        ).reshape(-1)

        top3_centroid_cosine = top_k_from_vector(
            class_centroid_cosines,
            class_names,
            k=3,
        )

        class_centroid_euclidean = np.sqrt(
            np.maximum(
                (
                    np.sum(unseen_class_centroid ** 2, axis=1, keepdims=True)
                    + np.sum(known_centroids ** 2, axis=1)[None, :]
                    - 2.0 * unseen_class_centroid @ known_centroids.T
                ),
                0.0,
            )
        ).reshape(-1)

        nearest_euclidean_order = np.argsort(class_centroid_euclidean)[:3]
        top3_centroid_euclidean = [
            (
                class_names[index],
                float(class_centroid_euclidean[index]),
            )
            for index in nearest_euclidean_order
        ]

        class_summary_rows.append(
            {
                "model_name": model_name,
                "unseen_target_class_l3": unseen_class,
                "molecule_class_pairs": len(group),
                "unique_molecules": group["molecule_chembl_id"].nunique(),
                "mean_max_pchembl_value": group["max_pchembl_value"].mean(),
                "median_max_pchembl_value": group["max_pchembl_value"].median(),
                "mean_activity_records": group["activity_records"].mean(),
                "mean_unique_targets": group["unique_targets"].mean(),

                "top1_softmax_mean_known_class": top3_softmax[0][0],
                "top1_softmax_mean_probability": top3_softmax[0][1],
                "top2_softmax_mean_known_class": top3_softmax[1][0],
                "top2_softmax_mean_probability": top3_softmax[1][1],
                "top3_softmax_mean_known_class": top3_softmax[2][0],
                "top3_softmax_mean_probability": top3_softmax[2][1],

                "mean_top1_softmax_probability": group["top1_softmax_probability"].mean(),
                "median_top1_softmax_probability": group["top1_softmax_probability"].median(),
                "mean_softmax_entropy": group["softmax_entropy"].mean(),
                "fraction_confident_ge_0_50": (
                    group["top1_softmax_probability"] >= 0.50
                ).mean(),
                "fraction_confident_ge_0_70": (
                    group["top1_softmax_probability"] >= 0.70
                ).mean(),
                "fraction_confident_ge_0_90": (
                    group["top1_softmax_probability"] >= 0.90
                ).mean(),

                "top1_per_molecule_softmax_mode": (
                    group["top1_softmax_known_class"].mode().iloc[0]
                    if len(group["top1_softmax_known_class"].mode()) > 0
                    else None
                ),

                "top1_centroid_cosine_known_class": top3_centroid_cosine[0][0],
                "top1_centroid_cosine_similarity": top3_centroid_cosine[0][1],
                "top2_centroid_cosine_known_class": top3_centroid_cosine[1][0],
                "top2_centroid_cosine_similarity": top3_centroid_cosine[1][1],
                "top3_centroid_cosine_known_class": top3_centroid_cosine[2][0],
                "top3_centroid_cosine_similarity": top3_centroid_cosine[2][1],

                "top1_centroid_euclidean_known_class": top3_centroid_euclidean[0][0],
                "top1_centroid_euclidean_distance": top3_centroid_euclidean[0][1],
                "top2_centroid_euclidean_known_class": top3_centroid_euclidean[1][0],
                "top2_centroid_euclidean_distance": top3_centroid_euclidean[1][1],
                "top3_centroid_euclidean_known_class": top3_centroid_euclidean[2][0],
                "top3_centroid_euclidean_distance": top3_centroid_euclidean[2][1],

                "mean_nearest_centroid_cosine_similarity": group[
                    "nearest_centroid_cosine_similarity"
                ].mean(),
                "top1_per_molecule_centroid_mode": (
                    group["nearest_centroid_cosine_known_class"].mode().iloc[0]
                    if len(group["nearest_centroid_cosine_known_class"].mode()) > 0
                    else None
                ),
            }
        )

    class_summary = pd.DataFrame(class_summary_rows)

    return rows, class_summary


def make_model_comparison_summary(class_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for unseen_class, group in class_summary.groupby("unseen_target_class_l3"):
        row = {
            "unseen_target_class_l3": unseen_class,
            "models_compared": group["model_name"].nunique(),
            "unique_top1_softmax_classes": group[
                "top1_softmax_mean_known_class"
            ].nunique(),
            "unique_top1_centroid_classes": group[
                "top1_centroid_cosine_known_class"
            ].nunique(),
        }

        for _, model_row in group.sort_values("model_name").iterrows():
            model_name = model_row["model_name"]

            row[f"{model_name}_top1_softmax"] = model_row[
                "top1_softmax_mean_known_class"
            ]
            row[f"{model_name}_top1_softmax_probability"] = model_row[
                "top1_softmax_mean_probability"
            ]
            row[f"{model_name}_top1_centroid_cosine"] = model_row[
                "top1_centroid_cosine_known_class"
            ]
            row[f"{model_name}_top1_centroid_cosine_similarity"] = model_row[
                "top1_centroid_cosine_similarity"
            ]
            row[f"{model_name}_mean_entropy"] = model_row[
                "mean_softmax_entropy"
            ]

        rows.append(row)

    comparison = pd.DataFrame(rows)

    return comparison


def main() -> None:
    prepare_directories()

    start_time = time.time()
    device = get_device()

    print("Unseen L3 class characterization with trained MLP models")
    print("=" * 80)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU name: {torch.cuda.get_device_name(CUDA_DEVICE_ID)}")
    print(f"Minimum unique molecules per unseen class: {MIN_UNIQUE_MOLECULES}")
    print(f"Exclude molecules from known 11-class dataset: {EXCLUDE_MOLECULES_FROM_KNOWN_DATASET}")
    print("=" * 80)
    print()

    x_known, y_known, metadata_known, class_names = load_known_data()

    n_classes = len(class_names)

    print("Known trained classes")
    print("=" * 80)
    for class_id, class_name in enumerate(class_names):
        print(f"{class_id:02d} | {class_name}")
    print("=" * 80)
    print()

    x_unseen, metadata_unseen = load_or_create_unseen_data(class_names)

    print()
    print("Unseen data ready")
    print("=" * 80)
    print(f"X unseen shape: {x_unseen.shape}")
    print(f"Unseen metadata rows: {len(metadata_unseen)}")
    print(f"Unseen classes: {metadata_unseen['target_class_l3'].nunique()}")
    print("=" * 80)
    print()

    all_prediction_tables = []
    all_class_summaries = []

    for run in RUNS:
        model_name = run["model_name"]
        checkpoint_path = run["checkpoint_path"]
        split_path = run["split_path"]

        print()
        print(f"Processing model: {model_name}")
        print("=" * 80)
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Split path: {split_path}")

        if not checkpoint_path.exists():
            print(f"Skipping {model_name}: checkpoint not found.")
            continue

        model, checkpoint = load_model(
            checkpoint_path=checkpoint_path,
            n_classes=n_classes,
            device=device,
        )

        checkpoint_class_names = checkpoint.get("class_names", None)

        if checkpoint_class_names is not None and list(checkpoint_class_names) != class_names:
            raise ValueError(
                f"Class name mismatch for {model_name}. "
                f"Checkpoint classes: {checkpoint_class_names}. "
                f"Expected classes: {class_names}."
            )

        train_indices = get_train_indices_for_run(
            split_path=split_path,
            n_known_samples=len(x_known),
        )

        print(f"Centroid base samples: {len(train_indices)}")

        print("Computing known embeddings...")
        _, known_embeddings = predict_logits_probabilities_embeddings(
            model=model,
            x=x_known,
            device=device,
        )

        known_centroids = compute_class_centroids(
            embeddings=known_embeddings,
            y=y_known,
            indices=train_indices,
            n_classes=n_classes,
        )

        print("Computing unseen probabilities and embeddings...")
        unseen_probabilities, unseen_embeddings = predict_logits_probabilities_embeddings(
            model=model,
            x=x_unseen,
            device=device,
        )

        predictions, class_summary = make_per_molecule_prediction_rows(
            model_name=model_name,
            metadata_unseen=metadata_unseen,
            probabilities=unseen_probabilities,
            unseen_embeddings=unseen_embeddings,
            known_centroids=known_centroids,
            class_names=class_names,
        )

        all_prediction_tables.append(predictions)
        all_class_summaries.append(class_summary)

        print(f"Completed model: {model_name}")
        print(f"Class summary rows: {len(class_summary)}")
        print("=" * 80)

    if len(all_prediction_tables) == 0:
        raise ValueError("No model predictions were generated.")

    per_molecule_predictions = pd.concat(
        all_prediction_tables,
        axis=0,
        ignore_index=True,
    )

    class_summary = pd.concat(
        all_class_summaries,
        axis=0,
        ignore_index=True,
    )

    model_comparison = make_model_comparison_summary(class_summary)

    per_molecule_predictions.to_csv(
        PER_MOLECULE_PREDICTIONS_PATH,
        index=False,
        encoding="utf-8",
    )

    class_summary.to_csv(
        CLASS_SUMMARY_PATH,
        index=False,
        encoding="utf-8",
    )

    model_comparison.to_csv(
        MODEL_COMPARISON_PATH,
        index=False,
        encoding="utf-8",
    )

    config = {
        "analysis": "unseen_l3_class_characterization_with_mlp",
        "min_unique_molecules": MIN_UNIQUE_MOLECULES,
        "exclude_molecules_from_known_dataset": EXCLUDE_MOLECULES_FROM_KNOWN_DATASET,
        "fingerprint_radius": FINGERPRINT_RADIUS,
        "fingerprint_size": FINGERPRINT_SIZE,
        "known_classes": class_names,
        "runs": [
            {
                "model_name": run["model_name"],
                "checkpoint_path": str(run["checkpoint_path"]),
                "split_path": str(run["split_path"]),
                "checkpoint_exists": run["checkpoint_path"].exists(),
            }
            for run in RUNS
        ],
        "output_files": {
            "unseen_dataset_path": str(UNSEEN_DATASET_PATH),
            "unseen_x_path": str(UNSEEN_X_PATH),
            "unseen_metadata_path": str(UNSEEN_METADATA_PATH),
            "unseen_class_stats_path": str(UNSEEN_CLASS_STATS_PATH),
            "per_molecule_predictions_path": str(PER_MOLECULE_PREDICTIONS_PATH),
            "class_summary_path": str(CLASS_SUMMARY_PATH),
            "model_comparison_path": str(MODEL_COMPARISON_PATH),
        },
    }

    with open(CONFIG_OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=4)

    elapsed_minutes = (time.time() - start_time) / 60

    print()
    print("Analysis completed.")
    print("=" * 80)
    print(f"Per-molecule predictions saved to: {PER_MOLECULE_PREDICTIONS_PATH}")
    print(f"Class summary saved to: {CLASS_SUMMARY_PATH}")
    print(f"Model comparison saved to: {MODEL_COMPARISON_PATH}")
    print(f"Config saved to: {CONFIG_OUTPUT_PATH}")
    print(f"Elapsed time: {elapsed_minutes:.1f} min")
    print("=" * 80)

    print()
    print("Top summary preview")
    print("=" * 80)
    preview_columns = [
        "model_name",
        "unseen_target_class_l3",
        "unique_molecules",
        "top1_softmax_mean_known_class",
        "top1_softmax_mean_probability",
        "top1_centroid_cosine_known_class",
        "top1_centroid_cosine_similarity",
        "mean_softmax_entropy",
    ]

    available_preview_columns = [
        column
        for column in preview_columns
        if column in class_summary.columns
    ]

    print(
        class_summary[
            available_preview_columns
        ]
        .sort_values(
            ["model_name", "unique_molecules"],
            ascending=[True, False],
        )
        .head(40)
        .to_string(index=False)
    )
    print("=" * 80)


if __name__ == "__main__":
    main()