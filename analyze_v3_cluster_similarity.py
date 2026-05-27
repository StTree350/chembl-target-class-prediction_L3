from pathlib import Path
import time

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"
REPORTS_DIAGNOSTICS_DIR = PROJECT_ROOT / "reports" / "diagnostics"

X_PATH = PROCESSED_DIR / "target_class_l3_2_2048_X.npy"
Y_PATH = PROCESSED_DIR / "target_class_l3_2_2048_y.npy"
METADATA_PATH = PROCESSED_DIR / "target_class_l3_2_2048_metadata.csv"

SPLIT_ASSIGNMENTS_PATH = (
    PROCESSED_DIR
    / "mlp_l3_2_2048_v3_cluster_split_assignments.csv"
)

TEST_PREDICTIONS_PATH = (
    REPORTS_DIAGNOSTICS_DIR
    / "mlp_l3_2_2048_v3_cluster_test_predictions.csv"
)

OUTPUT_SIMILARITY_PATH = (
    REPORTS_DIAGNOSTICS_DIR
    / "mlp_l3_2_2048_v3_cluster_nearest_train_similarity.csv"
)

SUMMARY_OUTPUT_PATH = (
    REPORTS_METRICS_DIR
    / "mlp_l3_2_2048_v3_cluster_similarity_summary.csv"
)

CLASS_SUMMARY_OUTPUT_PATH = (
    REPORTS_METRICS_DIR
    / "mlp_l3_2_2048_v3_cluster_similarity_by_class.csv"
)

BIN_SUMMARY_OUTPUT_PATH = (
    REPORTS_METRICS_DIR
    / "mlp_l3_2_2048_v3_cluster_similarity_bins.csv"
)

TARGET_CLASS_COLUMN = "target_class_l3"
LABEL_COLUMN = "label_id"

BATCH_SIZE = 256

SIMILARITY_THRESHOLDS = [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.85,
    0.90,
]


def prepare_directories() -> None:
    REPORTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)


def load_inputs() -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    if not X_PATH.exists():
        raise FileNotFoundError(f"X file not found: {X_PATH}")

    if not Y_PATH.exists():
        raise FileNotFoundError(f"y file not found: {Y_PATH}")

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")

    if not SPLIT_ASSIGNMENTS_PATH.exists():
        raise FileNotFoundError(
            f"Split assignments file not found: {SPLIT_ASSIGNMENTS_PATH}"
        )

    x = np.load(X_PATH)
    y = np.load(Y_PATH).astype(np.int64)
    metadata = pd.read_csv(METADATA_PATH)
    split_assignments = pd.read_csv(SPLIT_ASSIGNMENTS_PATH)

    if len(x) != len(y):
        raise ValueError(f"Length mismatch: X={len(x)}, y={len(y)}")

    if len(x) != len(metadata):
        raise ValueError(f"Length mismatch: X={len(x)}, metadata={len(metadata)}")

    if len(x) != len(split_assignments):
        raise ValueError(
            f"Length mismatch: X={len(x)}, split_assignments={len(split_assignments)}"
        )

    if "split" not in split_assignments.columns:
        raise ValueError("Column 'split' not found in split assignments file.")

    required_metadata_columns = [
        "molregno",
        "molecule_chembl_id",
        "canonical_smiles",
        TARGET_CLASS_COLUMN,
        LABEL_COLUMN,
    ]

    missing_metadata_columns = [
        column
        for column in required_metadata_columns
        if column not in metadata.columns
    ]

    if missing_metadata_columns:
        raise ValueError(
            f"Missing required columns in metadata: {missing_metadata_columns}"
        )

    return x, y, metadata, split_assignments


def get_class_names(metadata: pd.DataFrame) -> list[str]:
    label_map = (
        metadata[[LABEL_COLUMN, TARGET_CLASS_COLUMN]]
        .drop_duplicates()
        .sort_values(LABEL_COLUMN)
    )

    return label_map[TARGET_CLASS_COLUMN].tolist()


def tanimoto_batch(
    query_batch: np.ndarray,
    train_matrix: np.ndarray,
    query_bit_counts: np.ndarray,
    train_bit_counts: np.ndarray,
) -> np.ndarray:
    intersections = query_batch @ train_matrix.T

    unions = (
        query_bit_counts[:, None]
        + train_bit_counts[None, :]
        - intersections
    )

    similarities = np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections, dtype=np.float32),
        where=unions > 0,
    )

    return similarities


def compute_nearest_train_similarity(
    x: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    train_indices: np.ndarray,
    query_indices: np.ndarray,
    query_split_name: str,
    class_names: list[str],
) -> pd.DataFrame:
    start_time = time.time()

    print()
    print(f"Computing nearest train Tanimoto similarity for split: {query_split_name}")
    print("=" * 80)
    print(f"Train molecules: {len(train_indices)}")
    print(f"Query molecules: {len(query_indices)}")
    print(f"Batch size: {BATCH_SIZE}")
    print("=" * 80)

    x_train = x[train_indices].astype(np.float32, copy=False)
    y_train = y[train_indices]

    train_bit_counts = x_train.sum(axis=1).astype(np.float32)

    result_rows = []

    total_queries = len(query_indices)

    for start in range(0, total_queries, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total_queries)

        batch_query_indices = query_indices[start:end]
        x_query = x[batch_query_indices].astype(np.float32, copy=False)
        y_query = y[batch_query_indices]

        query_bit_counts = x_query.sum(axis=1).astype(np.float32)

        similarities = tanimoto_batch(
            query_batch=x_query,
            train_matrix=x_train,
            query_bit_counts=query_bit_counts,
            train_bit_counts=train_bit_counts,
        )

        nearest_any_positions = np.argmax(similarities, axis=1)
        nearest_any_similarities = similarities[
            np.arange(len(batch_query_indices)),
            nearest_any_positions,
        ]

        same_class_mask = y_train[None, :] == y_query[:, None]
        similarities_same_class = np.where(
            same_class_mask,
            similarities,
            -1.0,
        )

        nearest_same_positions = np.argmax(similarities_same_class, axis=1)
        nearest_same_similarities = similarities_same_class[
            np.arange(len(batch_query_indices)),
            nearest_same_positions,
        ]

        different_class_mask = y_train[None, :] != y_query[:, None]
        similarities_different_class = np.where(
            different_class_mask,
            similarities,
            -1.0,
        )

        nearest_different_positions = np.argmax(similarities_different_class, axis=1)
        nearest_different_similarities = similarities_different_class[
            np.arange(len(batch_query_indices)),
            nearest_different_positions,
        ]

        for local_row, query_index in enumerate(batch_query_indices):
            true_label_id = int(y[query_index])

            nearest_any_train_index = int(train_indices[nearest_any_positions[local_row]])
            nearest_same_train_index = int(train_indices[nearest_same_positions[local_row]])
            nearest_different_train_index = int(
                train_indices[nearest_different_positions[local_row]]
            )

            nearest_any_label_id = int(y[nearest_any_train_index])
            nearest_same_label_id = int(y[nearest_same_train_index])
            nearest_different_label_id = int(y[nearest_different_train_index])

            result_rows.append(
                {
                    "row_index": int(query_index),
                    "split": query_split_name,
                    "molregno": metadata.loc[query_index, "molregno"],
                    "molecule_chembl_id": metadata.loc[
                        query_index,
                        "molecule_chembl_id",
                    ],
                    "canonical_smiles": metadata.loc[
                        query_index,
                        "canonical_smiles",
                    ],
                    "true_label_id": true_label_id,
                    "true_target_class_l3": class_names[true_label_id],
                    "nearest_train_similarity": float(
                        nearest_any_similarities[local_row]
                    ),
                    "nearest_train_row_index": nearest_any_train_index,
                    "nearest_train_molregno": metadata.loc[
                        nearest_any_train_index,
                        "molregno",
                    ],
                    "nearest_train_molecule_chembl_id": metadata.loc[
                        nearest_any_train_index,
                        "molecule_chembl_id",
                    ],
                    "nearest_train_label_id": nearest_any_label_id,
                    "nearest_train_target_class_l3": class_names[
                        nearest_any_label_id
                    ],
                    "nearest_train_same_class": bool(
                        nearest_any_label_id == true_label_id
                    ),
                    "nearest_same_class_similarity": float(
                        nearest_same_similarities[local_row]
                    ),
                    "nearest_same_class_row_index": nearest_same_train_index,
                    "nearest_same_class_molecule_chembl_id": metadata.loc[
                        nearest_same_train_index,
                        "molecule_chembl_id",
                    ],
                    "nearest_different_class_similarity": float(
                        nearest_different_similarities[local_row]
                    ),
                    "nearest_different_class_row_index": nearest_different_train_index,
                    "nearest_different_class_molecule_chembl_id": metadata.loc[
                        nearest_different_train_index,
                        "molecule_chembl_id",
                    ],
                    "nearest_different_class_label_id": nearest_different_label_id,
                    "nearest_different_class_target_class_l3": class_names[
                        nearest_different_label_id
                    ],
                    "similarity_margin_same_minus_different": float(
                        nearest_same_similarities[local_row]
                        - nearest_different_similarities[local_row]
                    ),
                }
            )

        processed = end
        elapsed_minutes = (time.time() - start_time) / 60

        print(
            f"{query_split_name}: processed {processed}/{total_queries} "
            f"queries, elapsed {elapsed_minutes:.1f} min"
        )

    result = pd.DataFrame(result_rows)

    print(f"{query_split_name}: completed.")

    return result


def merge_test_predictions(
    similarity_data: pd.DataFrame,
) -> pd.DataFrame:
    if not TEST_PREDICTIONS_PATH.exists():
        print()
        print(f"Test predictions file not found, skipping merge: {TEST_PREDICTIONS_PATH}")
        return similarity_data

    predictions = pd.read_csv(TEST_PREDICTIONS_PATH)

    required_prediction_columns = [
        "molregno",
        "molecule_chembl_id",
        "predicted_label_id",
        "predicted_target_class_l3",
        "predicted_probability",
        "correct",
    ]

    missing_prediction_columns = [
        column
        for column in required_prediction_columns
        if column not in predictions.columns
    ]

    if missing_prediction_columns:
        print()
        print(
            "Prediction file exists, but required columns are missing. "
            f"Missing columns: {missing_prediction_columns}"
        )
        return similarity_data

    prediction_columns = [
        "molregno",
        "molecule_chembl_id",
        "predicted_label_id",
        "predicted_target_class_l3",
        "predicted_probability",
        "correct",
    ]

    merged = similarity_data.merge(
        predictions[prediction_columns],
        on=["molregno", "molecule_chembl_id"],
        how="left",
    )

    return merged


def make_summary(similarity_data: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for split_name, split_data in similarity_data.groupby("split"):
        row = {
            "split": split_name,
            "samples": len(split_data),
            "nearest_train_similarity_mean": split_data[
                "nearest_train_similarity"
            ].mean(),
            "nearest_train_similarity_std": split_data[
                "nearest_train_similarity"
            ].std(),
            "nearest_train_similarity_min": split_data[
                "nearest_train_similarity"
            ].min(),
            "nearest_train_similarity_q01": split_data[
                "nearest_train_similarity"
            ].quantile(0.01),
            "nearest_train_similarity_q05": split_data[
                "nearest_train_similarity"
            ].quantile(0.05),
            "nearest_train_similarity_q10": split_data[
                "nearest_train_similarity"
            ].quantile(0.10),
            "nearest_train_similarity_q25": split_data[
                "nearest_train_similarity"
            ].quantile(0.25),
            "nearest_train_similarity_median": split_data[
                "nearest_train_similarity"
            ].median(),
            "nearest_train_similarity_q75": split_data[
                "nearest_train_similarity"
            ].quantile(0.75),
            "nearest_train_similarity_q90": split_data[
                "nearest_train_similarity"
            ].quantile(0.90),
            "nearest_train_similarity_q95": split_data[
                "nearest_train_similarity"
            ].quantile(0.95),
            "nearest_train_similarity_q99": split_data[
                "nearest_train_similarity"
            ].quantile(0.99),
            "nearest_train_similarity_max": split_data[
                "nearest_train_similarity"
            ].max(),
            "nearest_same_class_similarity_mean": split_data[
                "nearest_same_class_similarity"
            ].mean(),
            "nearest_different_class_similarity_mean": split_data[
                "nearest_different_class_similarity"
            ].mean(),
            "similarity_margin_same_minus_different_mean": split_data[
                "similarity_margin_same_minus_different"
            ].mean(),
            "nearest_train_same_class_fraction": split_data[
                "nearest_train_same_class"
            ].mean(),
        }

        for threshold in SIMILARITY_THRESHOLDS:
            row[f"fraction_nearest_train_similarity_ge_{threshold:.2f}"] = (
                split_data["nearest_train_similarity"] >= threshold
            ).mean()

        if "correct" in split_data.columns and split_data["correct"].notna().any():
            row["model_accuracy"] = split_data["correct"].mean()

        rows.append(row)

    return pd.DataFrame(rows)


def make_class_summary(similarity_data: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        similarity_data
        .groupby(["split", "true_label_id", "true_target_class_l3"])
        .agg(
            samples=("row_index", "count"),
            nearest_train_similarity_mean=("nearest_train_similarity", "mean"),
            nearest_train_similarity_median=("nearest_train_similarity", "median"),
            nearest_train_similarity_q25=("nearest_train_similarity", lambda x: x.quantile(0.25)),
            nearest_train_similarity_q75=("nearest_train_similarity", lambda x: x.quantile(0.75)),
            nearest_same_class_similarity_mean=("nearest_same_class_similarity", "mean"),
            nearest_different_class_similarity_mean=("nearest_different_class_similarity", "mean"),
            similarity_margin_mean=("similarity_margin_same_minus_different", "mean"),
            nearest_train_same_class_fraction=("nearest_train_same_class", "mean"),
        )
        .reset_index()
        .sort_values(["split", "true_label_id"])
    )

    if "correct" in similarity_data.columns and similarity_data["correct"].notna().any():
        accuracy_by_class = (
            similarity_data
            .dropna(subset=["correct"])
            .groupby(["split", "true_label_id", "true_target_class_l3"])
            .agg(model_accuracy=("correct", "mean"))
            .reset_index()
        )

        grouped = grouped.merge(
            accuracy_by_class,
            on=["split", "true_label_id", "true_target_class_l3"],
            how="left",
        )

    return grouped


def make_bin_summary(similarity_data: pd.DataFrame) -> pd.DataFrame:
    bins = [
        0.0,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        1.000001,
    ]

    labels = [
        "0.00-0.20",
        "0.20-0.30",
        "0.30-0.40",
        "0.40-0.50",
        "0.50-0.60",
        "0.60-0.70",
        "0.70-0.80",
        "0.80-0.90",
        "0.90-1.00",
    ]

    data = similarity_data.copy()

    data["nearest_train_similarity_bin"] = pd.cut(
        data["nearest_train_similarity"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )

    grouped = (
        data
        .groupby(["split", "nearest_train_similarity_bin"], observed=False)
        .agg(
            samples=("row_index", "count"),
            nearest_train_similarity_mean=("nearest_train_similarity", "mean"),
            nearest_same_class_similarity_mean=("nearest_same_class_similarity", "mean"),
            nearest_different_class_similarity_mean=("nearest_different_class_similarity", "mean"),
            nearest_train_same_class_fraction=("nearest_train_same_class", "mean"),
        )
        .reset_index()
    )

    if "correct" in data.columns and data["correct"].notna().any():
        accuracy_by_bin = (
            data
            .dropna(subset=["correct"])
            .groupby(["split", "nearest_train_similarity_bin"], observed=False)
            .agg(model_accuracy=("correct", "mean"))
            .reset_index()
        )

        grouped = grouped.merge(
            accuracy_by_bin,
            on=["split", "nearest_train_similarity_bin"],
            how="left",
        )

    return grouped


def main() -> None:
    prepare_directories()

    start_time = time.time()

    print("V3 cluster split nearest-train similarity analysis")
    print("=" * 80)
    print(f"X path: {X_PATH}")
    print(f"y path: {Y_PATH}")
    print(f"Metadata path: {METADATA_PATH}")
    print(f"Split assignments path: {SPLIT_ASSIGNMENTS_PATH}")
    print("=" * 80)

    x, y, metadata, split_assignments = load_inputs()

    class_names = get_class_names(metadata)

    train_indices = split_assignments.index[
        split_assignments["split"] == "train"
    ].to_numpy(dtype=np.int64)

    validation_indices = split_assignments.index[
        split_assignments["split"] == "validation"
    ].to_numpy(dtype=np.int64)

    test_indices = split_assignments.index[
        split_assignments["split"] == "test"
    ].to_numpy(dtype=np.int64)

    print()
    print("Split sizes")
    print("=" * 80)
    print(f"Train: {len(train_indices)}")
    print(f"Validation: {len(validation_indices)}")
    print(f"Test: {len(test_indices)}")
    print("=" * 80)

    validation_similarity = compute_nearest_train_similarity(
        x=x,
        y=y,
        metadata=metadata,
        train_indices=train_indices,
        query_indices=validation_indices,
        query_split_name="validation",
        class_names=class_names,
    )

    test_similarity = compute_nearest_train_similarity(
        x=x,
        y=y,
        metadata=metadata,
        train_indices=train_indices,
        query_indices=test_indices,
        query_split_name="test",
        class_names=class_names,
    )

    similarity_data = pd.concat(
        [
            validation_similarity,
            test_similarity,
        ],
        axis=0,
        ignore_index=True,
    )

    similarity_data = merge_test_predictions(similarity_data)

    summary = make_summary(similarity_data)
    class_summary = make_class_summary(similarity_data)
    bin_summary = make_bin_summary(similarity_data)

    similarity_data.to_csv(
        OUTPUT_SIMILARITY_PATH,
        index=False,
        encoding="utf-8",
    )

    summary.to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    class_summary.to_csv(
        CLASS_SUMMARY_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    bin_summary.to_csv(
        BIN_SUMMARY_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    elapsed_minutes = (time.time() - start_time) / 60

    print()
    print("Similarity summary")
    print("=" * 80)
    print(summary.to_string(index=False))
    print("=" * 80)

    print()
    print("Files saved")
    print("=" * 80)
    print(f"Per-molecule similarity saved to: {OUTPUT_SIMILARITY_PATH}")
    print(f"Summary saved to: {SUMMARY_OUTPUT_PATH}")
    print(f"Class summary saved to: {CLASS_SUMMARY_OUTPUT_PATH}")
    print(f"Bin summary saved to: {BIN_SUMMARY_OUTPUT_PATH}")
    print(f"Elapsed time: {elapsed_minutes:.1f} min")
    print("=" * 80)


if __name__ == "__main__":
    main()