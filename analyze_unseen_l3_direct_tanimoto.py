from pathlib import Path
import json
import time

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"
REPORTS_DIAGNOSTICS_DIR = PROJECT_ROOT / "reports" / "diagnostics"

KNOWN_X_PATH = PROCESSED_DIR / "target_class_l3_2_2048_X.npy"
KNOWN_Y_PATH = PROCESSED_DIR / "target_class_l3_2_2048_y.npy"
KNOWN_METADATA_PATH = PROCESSED_DIR / "target_class_l3_2_2048_metadata.csv"

UNSEEN_X_PATH = PROCESSED_DIR / "unseen_l3_min100_2_2048_X.npy"
UNSEEN_METADATA_PATH = PROCESSED_DIR / "unseen_l3_min100_2_2048_metadata.csv"

MODEL_CLASS_SUMMARY_PATH = REPORTS_METRICS_DIR / "unseen_l3_min100_mlp_class_summary.csv"

PAIR_SCORE_PATH = REPORTS_METRICS_DIR / "unseen_l3_min100_direct_tanimoto_pair_scores.csv"
TOP_CLASSES_PATH = REPORTS_METRICS_DIR / "unseen_l3_min100_direct_tanimoto_top_known_classes.csv"
MODEL_AGREEMENT_PATH = REPORTS_METRICS_DIR / "unseen_l3_min100_direct_tanimoto_model_agreement.csv"
PER_MOLECULE_PATH = REPORTS_DIAGNOSTICS_DIR / "unseen_l3_min100_direct_tanimoto_per_molecule.csv"
CONFIG_PATH = REPORTS_METRICS_DIR / "unseen_l3_min100_direct_tanimoto_config.json"

BATCH_SIZE = 256

SIMILARITY_THRESHOLDS = [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
]


def prepare_directories() -> None:
    REPORTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    if not KNOWN_X_PATH.exists():
        raise FileNotFoundError(f"Known X not found: {KNOWN_X_PATH}")

    if not KNOWN_Y_PATH.exists():
        raise FileNotFoundError(f"Known y not found: {KNOWN_Y_PATH}")

    if not KNOWN_METADATA_PATH.exists():
        raise FileNotFoundError(f"Known metadata not found: {KNOWN_METADATA_PATH}")

    if not UNSEEN_X_PATH.exists():
        raise FileNotFoundError(f"Unseen X not found: {UNSEEN_X_PATH}")

    if not UNSEEN_METADATA_PATH.exists():
        raise FileNotFoundError(f"Unseen metadata not found: {UNSEEN_METADATA_PATH}")

    x_known = np.load(KNOWN_X_PATH).astype(np.float32)
    y_known = np.load(KNOWN_Y_PATH).astype(np.int64)
    known_metadata = pd.read_csv(KNOWN_METADATA_PATH)

    x_unseen = np.load(UNSEEN_X_PATH).astype(np.float32)
    unseen_metadata = pd.read_csv(UNSEEN_METADATA_PATH)

    if len(x_known) != len(y_known) or len(y_known) != len(known_metadata):
        raise ValueError(
            f"Known length mismatch: "
            f"X={len(x_known)}, y={len(y_known)}, metadata={len(known_metadata)}"
        )

    if len(x_unseen) != len(unseen_metadata):
        raise ValueError(
            f"Unseen length mismatch: "
            f"X={len(x_unseen)}, metadata={len(unseen_metadata)}"
        )

    label_map = (
        known_metadata[["label_id", "target_class_l3"]]
        .drop_duplicates()
        .sort_values("label_id")
    )

    known_class_names = label_map["target_class_l3"].tolist()

    return x_known, y_known, known_metadata, x_unseen, unseen_metadata, known_class_names


def compute_nearest_tanimoto(
    query_matrix: np.ndarray,
    train_matrix: np.ndarray,
    train_indices_global: np.ndarray,
    batch_size: int,
):
    train_bit_counts = train_matrix.sum(axis=1).astype(np.float32)

    nearest_similarities = []
    nearest_global_indices = []

    for start in range(0, len(query_matrix), batch_size):
        end = min(start + batch_size, len(query_matrix))

        query_batch = query_matrix[start:end]
        query_bit_counts = query_batch.sum(axis=1).astype(np.float32)

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

        nearest_local_positions = np.argmax(similarities, axis=1)
        batch_nearest_similarities = similarities[
            np.arange(len(query_batch)),
            nearest_local_positions,
        ]

        batch_nearest_global_indices = train_indices_global[nearest_local_positions]

        nearest_similarities.append(batch_nearest_similarities)
        nearest_global_indices.append(batch_nearest_global_indices)

    nearest_similarities = np.concatenate(nearest_similarities)
    nearest_global_indices = np.concatenate(nearest_global_indices)

    return nearest_similarities, nearest_global_indices


def summarize_pair(
    unseen_class_name: str,
    known_class_name: str,
    nearest_similarities: np.ndarray,
    n_unseen: int,
    n_known: int,
):
    row = {
        "unseen_target_class_l3": unseen_class_name,
        "known_target_class_l3": known_class_name,
        "unseen_molecules": int(n_unseen),
        "known_molecules": int(n_known),
        "mean_nearest_tanimoto": float(np.mean(nearest_similarities)),
        "median_nearest_tanimoto": float(np.median(nearest_similarities)),
        "std_nearest_tanimoto": float(np.std(nearest_similarities)),
        "min_nearest_tanimoto": float(np.min(nearest_similarities)),
        "q25_nearest_tanimoto": float(np.quantile(nearest_similarities, 0.25)),
        "q75_nearest_tanimoto": float(np.quantile(nearest_similarities, 0.75)),
        "q90_nearest_tanimoto": float(np.quantile(nearest_similarities, 0.90)),
        "max_nearest_tanimoto": float(np.max(nearest_similarities)),
    }

    for threshold in SIMILARITY_THRESHOLDS:
        row[f"fraction_ge_{threshold:.2f}"] = float(
            np.mean(nearest_similarities >= threshold)
        )

    return row


def make_top_class_summary(pair_scores: pd.DataFrame) -> pd.DataFrame:
    ranked = pair_scores.copy()

    ranked["rank_by_median_nearest_tanimoto"] = (
        ranked
        .groupby("unseen_target_class_l3")["median_nearest_tanimoto"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )

    ranked["rank_by_mean_nearest_tanimoto"] = (
        ranked
        .groupby("unseen_target_class_l3")["mean_nearest_tanimoto"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )

    top_rows = []

    for unseen_class, group in ranked.groupby("unseen_target_class_l3"):
        group_sorted = group.sort_values(
            "median_nearest_tanimoto",
            ascending=False,
        ).reset_index(drop=True)

        row = {
            "unseen_target_class_l3": unseen_class,
            "unseen_molecules": int(group_sorted.loc[0, "unseen_molecules"]),
        }

        for i in range(min(3, len(group_sorted))):
            prefix = f"top{i + 1}"

            row[f"{prefix}_known_class_by_tanimoto"] = group_sorted.loc[
                i,
                "known_target_class_l3",
            ]
            row[f"{prefix}_median_nearest_tanimoto"] = group_sorted.loc[
                i,
                "median_nearest_tanimoto",
            ]
            row[f"{prefix}_mean_nearest_tanimoto"] = group_sorted.loc[
                i,
                "mean_nearest_tanimoto",
            ]
            row[f"{prefix}_fraction_ge_0_40"] = group_sorted.loc[
                i,
                "fraction_ge_0.40",
            ]
            row[f"{prefix}_fraction_ge_0_50"] = group_sorted.loc[
                i,
                "fraction_ge_0.50",
            ]

        top_rows.append(row)

    top_summary = pd.DataFrame(top_rows).sort_values(
        "unseen_molecules",
        ascending=False,
    )

    return ranked, top_summary


def make_model_agreement(pair_scores_ranked: pd.DataFrame) -> pd.DataFrame | None:
    if not MODEL_CLASS_SUMMARY_PATH.exists():
        print()
        print(f"Model class summary not found, skipping agreement analysis:")
        print(MODEL_CLASS_SUMMARY_PATH)
        return None

    model_summary = pd.read_csv(MODEL_CLASS_SUMMARY_PATH)

    required_columns = [
        "model_name",
        "unseen_target_class_l3",
        "top1_softmax_mean_known_class",
        "top1_centroid_cosine_known_class",
    ]

    missing = [
        column
        for column in required_columns
        if column not in model_summary.columns
    ]

    if missing:
        print()
        print(f"Model class summary missing columns, skipping agreement: {missing}")
        return None

    rank_lookup = pair_scores_ranked[
        [
            "unseen_target_class_l3",
            "known_target_class_l3",
            "rank_by_median_nearest_tanimoto",
            "median_nearest_tanimoto",
            "mean_nearest_tanimoto",
            "fraction_ge_0.40",
            "fraction_ge_0.50",
        ]
    ].copy()

    softmax_merge = model_summary.merge(
        rank_lookup,
        left_on=[
            "unseen_target_class_l3",
            "top1_softmax_mean_known_class",
        ],
        right_on=[
            "unseen_target_class_l3",
            "known_target_class_l3",
        ],
        how="left",
    )

    softmax_merge = softmax_merge.rename(
        columns={
            "rank_by_median_nearest_tanimoto": "softmax_class_tanimoto_rank",
            "median_nearest_tanimoto": "softmax_class_median_nearest_tanimoto",
            "mean_nearest_tanimoto": "softmax_class_mean_nearest_tanimoto",
            "fraction_ge_0.40": "softmax_class_fraction_ge_0_40",
            "fraction_ge_0.50": "softmax_class_fraction_ge_0_50",
        }
    )

    softmax_merge = softmax_merge.drop(
        columns=["known_target_class_l3"],
        errors="ignore",
    )

    centroid_merge = softmax_merge.merge(
        rank_lookup,
        left_on=[
            "unseen_target_class_l3",
            "top1_centroid_cosine_known_class",
        ],
        right_on=[
            "unseen_target_class_l3",
            "known_target_class_l3",
        ],
        how="left",
    )

    centroid_merge = centroid_merge.rename(
        columns={
            "rank_by_median_nearest_tanimoto": "centroid_class_tanimoto_rank",
            "median_nearest_tanimoto": "centroid_class_median_nearest_tanimoto",
            "mean_nearest_tanimoto": "centroid_class_mean_nearest_tanimoto",
            "fraction_ge_0.40": "centroid_class_fraction_ge_0_40",
            "fraction_ge_0.50": "centroid_class_fraction_ge_0_50",
        }
    )

    centroid_merge = centroid_merge.drop(
        columns=["known_target_class_l3"],
        errors="ignore",
    )

    centroid_merge["softmax_matches_direct_tanimoto_top1"] = (
        centroid_merge["softmax_class_tanimoto_rank"] == 1
    )

    centroid_merge["centroid_matches_direct_tanimoto_top1"] = (
        centroid_merge["centroid_class_tanimoto_rank"] == 1
    )

    selected_columns = [
        "model_name",
        "unseen_target_class_l3",
        "unique_molecules",
        "top1_softmax_mean_known_class",
        "top1_softmax_mean_probability",
        "softmax_class_tanimoto_rank",
        "softmax_class_median_nearest_tanimoto",
        "softmax_matches_direct_tanimoto_top1",
        "top1_centroid_cosine_known_class",
        "top1_centroid_cosine_similarity",
        "centroid_class_tanimoto_rank",
        "centroid_class_median_nearest_tanimoto",
        "centroid_matches_direct_tanimoto_top1",
        "mean_softmax_entropy",
    ]

    selected_columns = [
        column
        for column in selected_columns
        if column in centroid_merge.columns
    ]

    agreement = centroid_merge[selected_columns].copy()

    return agreement


def main() -> None:
    prepare_directories()

    start_time = time.time()

    print("Direct class-to-class Tanimoto analysis")
    print("=" * 80)

    (
        x_known,
        y_known,
        known_metadata,
        x_unseen,
        unseen_metadata,
        known_class_names,
    ) = load_data()

    unseen_class_names = sorted(
        unseen_metadata["target_class_l3"].dropna().astype(str).unique().tolist()
    )

    print(f"Known molecules: {len(x_known)}")
    print(f"Known classes: {len(known_class_names)}")
    print(f"Unseen molecules: {len(x_unseen)}")
    print(f"Unseen classes: {len(unseen_class_names)}")
    print("=" * 80)
    print()

    known_indices_by_class = {
        class_id: np.where(y_known == class_id)[0].astype(np.int64)
        for class_id in range(len(known_class_names))
    }

    unseen_indices_by_class = {
        class_name: unseen_metadata.index[
            unseen_metadata["target_class_l3"].astype(str) == class_name
        ].to_numpy(dtype=np.int64)
        for class_name in unseen_class_names
    }

    pair_rows = []
    per_molecule_rows = []

    total_pairs = len(unseen_class_names) * len(known_class_names)
    pair_counter = 0

    for unseen_class_name in unseen_class_names:
        unseen_indices = unseen_indices_by_class[unseen_class_name]
        query_matrix = x_unseen[unseen_indices].astype(np.float32, copy=False)

        print()
        print(f"Unseen class: {unseen_class_name}")
        print(f"Unseen molecules: {len(unseen_indices)}")
        print("-" * 80)

        for known_class_id, known_class_name in enumerate(known_class_names):
            pair_counter += 1

            known_indices = known_indices_by_class[known_class_id]
            train_matrix = x_known[known_indices].astype(np.float32, copy=False)

            nearest_similarities, nearest_known_indices = compute_nearest_tanimoto(
                query_matrix=query_matrix,
                train_matrix=train_matrix,
                train_indices_global=known_indices,
                batch_size=BATCH_SIZE,
            )

            pair_row = summarize_pair(
                unseen_class_name=unseen_class_name,
                known_class_name=known_class_name,
                nearest_similarities=nearest_similarities,
                n_unseen=len(unseen_indices),
                n_known=len(known_indices),
            )

            pair_rows.append(pair_row)

            for local_position, unseen_row_index in enumerate(unseen_indices):
                nearest_known_row_index = int(nearest_known_indices[local_position])

                per_molecule_rows.append(
                    {
                        "unseen_row_index": int(unseen_row_index),
                        "unseen_molecule_chembl_id": unseen_metadata.loc[
                            unseen_row_index,
                            "molecule_chembl_id",
                        ],
                        "unseen_target_class_l3": unseen_class_name,
                        "known_target_class_l3": known_class_name,
                        "nearest_tanimoto": float(nearest_similarities[local_position]),
                        "nearest_known_row_index": nearest_known_row_index,
                        "nearest_known_molecule_chembl_id": known_metadata.loc[
                            nearest_known_row_index,
                            "molecule_chembl_id",
                        ],
                    }
                )

            print(
                f"[{pair_counter:03d}/{total_pairs:03d}] "
                f"{known_class_name}: "
                f"median={pair_row['median_nearest_tanimoto']:.4f}, "
                f"mean={pair_row['mean_nearest_tanimoto']:.4f}, "
                f"max={pair_row['max_nearest_tanimoto']:.4f}"
            )

    pair_scores = pd.DataFrame(pair_rows)

    pair_scores_ranked, top_summary = make_top_class_summary(pair_scores)

    per_molecule = pd.DataFrame(per_molecule_rows)

    pair_scores_ranked.to_csv(
        PAIR_SCORE_PATH,
        index=False,
        encoding="utf-8",
    )

    top_summary.to_csv(
        TOP_CLASSES_PATH,
        index=False,
        encoding="utf-8",
    )

    per_molecule.to_csv(
        PER_MOLECULE_PATH,
        index=False,
        encoding="utf-8",
    )

    agreement = make_model_agreement(pair_scores_ranked)

    if agreement is not None:
        agreement.to_csv(
            MODEL_AGREEMENT_PATH,
            index=False,
            encoding="utf-8",
        )

    config = {
        "analysis": "direct_class_to_class_tanimoto_similarity",
        "known_x_path": str(KNOWN_X_PATH),
        "known_y_path": str(KNOWN_Y_PATH),
        "known_metadata_path": str(KNOWN_METADATA_PATH),
        "unseen_x_path": str(UNSEEN_X_PATH),
        "unseen_metadata_path": str(UNSEEN_METADATA_PATH),
        "known_classes": known_class_names,
        "unseen_classes": unseen_class_names,
        "batch_size": BATCH_SIZE,
        "similarity_thresholds": SIMILARITY_THRESHOLDS,
        "method": (
            "For each unseen molecule and each known L3 class, "
            "the nearest known molecule by Tanimoto similarity was found. "
            "Class-to-class similarity was summarized by median nearest-neighbor "
            "Tanimoto similarity."
        ),
        "output_files": {
            "pair_score_path": str(PAIR_SCORE_PATH),
            "top_classes_path": str(TOP_CLASSES_PATH),
            "model_agreement_path": str(MODEL_AGREEMENT_PATH),
            "per_molecule_path": str(PER_MOLECULE_PATH),
        },
    }

    with open(CONFIG_PATH, "w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=4)

    elapsed_minutes = (time.time() - start_time) / 60

    print()
    print("Direct Tanimoto analysis completed.")
    print("=" * 80)
    print(f"Pair scores saved to: {PAIR_SCORE_PATH}")
    print(f"Top known classes saved to: {TOP_CLASSES_PATH}")
    print(f"Per-molecule results saved to: {PER_MOLECULE_PATH}")

    if agreement is not None:
        print(f"Model agreement saved to: {MODEL_AGREEMENT_PATH}")

    print(f"Config saved to: {CONFIG_PATH}")
    print(f"Elapsed time: {elapsed_minutes:.2f} min")
    print("=" * 80)
    print()

    print("Top class by direct Tanimoto")
    print("=" * 80)

    preview_columns = [
        "unseen_target_class_l3",
        "unseen_molecules",
        "top1_known_class_by_tanimoto",
        "top1_median_nearest_tanimoto",
        "top1_mean_nearest_tanimoto",
        "top1_fraction_ge_0_40",
        "top2_known_class_by_tanimoto",
        "top2_median_nearest_tanimoto",
        "top3_known_class_by_tanimoto",
        "top3_median_nearest_tanimoto",
    ]

    available_columns = [
        column
        for column in preview_columns
        if column in top_summary.columns
    ]

    print(top_summary[available_columns].to_string(index=False))
    print("=" * 80)

    if agreement is not None:
        print()
        print("Agreement with model outputs")
        print("=" * 80)

        agreement_summary = (
            agreement
            .groupby("model_name")
            .agg(
                unseen_classes=("unseen_target_class_l3", "nunique"),
                softmax_top1_tanimoto_agreement=(
                    "softmax_matches_direct_tanimoto_top1",
                    "mean",
                ),
                centroid_top1_tanimoto_agreement=(
                    "centroid_matches_direct_tanimoto_top1",
                    "mean",
                ),
                mean_softmax_tanimoto_rank=(
                    "softmax_class_tanimoto_rank",
                    "mean",
                ),
                mean_centroid_tanimoto_rank=(
                    "centroid_class_tanimoto_rank",
                    "mean",
                ),
            )
            .reset_index()
        )

        print(agreement_summary.to_string(index=False))
        print("=" * 80)


if __name__ == "__main__":
    main()