from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

RAW_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "chembl_target_class_activities_large.csv"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports" / "metrics"

OUTPUT_DATASET_PATH = (
    PROCESSED_DIR
    / "target_class_l3_single_label_dataset.csv"
)

ALL_CLASS_STATS_PATH = (
    REPORTS_DIR
    / "target_class_l3_all_class_stats.csv"
)

SELECTED_CLASS_STATS_PATH = (
    REPORTS_DIR
    / "target_class_l3_selected_class_stats_before_single_label.csv"
)

FINAL_CLASS_DISTRIBUTION_PATH = (
    REPORTS_DIR
    / "target_class_l3_single_label_final_class_distribution.csv"
)

LABEL_MAP_PATH = (
    REPORTS_DIR
    / "target_class_l3_single_label_label_map.csv"
)

OVERVIEW_PATH = (
    REPORTS_DIR
    / "target_class_l3_single_label_overview.csv"
)

MIN_UNIQUE_MOLECULES = 1000

TARGET_CLASS_COLUMN = "target_class_l3"


def prepare_directories() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def make_class_stats(data: pd.DataFrame, class_column: str) -> pd.DataFrame:
    stats = (
        data
        .dropna(subset=[class_column])
        .groupby(class_column)
        .agg(
            activity_records=("activity_id", "count"),
            unique_molecules=("molregno", "nunique"),
            unique_targets=("tid", "nunique"),
            mean_pchembl=("pchembl_value", "mean"),
            median_pchembl=("pchembl_value", "median"),
            min_pchembl=("pchembl_value", "min"),
            max_pchembl=("pchembl_value", "max"),
        )
        .reset_index()
        .sort_values(
            by=["unique_molecules", "activity_records"],
            ascending=False,
        )
    )

    return stats


def make_overview(
    raw_data: pd.DataFrame,
    l3_data: pd.DataFrame,
    selected_data: pd.DataFrame,
    molecule_class_pairs: pd.DataFrame,
    final_dataset: pd.DataFrame,
    all_class_stats: pd.DataFrame,
    selected_class_stats: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {
            "metric": "raw_activity_records",
            "value": len(raw_data),
        },
        {
            "metric": "raw_unique_molecules",
            "value": raw_data["molregno"].nunique(),
        },
        {
            "metric": "records_with_l3_class",
            "value": len(l3_data),
        },
        {
            "metric": "unique_molecules_with_l3_class",
            "value": l3_data["molregno"].nunique(),
        },
        {
            "metric": "all_l3_classes",
            "value": len(all_class_stats),
        },
        {
            "metric": "min_unique_molecules_threshold",
            "value": MIN_UNIQUE_MOLECULES,
        },
        {
            "metric": "threshold_rule",
            "value": f"unique_molecules > {MIN_UNIQUE_MOLECULES}",
        },
        {
            "metric": "selected_l3_classes",
            "value": len(selected_class_stats),
        },
        {
            "metric": "selected_activity_records",
            "value": len(selected_data),
        },
        {
            "metric": "selected_unique_molecules",
            "value": selected_data["molregno"].nunique(),
        },
        {
            "metric": "molecule_class_pairs",
            "value": len(molecule_class_pairs),
        },
        {
            "metric": "final_single_label_molecules",
            "value": len(final_dataset),
        },
        {
            "metric": "final_classes",
            "value": final_dataset[TARGET_CLASS_COLUMN].nunique(),
        },
    ]

    return pd.DataFrame(rows)


def main() -> None:
    if not RAW_CSV_PATH.exists():
        raise FileNotFoundError(f"Input CSV file not found: {RAW_CSV_PATH}")

    prepare_directories()

    use_columns = [
        "activity_id",
        "molregno",
        "molecule_chembl_id",
        "canonical_smiles",
        "pchembl_value",
        "standard_type",
        "tid",
        "target_chembl_id",
        "target_name",
        TARGET_CLASS_COLUMN,
    ]

    print(f"Loading input CSV: {RAW_CSV_PATH}")

    data = pd.read_csv(RAW_CSV_PATH, usecols=use_columns)

    data["pchembl_value"] = pd.to_numeric(
        data["pchembl_value"],
        errors="coerce",
    )

    data = data.dropna(
        subset=[
            "molregno",
            "molecule_chembl_id",
            "canonical_smiles",
            "pchembl_value",
        ]
    ).copy()

    print(f"Raw rows after basic cleanup: {len(data)}")
    print(f"Raw unique molecules: {data['molregno'].nunique()}")

    l3_data = data.dropna(subset=[TARGET_CLASS_COLUMN]).copy()

    print(f"Rows with {TARGET_CLASS_COLUMN}: {len(l3_data)}")
    print(f"Unique molecules with {TARGET_CLASS_COLUMN}: {l3_data['molregno'].nunique()}")

    all_class_stats = make_class_stats(l3_data, TARGET_CLASS_COLUMN)

    selected_class_stats = all_class_stats[
        all_class_stats["unique_molecules"] > MIN_UNIQUE_MOLECULES
    ].copy()

    selected_classes = selected_class_stats[TARGET_CLASS_COLUMN].tolist()

    print()
    print(f"Selected classes with unique_molecules > {MIN_UNIQUE_MOLECULES}:")
    print(selected_class_stats.to_string(index=False))

    selected_data = l3_data[
        l3_data[TARGET_CLASS_COLUMN].isin(selected_classes)
    ].copy()

    print()
    print(f"Rows after selected class filter: {len(selected_data)}")
    print(f"Unique molecules after selected class filter: {selected_data['molregno'].nunique()}")

    molecule_class_pairs = (
        selected_data
        .groupby(
            [
                "molregno",
                "molecule_chembl_id",
                "canonical_smiles",
                TARGET_CLASS_COLUMN,
            ],
            dropna=False,
        )
        .agg(
            max_pchembl_value=("pchembl_value", "max"),
            mean_pchembl_value=("pchembl_value", "mean"),
            activity_records_for_label=("activity_id", "count"),
            unique_targets_for_label=("tid", "nunique"),
        )
        .reset_index()
    )

    best_records = (
        selected_data
        .sort_values(
            by=[
                "molregno",
                TARGET_CLASS_COLUMN,
                "pchembl_value",
                "activity_id",
            ],
            ascending=[True, True, False, True],
        )
        .drop_duplicates(
            subset=["molregno", TARGET_CLASS_COLUMN],
            keep="first",
        )
        [
            [
                "molregno",
                TARGET_CLASS_COLUMN,
                "activity_id",
                "tid",
                "target_chembl_id",
                "target_name",
                "standard_type",
                "pchembl_value",
            ]
        ]
        .rename(
            columns={
                "activity_id": "best_activity_id",
                "tid": "best_tid",
                "target_chembl_id": "best_target_chembl_id",
                "target_name": "best_target_name",
                "standard_type": "best_standard_type",
                "pchembl_value": "best_pchembl_value",
            }
        )
    )

    molecule_class_pairs = molecule_class_pairs.merge(
        best_records,
        on=["molregno", TARGET_CLASS_COLUMN],
        how="left",
    )

    molecule_class_pairs = molecule_class_pairs.sort_values(
        by=[
            "molregno",
            "max_pchembl_value",
            "activity_records_for_label",
            "unique_targets_for_label",
            TARGET_CLASS_COLUMN,
        ],
        ascending=[True, False, False, False, True],
    )

    final_dataset = molecule_class_pairs.drop_duplicates(
        subset=["molregno"],
        keep="first",
    ).copy()

    final_class_distribution = (
        final_dataset
        .groupby(TARGET_CLASS_COLUMN)
        .agg(
            final_molecules=("molregno", "count"),
            mean_max_pchembl=("max_pchembl_value", "mean"),
            median_max_pchembl=("max_pchembl_value", "median"),
            mean_activity_records_for_label=("activity_records_for_label", "mean"),
            mean_unique_targets_for_label=("unique_targets_for_label", "mean"),
        )
        .reset_index()
        .sort_values(
            by=["final_molecules", TARGET_CLASS_COLUMN],
            ascending=[False, True],
        )
    )

    label_map = final_class_distribution[
        [TARGET_CLASS_COLUMN, "final_molecules"]
    ].copy()

    label_map = label_map.reset_index(drop=True)
    label_map["label_id"] = label_map.index

    final_dataset = final_dataset.merge(
        label_map[[TARGET_CLASS_COLUMN, "label_id"]],
        on=TARGET_CLASS_COLUMN,
        how="left",
    )

    final_dataset = final_dataset[
        [
            "molregno",
            "molecule_chembl_id",
            "canonical_smiles",
            TARGET_CLASS_COLUMN,
            "label_id",
            "max_pchembl_value",
            "mean_pchembl_value",
            "activity_records_for_label",
            "unique_targets_for_label",
            "best_activity_id",
            "best_tid",
            "best_target_chembl_id",
            "best_target_name",
            "best_standard_type",
            "best_pchembl_value",
        ]
    ].sort_values(
        by=["label_id", "molregno"],
        ascending=[True, True],
    )

    overview = make_overview(
        raw_data=data,
        l3_data=l3_data,
        selected_data=selected_data,
        molecule_class_pairs=molecule_class_pairs,
        final_dataset=final_dataset,
        all_class_stats=all_class_stats,
        selected_class_stats=selected_class_stats,
    )

    final_dataset.to_csv(
        OUTPUT_DATASET_PATH,
        index=False,
        encoding="utf-8",
    )

    all_class_stats.to_csv(
        ALL_CLASS_STATS_PATH,
        index=False,
        encoding="utf-8",
    )

    selected_class_stats.to_csv(
        SELECTED_CLASS_STATS_PATH,
        index=False,
        encoding="utf-8",
    )

    final_class_distribution.to_csv(
        FINAL_CLASS_DISTRIBUTION_PATH,
        index=False,
        encoding="utf-8",
    )

    label_map.to_csv(
        LABEL_MAP_PATH,
        index=False,
        encoding="utf-8",
    )

    overview.to_csv(
        OVERVIEW_PATH,
        index=False,
        encoding="utf-8",
    )

    print()
    print("Final dataset overview:")
    print(overview.to_string(index=False))

    print()
    print("Final class distribution:")
    print(final_class_distribution.to_string(index=False))

    print()
    print(f"Dataset saved to: {OUTPUT_DATASET_PATH}")
    print(f"Label map saved to: {LABEL_MAP_PATH}")
    print(f"Final class distribution saved to: {FINAL_CLASS_DISTRIBUTION_PATH}")
    print(f"Overview saved to: {OVERVIEW_PATH}")


if __name__ == "__main__":
    main()