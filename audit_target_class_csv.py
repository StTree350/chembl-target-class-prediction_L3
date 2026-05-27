from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

RAW_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "chembl_target_class_activities_large.csv"
)

REPORTS_DIR = PROJECT_ROOT / "reports" / "metrics"

OVERVIEW_PATH = REPORTS_DIR / "target_class_large_overview.csv"
L1_STATS_PATH = REPORTS_DIR / "target_class_large_l1_stats.csv"
L2_STATS_PATH = REPORTS_DIR / "target_class_large_l2_stats.csv"
L3_STATS_PATH = REPORTS_DIR / "target_class_large_l3_stats.csv"
L4_STATS_PATH = REPORTS_DIR / "target_class_large_l4_stats.csv"
L5_STATS_PATH = REPORTS_DIR / "target_class_large_l5_stats.csv"

POTENTIAL_LABEL_COLUMNS = [
    "target_class_l1",
    "target_class_l2",
    "target_class_l3",
    "target_class_l4",
    "target_class_l5",
]


def make_class_stats(data: pd.DataFrame, class_column: str) -> pd.DataFrame:
    filtered = data.dropna(subset=[class_column]).copy()

    if filtered.empty:
        return pd.DataFrame(
            columns=[
                class_column,
                "activity_records",
                "unique_molecules",
                "unique_targets",
                "mean_pchembl",
                "median_pchembl",
                "min_pchembl",
                "max_pchembl",
            ]
        )

    stats = (
        filtered
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


def make_overview(data: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "metric": "activity_records",
            "value": len(data),
        },
        {
            "metric": "unique_molecules",
            "value": data["molregno"].nunique(),
        },
        {
            "metric": "unique_molecule_chembl_ids",
            "value": data["molecule_chembl_id"].nunique(),
        },
        {
            "metric": "unique_targets",
            "value": data["tid"].nunique(),
        },
        {
            "metric": "unique_target_chembl_ids",
            "value": data["target_chembl_id"].nunique(),
        },
        {
            "metric": "unique_standard_types",
            "value": data["standard_type"].nunique(dropna=True),
        },
        {
            "metric": "mean_pchembl_value",
            "value": data["pchembl_value"].mean(),
        },
        {
            "metric": "median_pchembl_value",
            "value": data["pchembl_value"].median(),
        },
        {
            "metric": "min_pchembl_value",
            "value": data["pchembl_value"].min(),
        },
        {
            "metric": "max_pchembl_value",
            "value": data["pchembl_value"].max(),
        },
    ]

    for column in POTENTIAL_LABEL_COLUMNS:
        rows.append(
            {
                "metric": f"unique_{column}",
                "value": data[column].nunique(dropna=True),
            }
        )

        rows.append(
            {
                "metric": f"non_null_{column}",
                "value": data[column].notna().sum(),
            }
        )

    return pd.DataFrame(rows)


def make_standard_type_stats(data: pd.DataFrame) -> pd.DataFrame:
    stats = (
        data
        .groupby("standard_type")
        .agg(
            activity_records=("activity_id", "count"),
            unique_molecules=("molregno", "nunique"),
            unique_targets=("tid", "nunique"),
            mean_pchembl=("pchembl_value", "mean"),
            median_pchembl=("pchembl_value", "median"),
        )
        .reset_index()
        .sort_values("activity_records", ascending=False)
    )

    return stats


def main() -> None:
    if not RAW_CSV_PATH.exists():
        raise FileNotFoundError(f"CSV file not found: {RAW_CSV_PATH}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading CSV: {RAW_CSV_PATH}")

    data = pd.read_csv(RAW_CSV_PATH)

    overview = make_overview(data)
    l1_stats = make_class_stats(data, "target_class_l1")
    l2_stats = make_class_stats(data, "target_class_l2")
    l3_stats = make_class_stats(data, "target_class_l3")
    l4_stats = make_class_stats(data, "target_class_l4")
    l5_stats = make_class_stats(data, "target_class_l5")
    standard_type_stats = make_standard_type_stats(data)

    overview.to_csv(OVERVIEW_PATH, index=False, encoding="utf-8")
    l1_stats.to_csv(L1_STATS_PATH, index=False, encoding="utf-8")
    l2_stats.to_csv(L2_STATS_PATH, index=False, encoding="utf-8")
    l3_stats.to_csv(L3_STATS_PATH, index=False, encoding="utf-8")
    l4_stats.to_csv(L4_STATS_PATH, index=False, encoding="utf-8")
    l5_stats.to_csv(L5_STATS_PATH, index=False, encoding="utf-8")
    standard_type_stats.to_csv(
        REPORTS_DIR / "target_class_large_standard_type_stats.csv",
        index=False,
        encoding="utf-8",
    )

    print()
    print("Dataset overview:")
    print(overview.to_string(index=False))

    print()
    print("Standard type stats:")
    print(standard_type_stats.to_string(index=False))

    print()
    print("Top target_class_l1:")
    print(l1_stats.head(20).to_string(index=False))

    print()
    print("Top target_class_l2:")
    print(l2_stats.head(30).to_string(index=False))

    print()
    print("Top target_class_l3:")
    print(l3_stats.head(30).to_string(index=False))

    print()
    print(f"Overview saved to: {OVERVIEW_PATH}")
    print(f"L1 stats saved to: {L1_STATS_PATH}")
    print(f"L2 stats saved to: {L2_STATS_PATH}")
    print(f"L3 stats saved to: {L3_STATS_PATH}")
    print(f"L4 stats saved to: {L4_STATS_PATH}")
    print(f"L5 stats saved to: {L5_STATS_PATH}")


if __name__ == "__main__":
    main()