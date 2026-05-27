from pathlib import Path
import time

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator


PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "target_class_l3_single_label_dataset.csv"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports" / "metrics"

X_OUTPUT_PATH = PROCESSED_DIR / "target_class_l3_2_2048_X.npy"
Y_OUTPUT_PATH = PROCESSED_DIR / "target_class_l3_2_2048_y.npy"
METADATA_OUTPUT_PATH = PROCESSED_DIR / "target_class_l3_2_2048_metadata.csv"

SUMMARY_OUTPUT_PATH = REPORTS_DIR / "target_class_l3_2_2048_summary.csv"
INVALID_SMILES_OUTPUT_PATH = REPORTS_DIR / "target_class_l3_2_2048_invalid_smiles.csv"

MORGAN_RADIUS = 2
FINGERPRINT_SIZE = 2048

TARGET_CLASS_COLUMN = "target_class_l3"
LABEL_COLUMN = "label_id"

REQUIRED_COLUMNS = [
    "molregno",
    "molecule_chembl_id",
    "canonical_smiles",
    TARGET_CLASS_COLUMN,
    LABEL_COLUMN,
]


def prepare_directories() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def validate_input_columns(data: pd.DataFrame) -> None:
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def create_morgan_generator():
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS,
        fpSize=FINGERPRINT_SIZE,
    )


def smiles_to_fingerprint(smiles: str, generator) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    fingerprint = generator.GetFingerprint(mol)

    array = np.zeros((FINGERPRINT_SIZE,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fingerprint, array)

    return array


def make_summary(
    input_rows: int,
    valid_rows: int,
    invalid_rows: int,
    n_classes: int,
    elapsed_minutes: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "input_dataset": str(INPUT_DATASET_PATH),
                "fingerprint_type": "Morgan",
                "radius": MORGAN_RADIUS,
                "fingerprint_size": FINGERPRINT_SIZE,
                "input_rows": input_rows,
                "valid_rows": valid_rows,
                "invalid_rows": invalid_rows,
                "n_classes": n_classes,
                "x_output_path": str(X_OUTPUT_PATH),
                "y_output_path": str(Y_OUTPUT_PATH),
                "metadata_output_path": str(METADATA_OUTPUT_PATH),
                "elapsed_minutes": elapsed_minutes,
            }
        ]
    )


def main() -> None:
    RDLogger.DisableLog("rdApp.*")

    if not INPUT_DATASET_PATH.exists():
        raise FileNotFoundError(f"Input dataset not found: {INPUT_DATASET_PATH}")

    prepare_directories()

    start_time = time.time()

    print("Morgan fingerprint generation")
    print("=" * 80)
    print(f"Input dataset: {INPUT_DATASET_PATH}")
    print(f"Radius: {MORGAN_RADIUS}")
    print(f"Fingerprint size: {FINGERPRINT_SIZE}")
    print("=" * 80)
    print()

    data = pd.read_csv(INPUT_DATASET_PATH)
    validate_input_columns(data)

    data[LABEL_COLUMN] = pd.to_numeric(
        data[LABEL_COLUMN],
        errors="raise",
    ).astype(np.int64)

    generator = create_morgan_generator()

    fingerprints = []
    labels = []
    metadata_rows = []
    invalid_rows = []

    total_rows = len(data)

    for row_number, row in enumerate(data.itertuples(index=False), start=1):
        row_dict = row._asdict()

        smiles = row_dict["canonical_smiles"]
        fingerprint = smiles_to_fingerprint(smiles, generator)

        if fingerprint is None:
            invalid_rows.append(row_dict)
            continue

        fingerprints.append(fingerprint)
        labels.append(int(row_dict[LABEL_COLUMN]))

        metadata_rows.append(
            {
                "molregno": row_dict["molregno"],
                "molecule_chembl_id": row_dict["molecule_chembl_id"],
                "canonical_smiles": row_dict["canonical_smiles"],
                TARGET_CLASS_COLUMN: row_dict[TARGET_CLASS_COLUMN],
                LABEL_COLUMN: int(row_dict[LABEL_COLUMN]),
                "max_pchembl_value": row_dict.get("max_pchembl_value", np.nan),
                "mean_pchembl_value": row_dict.get("mean_pchembl_value", np.nan),
                "activity_records_for_label": row_dict.get(
                    "activity_records_for_label",
                    np.nan,
                ),
                "unique_targets_for_label": row_dict.get(
                    "unique_targets_for_label",
                    np.nan,
                ),
                "best_target_chembl_id": row_dict.get("best_target_chembl_id", ""),
                "best_target_name": row_dict.get("best_target_name", ""),
                "best_standard_type": row_dict.get("best_standard_type", ""),
                "best_pchembl_value": row_dict.get("best_pchembl_value", np.nan),
            }
        )

        if row_number % 5000 == 0:
            elapsed_minutes = (time.time() - start_time) / 60
            print(
                f"Processed {row_number}/{total_rows} rows, "
                f"valid fingerprints: {len(fingerprints)}, "
                f"elapsed: {elapsed_minutes:.1f} min"
            )

    if not fingerprints:
        raise RuntimeError("No valid fingerprints were generated.")

    x = np.vstack(fingerprints).astype(np.uint8)
    y = np.asarray(labels, dtype=np.int64)
    metadata = pd.DataFrame(metadata_rows)

    invalid_smiles = pd.DataFrame(invalid_rows)

    np.save(X_OUTPUT_PATH, x)
    np.save(Y_OUTPUT_PATH, y)
    metadata.to_csv(METADATA_OUTPUT_PATH, index=False, encoding="utf-8")

    if not invalid_smiles.empty:
        invalid_smiles.to_csv(
            INVALID_SMILES_OUTPUT_PATH,
            index=False,
            encoding="utf-8",
        )

    elapsed_minutes = (time.time() - start_time) / 60

    summary = make_summary(
        input_rows=total_rows,
        valid_rows=len(metadata),
        invalid_rows=len(invalid_smiles),
        n_classes=int(metadata[LABEL_COLUMN].nunique()),
        elapsed_minutes=elapsed_minutes,
    )

    summary.to_csv(SUMMARY_OUTPUT_PATH, index=False, encoding="utf-8")

    print()
    print("Fingerprint generation completed.")
    print(f"Input rows: {total_rows}")
    print(f"Valid fingerprints: {len(metadata)}")
    print(f"Invalid SMILES: {len(invalid_smiles)}")
    print(f"Classes: {metadata[LABEL_COLUMN].nunique()}")
    print(f"X shape: {x.shape}")
    print(f"y shape: {y.shape}")
    print(f"Elapsed time: {elapsed_minutes:.1f} min")
    print()
    print(f"X saved to: {X_OUTPUT_PATH}")
    print(f"y saved to: {Y_OUTPUT_PATH}")
    print(f"Metadata saved to: {METADATA_OUTPUT_PATH}")
    print(f"Summary saved to: {SUMMARY_OUTPUT_PATH}")

    if not invalid_smiles.empty:
        print(f"Invalid SMILES saved to: {INVALID_SMILES_OUTPUT_PATH}")


if __name__ == "__main__":
    main()