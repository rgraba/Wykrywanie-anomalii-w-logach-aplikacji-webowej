import hashlib
import json

import pandas as pd
from sklearn.model_selection import (
    StratifiedGroupKFold,
)

import config
from podzial_danych import (
    add_request_groups,
    ensure_no_group_overlap,
    validate_request_groups,
)


PROTOCOL_VERSION = 1


def calculate_dataset_fingerprint(
    df: pd.DataFrame,
) -> str:
    fingerprint_data = pd.DataFrame(
        {
            "row_id": df.index,
            "request_group": (
                df["request_group"].to_numpy()
            ),
            "classification": (
                df["classification"].to_numpy()
            ),
        }
    )

    payload = fingerprint_data.to_csv(
        index=False,
        lineterminator="\n",
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def create_and_save_experimental_protocol(
    df: pd.DataFrame,
) -> pd.DataFrame:
    group_audit = validate_request_groups(df)

    grouped_df = add_request_groups(df)

    y = grouped_df["classification"]
    groups = grouped_df["request_group"]

    holdout_splitter = StratifiedGroupKFold(
        n_splits=config.FINAL_TEST_N_SPLITS,
        shuffle=True,
        random_state=(
            config.PROTOCOL_RANDOM_STATE
        ),
    )

    (
        development_positions,
        final_test_positions,
    ) = next(
        holdout_splitter.split(
            grouped_df,
            y,
            groups=groups,
        )
    )

    development_indices = grouped_df.index[
        development_positions
    ]

    final_test_indices = grouped_df.index[
        final_test_positions
    ]

    ensure_no_group_overlap(
        df=grouped_df,
        first_indices=development_indices,
        second_indices=final_test_indices,
        first_name="development",
        second_name="final_test",
    )

    if (
        len(development_indices)
        + len(final_test_indices)
        != len(grouped_df)
    ):
        raise RuntimeError(
            "Podział development/final_test "
            "nie obejmuje wszystkich rekordów."
        )

    manifest = pd.DataFrame(
        {
            "row_id": (
                grouped_df.index.to_numpy()
            ),
            "request_group": (
                groups.to_numpy()
            ),
            "classification": (
                y.to_numpy()
            ),
            "partition": "development",
        }
    )

    manifest["cv_fold"] = pd.Series(
        pd.NA,
        index=manifest.index,
        dtype="Int64",
    )

    manifest.loc[
        manifest["row_id"].isin(
            final_test_indices
        ),
        "partition",
    ] = "final_test"

    development_df = grouped_df.loc[
        development_indices
    ]

    cv_splitter = StratifiedGroupKFold(
        n_splits=(
            config.DEVELOPMENT_CV_N_SPLITS
        ),
        shuffle=True,
        random_state=(
            config.PROTOCOL_RANDOM_STATE
        ),
    )

    for fold_number, (
        train_positions,
        validation_positions,
    ) in enumerate(
        cv_splitter.split(
            development_df,
            development_df["classification"],
            groups=(
                development_df["request_group"]
            ),
        ),
        start=1,
    ):
        train_indices = development_df.index[
            train_positions
        ]

        validation_indices = (
            development_df.index[
                validation_positions
            ]
        )

        group_overlap = ensure_no_group_overlap(
            df=grouped_df,
            first_indices=train_indices,
            second_indices=validation_indices,
            first_name=(
                f"fold_{fold_number}_train"
            ),
            second_name=(
                f"fold_{fold_number}_validation"
            ),
        )

        if group_overlap != 0:
            raise RuntimeError(
                f"Nieprawidłowy overlap w foldzie "
                f"{fold_number}."
            )

        manifest.loc[
            manifest["row_id"].isin(
                validation_indices
            ),
            "cv_fold",
        ] = fold_number

    development_manifest = manifest.loc[
        manifest["partition"]
        == "development"
    ]

    final_test_manifest = manifest.loc[
        manifest["partition"]
        == "final_test"
    ]

    if development_manifest[
        "cv_fold"
    ].isna().any():
        raise RuntimeError(
            "Nie wszystkie rekordy development "
            "mają przypisany fold."
        )

    if final_test_manifest[
        "cv_fold"
    ].notna().any():
        raise RuntimeError(
            "Final test nie może mieć "
            "przypisanego foldu CV."
        )

    partitions_per_group = (
        manifest
        .groupby("request_group")[
            "partition"
        ]
        .nunique()
    )

    if partitions_per_group.max() != 1:
        raise RuntimeError(
            "Jedna grupa trafiła do więcej "
            "niż jednej partycji."
        )

    folds_per_group = (
        development_manifest
        .groupby("request_group")[
            "cv_fold"
        ]
        .nunique()
    )

    if folds_per_group.max() != 1:
        raise RuntimeError(
            "Jedna grupa trafiła do więcej "
            "niż jednego foldu walidacyjnego."
        )

    config.SPLITS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest.to_csv(
        config.EXPERIMENTAL_PROTOCOL_FILE,
        index=False,
    )

    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset_fingerprint": (
            calculate_dataset_fingerprint(
                grouped_df
            )
        ),
        "group_definition": (
            "SHA-256(JSON([Method, URL, content]))"
        ),
        "protocol_random_state": (
            config.PROTOCOL_RANDOM_STATE
        ),
        "final_test_n_splits": (
            config.FINAL_TEST_N_SPLITS
        ),
        "development_cv_n_splits": (
            config.DEVELOPMENT_CV_N_SPLITS
        ),
        "total_records": len(grouped_df),
        "development_records": len(
            development_indices
        ),
        "final_test_records": len(
            final_test_indices
        ),
        "group_audit": group_audit,
    }

    config.EXPERIMENTAL_PROTOCOL_METADATA_FILE.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return manifest


def load_experimental_protocol(
    df: pd.DataFrame,
) -> pd.DataFrame:
    if not config.EXPERIMENTAL_PROTOCOL_FILE.exists():
        raise FileNotFoundError(
            "Nie znaleziono manifestu protokołu. "
            "Najpierw wygeneruj protokół "
            "eksperymentalny."
        )

    if not (
        config
        .EXPERIMENTAL_PROTOCOL_METADATA_FILE
        .exists()
    ):
        raise FileNotFoundError(
            "Nie znaleziono metadanych protokołu."
        )

    manifest = pd.read_csv(
        config.EXPERIMENTAL_PROTOCOL_FILE,
        dtype={
            "cv_fold": "Int64",
        },
    )

    required_columns = {
        "row_id",
        "request_group",
        "classification",
        "partition",
        "cv_fold",
    }

    missing_columns = (
        required_columns.difference(
            manifest.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Manifest nie zawiera kolumn: "
            f"{sorted(missing_columns)}."
        )

    if manifest["row_id"].duplicated().any():
        raise ValueError(
            "Manifest zawiera powtórzone row_id."
        )

    if len(manifest) != len(df):
        raise ValueError(
            "Liczba rekordów manifestu "
            "nie odpowiada danym."
        )

    if set(manifest["row_id"]) != set(df.index):
        raise ValueError(
            "Indeksy manifestu "
            "nie odpowiadają danym."
        )

    grouped_df = add_request_groups(df)

    ordered_manifest = (
        manifest
        .set_index("row_id")
        .reindex(grouped_df.index)
    )

    ordered_manifest.index.name = "row_id"

    ordered_manifest = (
        ordered_manifest.reset_index()
    )

    groups_are_equal = (
        ordered_manifest[
            "request_group"
        ].to_numpy()
        == grouped_df[
            "request_group"
        ].to_numpy()
    ).all()

    if not groups_are_equal:
        raise ValueError(
            "Identyfikatory grup w manifeście "
            "są nieaktualne."
        )

    labels_are_equal = (
        ordered_manifest[
            "classification"
        ].to_numpy()
        == grouped_df[
            "classification"
        ].to_numpy()
    ).all()

    if not labels_are_equal:
        raise ValueError(
            "Etykiety w manifeście "
            "nie odpowiadają danym."
        )

    metadata = json.loads(
        config
        .EXPERIMENTAL_PROTOCOL_METADATA_FILE
        .read_text(encoding="utf-8")
    )

    if (
        metadata.get("protocol_version")
        != PROTOCOL_VERSION
    ):
        raise ValueError(
            "Nieobsługiwana wersja protokołu."
        )

    current_fingerprint = (
        calculate_dataset_fingerprint(
            grouped_df
        )
    )

    if (
        metadata.get("dataset_fingerprint")
        != current_fingerprint
    ):
        raise ValueError(
            "Dane zmieniły się od czasu "
            "wygenerowania protokołu."
        )

    allowed_partitions = {
        "development",
        "final_test",
    }

    actual_partitions = set(
        ordered_manifest[
            "partition"
        ].unique()
    )

    if actual_partitions != allowed_partitions:
        raise ValueError(
            "Manifest zawiera "
            "nieprawidłowe partycje."
        )

    development_manifest = (
        ordered_manifest.loc[
            ordered_manifest["partition"]
            == "development"
        ]
    )

    final_test_manifest = (
        ordered_manifest.loc[
            ordered_manifest["partition"]
            == "final_test"
        ]
    )

    if development_manifest[
        "cv_fold"
    ].isna().any():
        raise ValueError(
            "Rekord development nie ma "
            "przypisanego foldu."
        )

    if final_test_manifest[
        "cv_fold"
    ].notna().any():
        raise ValueError(
            "Rekord final_test ma "
            "przypisany fold."
        )

    expected_folds = set(
        range(
            1,
            (
                config
                .DEVELOPMENT_CV_N_SPLITS
                + 1
            ),
        )
    )

    actual_folds = set(
        development_manifest[
            "cv_fold"
        ].astype(int).unique()
    )

    if actual_folds != expected_folds:
        raise ValueError(
            "Nieprawidłowy zestaw foldów: "
            f"{sorted(actual_folds)}."
        )

    development_indices = pd.Index(
        development_manifest["row_id"]
    )

    final_test_indices = pd.Index(
        final_test_manifest["row_id"]
    )

    ensure_no_group_overlap(
        df=grouped_df,
        first_indices=development_indices,
        second_indices=final_test_indices,
        first_name="development",
        second_name="final_test",
    )

    for fold_number in sorted(
        expected_folds
    ):
        validation_indices = pd.Index(
            development_manifest.loc[
                development_manifest["cv_fold"]
                == fold_number,
                "row_id",
            ]
        )

        train_indices = pd.Index(
            development_manifest.loc[
                development_manifest["cv_fold"]
                != fold_number,
                "row_id",
            ]
        )

        ensure_no_group_overlap(
            df=grouped_df,
            first_indices=train_indices,
            second_indices=(
                validation_indices
            ),
            first_name=(
                f"fold_{fold_number}_train"
            ),
            second_name=(
                f"fold_{fold_number}_validation"
            ),
        )

    return ordered_manifest


def get_development_and_final_test_indices(
    df: pd.DataFrame,
) -> tuple[pd.Index, pd.Index]:
    manifest = load_experimental_protocol(df)

    development_indices = pd.Index(
        manifest.loc[
            manifest["partition"]
            == "development",
            "row_id",
        ]
    )

    final_test_indices = pd.Index(
        manifest.loc[
            manifest["partition"]
            == "final_test",
            "row_id",
        ]
    )

    return (
        development_indices,
        final_test_indices,
    )


def get_development_folds(
    df: pd.DataFrame,
) -> list[
    tuple[
        int,
        pd.Index,
        pd.Index,
    ]
]:
    manifest = load_experimental_protocol(df)

    development_manifest = manifest.loc[
        manifest["partition"]
        == "development"
    ]

    folds = []

    for fold_number in range(
        1,
        config.DEVELOPMENT_CV_N_SPLITS + 1,
    ):
        validation_indices = pd.Index(
            development_manifest.loc[
                development_manifest["cv_fold"]
                == fold_number,
                "row_id",
            ]
        )

        train_indices = pd.Index(
            development_manifest.loc[
                development_manifest["cv_fold"]
                != fold_number,
                "row_id",
            ]
        )

        folds.append(
            (
                fold_number,
                train_indices,
                validation_indices,
            )
        )

    return folds