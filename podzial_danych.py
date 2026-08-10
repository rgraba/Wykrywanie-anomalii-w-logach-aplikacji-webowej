import hashlib

import pandas as pd
from sklearn.model_selection import (
    StratifiedGroupKFold,
    train_test_split,
)

import config


def calculate_request_hash(request_text: str) -> str:
    normalized_text = str(request_text)

    return hashlib.sha256(
        normalized_text.encode("utf-8", errors="replace")
    ).hexdigest()


def add_request_groups(df: pd.DataFrame) -> pd.DataFrame:
    if "request_text" not in df.columns:
        raise ValueError("Brakuje kolumny request_text.")

    result = df.copy()

    result["request_group"] = result["request_text"].apply(
        calculate_request_hash
    )

    return result


def create_and_save_group_split(df: pd.DataFrame) -> pd.DataFrame:
    grouped_df = add_request_groups(df)

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=config.RANDOM_STATE,
    )

    train_positions, test_positions = next(
        splitter.split(
            X=grouped_df,
            y=grouped_df["classification"],
            groups=grouped_df["request_group"],
        )
    )

    assignments = pd.DataFrame(
        {
            "row_id": grouped_df.index.to_numpy(),
            "request_group": grouped_df["request_group"].to_numpy(),
            "classification": grouped_df["classification"].to_numpy(),
            "split": "train",
        }
    )

    split_column = assignments.columns.get_loc("split")

    assignments.iloc[test_positions, split_column] = "test"

    train_groups = set(
        assignments.loc[
            assignments["split"] == "train",
            "request_group",
        ]
    )

    test_groups = set(
        assignments.loc[
            assignments["split"] == "test",
            "request_group",
        ]
    )

    overlap = train_groups.intersection(test_groups)

    if overlap:
        raise RuntimeError(
            f"Wykryto {len(overlap)} wspólnych grup."
        )

    config.SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    assignments.to_csv(
        config.GROUP_SPLIT_FILE,
        index=False,
    )

    print("\nAUDYT DANYCH")
    print(f"Liczba wszystkich rekordów: {len(grouped_df)}")
    print(
        "Liczba unikalnych żądań: "
        f"{grouped_df['request_group'].nunique()}"
    )
    print(
        "Liczba powtórzeń: "
        f"{grouped_df['request_group'].duplicated().sum()}"
    )

    print("\nPODZIAŁ DANYCH")

    for split_name in ["train", "test"]:
        split_ids = assignments.loc[
            assignments["split"] == split_name,
            "row_id",
        ]

        split_labels = grouped_df.loc[
            split_ids,
            "classification",
        ]

        print(f"\n{split_name.upper()}")
        print(f"Liczba rekordów: {len(split_ids)}")
        print("Rozkład klas:")
        print(split_labels.value_counts())
        print("Rozkład procentowy:")
        print(split_labels.value_counts(normalize=True).round(4))

    print(
        "\nLiczba wspólnych grup między treningiem "
        f"i testem: {len(overlap)}"
    )

    print(
        f"Podział zapisano w: {config.GROUP_SPLIT_FILE}"
    )

    return assignments


def load_group_split(
    df: pd.DataFrame,
) -> tuple[pd.Index, pd.Index]:
    if not config.GROUP_SPLIT_FILE.exists():
        raise FileNotFoundError(
            "Nie znaleziono zapisanego podziału. "
            "Najpierw uruchom przygotowanie_podzialu.py."
        )

    assignments = pd.read_csv(config.GROUP_SPLIT_FILE)

    if len(assignments) != len(df):
        raise ValueError(
            "Liczba rekordów w zapisanym podziale "
            "nie odpowiada aktualnym danym."
        )

    if set(assignments["row_id"]) != set(df.index):
        raise ValueError(
            "Indeksy w zapisanym podziale "
            "nie odpowiadają aktualnym danym."
        )

    current_groups = add_request_groups(df)[
        ["request_group"]
    ]

    saved_groups = assignments.set_index("row_id").loc[
        df.index,
        "request_group",
    ]

    if not (
        current_groups["request_group"].to_numpy()
        == saved_groups.to_numpy()
    ).all():
        raise ValueError(
            "Dane zmieniły się od czasu utworzenia podziału."
        )

    train_indices = pd.Index(
        assignments.loc[
            assignments["split"] == "train",
            "row_id",
        ]
    )

    test_indices = pd.Index(
        assignments.loc[
            assignments["split"] == "test",
            "row_id",
        ]
    )

    return train_indices, test_indices


def create_random_split(
    df: pd.DataFrame,
) -> tuple[pd.Index, pd.Index]:
    train_indices, test_indices = train_test_split(
        df.index,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=df["classification"],
    )

    return pd.Index(train_indices), pd.Index(test_indices)

def get_split_indices(
    df: pd.DataFrame,
    split_type: str,
) -> tuple[pd.Index, pd.Index]:
    if split_type == "group":
        return load_group_split(df)

    if split_type == "random":
        return create_random_split(df)

    raise ValueError(
        "Nieznany rodzaj podziału: "
        f"{split_type}. Dostępne: random, group."
    )


def calculate_group_overlap(
    df: pd.DataFrame,
    train_indices: pd.Index,
    test_indices: pd.Index,
) -> int:
    grouped_df = add_request_groups(df)

    train_groups = set(
        grouped_df.loc[
            train_indices,
            "request_group",
        ]
    )

    test_groups = set(
        grouped_df.loc[
            test_indices,
            "request_group",
        ]
    )

    return len(
        train_groups.intersection(test_groups)
    )