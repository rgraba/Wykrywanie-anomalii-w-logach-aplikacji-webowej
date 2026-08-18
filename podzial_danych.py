import hashlib
import json

import pandas as pd
import config


REQUEST_COMPONENT_COLUMNS = (
    "Method",
    "URL",
    "content",
)


def normalize_request_component(value) -> str:
    if pd.isna(value):
        return ""

    return str(value)


def create_canonical_request(
    method,
    url,
    content,
) -> str:
    components = [
        normalize_request_component(method),
        normalize_request_component(url),
        normalize_request_component(content),
    ]

    return json.dumps(
        components,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def calculate_request_hash(
    canonical_request: str,
) -> str:
    return hashlib.sha256(
        canonical_request.encode(
            "utf-8",
            errors="replace",
        )
    ).hexdigest()


def create_canonical_requests(
    df: pd.DataFrame,
) -> pd.Series:
    missing_columns = [
        column
        for column in REQUEST_COMPONENT_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Brakuje kolumn potrzebnych do grupowania: "
            f"{missing_columns}."
        )

    canonical_requests = [
        create_canonical_request(
            method,
            url,
            content,
        )
        for method, url, content in zip(
            df["Method"],
            df["URL"],
            df["content"],
        )
    ]

    return pd.Series(
        canonical_requests,
        index=df.index,
        dtype="string",
    )


def add_request_groups(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    canonical_requests = create_canonical_requests(
        result
    )

    result["request_group"] = (
        canonical_requests.apply(
            calculate_request_hash
        )
    )

    return result


def validate_request_groups(
    df: pd.DataFrame,
) -> dict:
    if not df.index.is_unique:
        raise ValueError(
            "Indeksy rekordów nie są unikalne."
        )

    if "classification" not in df.columns:
        raise ValueError(
            "Brakuje kolumny classification."
        )

    grouped_df = add_request_groups(df)

    canonical_requests = create_canonical_requests(
        grouped_df
    )

    group_audit = pd.DataFrame(
        {
            "request_group": (
                grouped_df["request_group"]
            ),
            "canonical_request": (
                canonical_requests
            ),
            "classification": (
                grouped_df["classification"]
            ),
        },
        index=grouped_df.index,
    )

    requests_per_hash = (
        group_audit
        .groupby("request_group")[
            "canonical_request"
        ]
        .nunique(dropna=False)
    )

    collision_groups = requests_per_hash[
        requests_per_hash > 1
    ]

    if not collision_groups.empty:
        raise RuntimeError(
            "Wykryto kolizję identyfikatora grupy "
            f"dla {len(collision_groups)} grup."
        )

    labels_per_group = (
        group_audit
        .groupby("request_group")[
            "classification"
        ]
        .nunique(dropna=False)
    )

    mixed_label_groups = labels_per_group[
        labels_per_group > 1
    ]

    if not mixed_label_groups.empty:
        raise RuntimeError(
            "Wykryto identyczne żądania z różnymi "
            f"etykietami w {len(mixed_label_groups)} grupach."
        )

    group_sizes = (
        grouped_df["request_group"]
        .value_counts()
    )

    audit_results = {
        "total_records": len(grouped_df),
        "unique_groups": int(
            grouped_df["request_group"].nunique()
        ),
        "duplicate_records": int(
            len(grouped_df)
            - grouped_df["request_group"].nunique()
        ),
        "records_in_duplicate_groups": int(
            group_sizes[group_sizes > 1].sum()
        ),
        "duplicate_groups": int(
            (group_sizes > 1).sum()
        ),
        "maximum_group_size": int(
            group_sizes.max()
        ),
        "mixed_label_groups": 0,
        "hash_collisions": 0,
    }

    return audit_results


def validate_partition_indices(
    df: pd.DataFrame,
    indices,
    partition_name: str,
) -> pd.Index:
    validated_indices = pd.Index(indices)

    if validated_indices.has_duplicates:
        raise ValueError(
            f"Partycja {partition_name} zawiera "
            "powtórzone indeksy rekordów."
        )

    missing_indices = validated_indices.difference(
        df.index
    )

    if not missing_indices.empty:
        raise ValueError(
            f"Partycja {partition_name} zawiera "
            f"{len(missing_indices)} indeksów, których "
            "nie ma w zbiorze danych."
        )

    return validated_indices


def find_group_overlap(
    df: pd.DataFrame,
    first_indices,
    second_indices,
    first_name: str = "train",
    second_name: str = "evaluation",
) -> set[str]:
    first_indices = validate_partition_indices(
        df,
        first_indices,
        first_name,
    )

    second_indices = validate_partition_indices(
        df,
        second_indices,
        second_name,
    )

    row_overlap = first_indices.intersection(
        second_indices
    )

    if not row_overlap.empty:
        raise RuntimeError(
            f"Partycje {first_name} i {second_name} "
            f"zawierają {len(row_overlap)} wspólnych "
            "indeksów rekordów."
        )

    grouped_df = add_request_groups(df)

    first_groups = set(
        grouped_df.loc[
            first_indices,
            "request_group",
        ]
    )

    second_groups = set(
        grouped_df.loc[
            second_indices,
            "request_group",
        ]
    )

    return first_groups.intersection(
        second_groups
    )


def calculate_group_overlap(
    df: pd.DataFrame,
    train_indices,
    test_indices,
) -> int:
    overlap = find_group_overlap(
        df=df,
        first_indices=train_indices,
        second_indices=test_indices,
        first_name="train",
        second_name="test",
    )

    return len(overlap)


def ensure_no_group_overlap(
    df: pd.DataFrame,
    first_indices,
    second_indices,
    first_name: str = "train",
    second_name: str = "evaluation",
) -> int:
    overlap = find_group_overlap(
        df=df,
        first_indices=first_indices,
        second_indices=second_indices,
        first_name=first_name,
        second_name=second_name,
    )

    if overlap:
        example_groups = sorted(overlap)[:3]

        raise RuntimeError(
            f"Wykryto {len(overlap)} wspólnych grup "
            f"pomiędzy {first_name} i {second_name}. "
            f"Przykładowe grupy: {example_groups}"
        )

    return 0