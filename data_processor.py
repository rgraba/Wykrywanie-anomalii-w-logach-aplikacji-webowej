import pandas as pd
import config


def load_data() -> pd.DataFrame:

    print(f"Wczytywanie pliku: {config.DATA_FILE}...")

    try:
        df = pd.read_csv(config.DATA_FILE)

        missing_columns = [
            col for col in config.BASE_COLUMNS
            if col not in df.columns
        ]

        if missing_columns:
            raise ValueError(f"Brakuje wymaganych kolumn: {missing_columns}")

        df = df[config.BASE_COLUMNS].copy()

        print(f"Wczytano pomyślnie. Kształt: {df.shape}")
        return df

    except FileNotFoundError:
        print(f"Błąd: brak pliku {config.DATA_FILE}.")
        raise

    except Exception as e:
        print(f"Błąd wczytywania danych: {e}")
        raise


def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    print("Czyszczenie danych...")

    df = df.copy()

    text_columns = ["URL", "content"]

    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    return df


def count_special_chars(text: str) -> int:

    if not isinstance(text, str):
        return 0

    return sum(char in config.SPECIAL_CHARS for char in text)


def add_length_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["url_len"] = df["URL"].str.len()
    df["content_len"] = df["content"].str.len()

    return df


def add_special_char_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["url_special_chars"] = df["URL"].apply(count_special_chars)
    df["content_special_chars"] = df["content"].apply(count_special_chars)

    return df


def add_security_flags(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    combined_text = df["URL"].astype(str) + " " + df["content"].astype(str)

    df["sqli_flag"] = combined_text.str.contains(
        config.SQLI_PATTERN,
        na=False
    ).astype(int)

    df["xss_flag"] = combined_text.str.contains(
        config.XSS_PATTERN,
        na=False
    ).astype(int)

    df["traversal_flag"] = combined_text.str.contains(
        config.TRAVERSAL_PATTERN,
        na=False
    ).astype(int)

    return df


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    anomaly_labels = {"anomaly", "anomalous", "attack", "malicious", "1"}

    df["classification"] = df["classification"].apply(
        lambda x: 1 if str(x).lower().strip() in anomaly_labels else 0
    )

    return df


def extract_features(df: pd.DataFrame) -> pd.DataFrame:

    print("Ekstrakcja cech...")

    df = add_length_features(df)
    df = add_special_char_features(df)
    df = add_security_flags(df)
    df = encode_labels(df)

    return df


def get_processed_data() -> pd.DataFrame:

    print("\nStart przetwarzania danych")

    df = load_data()
    df = clean_data(df)
    df = extract_features(df)

    print("Zakończono przetwarzanie danych\n")

    return df


if __name__ == "__main__":
    try:
        processed_df = get_processed_data()
        print(processed_df.head())
        print("\nRozkład klas:")
        print(processed_df["classification"].value_counts())

    except Exception:
        print("Przetwarzanie przerwane.")

        import traceback
        traceback.print_exc()