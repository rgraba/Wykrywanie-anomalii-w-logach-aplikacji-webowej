import re
from urllib.parse import urlsplit

import pandas as pd

import config


HTTP_VERSION_SUFFIX = re.compile(
    r"\s+HTTP/\d(?:\.\d)?\s*$",
    flags=re.IGNORECASE,
)


def remove_http_version(raw_url: str) -> str:
    raw_url = str(raw_url).strip()

    return HTTP_VERSION_SUFFIX.sub(
        "",
        raw_url,
    )


def extract_path_and_query(
    raw_url: str,
) -> tuple[str, str]:
    cleaned_url = remove_http_version(raw_url)

    try:
        parsed_url = urlsplit(cleaned_url)

        path = parsed_url.path or ""
        query = parsed_url.query or ""

        return path, query

    except ValueError:
        if "?" in cleaned_url:
            path, query = cleaned_url.split(
                "?",
                maxsplit=1,
            )

            return path, query

        return cleaned_url, ""


def combine_arguments(
    query: str,
    content: str,
) -> str:
    parts = []

    if str(query).strip():
        parts.append(str(query).strip())

    if str(content).strip():
        parts.append(str(content).strip())

    return "&".join(parts)


def count_arguments(arguments: str) -> int:
    arguments = str(arguments).strip()

    if not arguments:
        return 0

    return sum(
        1
        for argument in arguments.split("&")
        if argument.strip()
    )


def count_letters(text: str) -> int:
    return sum(
        character.isalpha()
        for character in str(text)
    )


def count_digits(text: str) -> int:
    return sum(
        character.isdigit()
        for character in str(text)
    )


def count_special_chars(text: str) -> int:
    return sum(
        character in config.SPECIAL_CHARS
        for character in str(text)
    )


def calculate_max_byte(text: str) -> int:
    encoded_request = str(text).encode(
        "utf-8",
        errors="replace",
    )

    return max(
        encoded_request,
        default=0,
    )


def add_althubiti_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    if "request_text" not in df.columns:
        raise ValueError(
            "Brakuje kolumny request_text. "
            "Najpierw uruchom add_request_text()."
        )

    result = df.copy()

    url_parts = result["URL"].apply(
        extract_path_and_query
    )

    paths = url_parts.apply(
        lambda parts: parts[0]
    )

    queries = url_parts.apply(
        lambda parts: parts[1]
    )

    contents = (
        result["content"]
        .fillna("")
        .astype(str)
    )

    arguments = pd.Series(
        [
            combine_arguments(query, content)
            for query, content in zip(
                queries,
                contents,
            )
        ],
        index=result.index,
    )

    # 1. Length of the request
    result["request_length"] = (
        result["request_text"]
        .astype(str)
        .str.len()
    )

    # 2. Length of the arguments
    result["arguments_length"] = (
        arguments.str.len()
    )

    # 3. Number of arguments
    result["arguments_count"] = (
        arguments.apply(count_arguments)
    )

    # 4. Number of digits in the arguments
    result["arguments_digit_count"] = (
        arguments.apply(count_digits)
    )

    # 5. Length of the path
    result["path_length"] = (
        paths.astype(str).str.len()
    )

    # 6. Number of letters in the arguments
    result["arguments_letter_count"] = (
        arguments.apply(count_letters)
    )

    # 7. Number of letters in the path
    result["path_letter_count"] = (
        paths.apply(count_letters)
    )

    # 8. Number of special characters in the path
    result["path_special_char_count"] = (
        paths.apply(count_special_chars)
    )

    # 9. Maximum byte value in the request
    result["max_request_byte"] = (
        result["request_text"].apply(
            calculate_max_byte
        )
    )

    return result