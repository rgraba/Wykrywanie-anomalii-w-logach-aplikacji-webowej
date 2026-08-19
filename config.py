from pathlib import Path
import re

# Ścieżki projektu

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
SPLITS_DIR = BASE_DIR / "splits"

EXPERIMENTAL_PROTOCOL_FILE = (
    SPLITS_DIR
    / "experimental_protocol_seed_2026.csv"
)

EXPERIMENTAL_PROTOCOL_METADATA_FILE = (
    SPLITS_DIR
    / "experimental_protocol_seed_2026.json"
)
DATA_FILE = DATA_DIR / "csic_database.csv"

# Kolumny zbioru danych

BASE_COLUMNS = [
    "Method",
    "URL",
    "content",
    "classification"
]

# Zestawy cech

ML_FEATURES_ADVANCED = [
    "url_len",
    "content_len",
    "url_special_chars",
    "content_special_chars",
    "sqli_flag",
    "xss_flag",
    "traversal_flag"
]

ML_FEATURES_BASIC = [
    "url_len",
    "content_len",
    "url_special_chars",
    "content_special_chars"
]

ML_FEATURES_ALTHUBITI_9 = [
    "request_length",
    "arguments_length",
    "arguments_count",
    "arguments_digit_count",
    "path_length",
    "arguments_letter_count",
    "path_letter_count",
    "path_special_char_count",
    "max_request_byte",
]


ML_FEATURES_ALTHUBITI_5 = [
    "request_length",
    "arguments_length",
    "arguments_count",
    "path_length",
    "path_special_char_count",
]

ML_FEATURES = ML_FEATURES_BASIC

# Wzorce bezpieczeństwa

SPECIAL_CHARS = "!@#$%^&*()_+{}|:\"<>?-=[]\\;',./"

SQLI_PATTERN = re.compile(
    r"(?i)(?:union\s+select|select\s+.*?\s+from|insert\s+into|drop\s+table|update\s+.*?\s+set|--|/\*|\*/|;)"
)

XSS_PATTERN = re.compile(
    r"(?i)(?:<script>|javascript:|onerror=|onload=|eval\(|document\.cookie|<iframe)"
)

TRAVERSAL_PATTERN = re.compile(
    r"(?i)(?:\.\./|\.\.\\|etc/passwd|cmd\.exe|\.exe)"
)

# Generalne parametry

RANDOM_STATE = 42
PROTOCOL_RANDOM_STATE = 2026
FINAL_TEST_N_SPLITS = 5
DEVELOPMENT_CV_N_SPLITS = 10

# Hiperparametry

# Isolation Forest
ISOLATION_FOREST_PARAMS = {
    "n_estimators": 100,
    "max_samples": "auto",
    "contamination": 0.4,
}

# Random Forest
RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "class_weight": "balanced"
}

# One-Class SVM
OCSVM_PARAMS = {
    "kernel": "rbf",
    "gamma": "scale",
    "nu": 0.3
}

# Finalne konfiguracje wybrane na zbiorze development

FINAL_RANDOM_FOREST_FEATURE_SET = "INFORMATION_GAIN"

FINAL_RANDOM_FOREST_FEATURES = [
    "request_length",
    "arguments_length",
    "arguments_letter_count",
]


FINAL_IFOREST_CALIBRATED_FEATURE_SET = "ALTHUBITI_9"

FINAL_IFOREST_CALIBRATED_FEATURES = (
    ML_FEATURES_ALTHUBITI_9.copy()
)


FINAL_OCSVM_CALIBRATED_FEATURE_SET = "BASIC"

FINAL_OCSVM_CALIBRATED_FEATURES = (
    ML_FEATURES_BASIC.copy()
)


FINAL_IFOREST_STRICT_FEATURE_SET = "BASIC"

FINAL_IFOREST_STRICT_FEATURES = (
    ML_FEATURES_BASIC.copy()
)


FINAL_OCSVM_STRICT_FEATURE_SET = "BASIC"

FINAL_OCSVM_STRICT_FEATURES = (
    ML_FEATURES_BASIC.copy()
)


STRICT_ONE_CLASS_TARGET_FPR = 0.05