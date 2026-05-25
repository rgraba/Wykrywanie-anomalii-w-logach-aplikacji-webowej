from pathlib import Path
import re

# Project Paths

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

DATA_FILE = DATA_DIR / "csic_database.csv"

# Dataset columns

BASE_COLUMNS = [
    "Method",
    "URL",
    "content",
    "classification"
]

# ML features

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

ML_FEATURES = ML_FEATURES_BASIC

# Security patterns

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
TEST_SIZE = 0.2

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
