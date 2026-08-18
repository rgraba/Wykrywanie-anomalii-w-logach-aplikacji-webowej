import unittest
from pathlib import Path

import config
from podzial_danych import ensure_no_group_overlap
from protokol_eksperymentalny import (
    get_development_and_final_test_indices,
    get_development_folds,
)
from przetwarzanie_danych import get_processed_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IGNORED_DIRECTORIES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "tests",
}


def read_project_python_files() -> list[tuple[str, str]]:
    source_files = []

    for path in PROJECT_ROOT.rglob("*.py"):
        relative_path = path.relative_to(PROJECT_ROOT)

        if any(
            part in IGNORED_DIRECTORIES
            for part in relative_path.parts
        ):
            continue

        source = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        source_files.append(
            (
                relative_path.as_posix(),
                source,
            )
        )

    return source_files


class ExperimentalProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = get_processed_data()

        (
            cls.development_indices,
            cls.final_test_indices,
        ) = get_development_and_final_test_indices(cls.df)

        cls.folds = get_development_folds(cls.df)

    def test_development_and_final_test_have_no_group_overlap(self):
        overlap = ensure_no_group_overlap(
            self.df,
            self.development_indices,
            self.final_test_indices,
            first_name="development",
            second_name="final_test",
        )

        self.assertEqual(overlap, 0)

    def test_expected_fold_numbers_are_present(self):
        fold_numbers = [
            fold_number
            for fold_number, _, _ in self.folds
        ]

        expected_fold_numbers = list(
            range(
                1,
                config.DEVELOPMENT_CV_N_SPLITS + 1,
            )
        )

        self.assertEqual(
            fold_numbers,
            expected_fold_numbers,
        )

    def test_each_fold_has_disjoint_rows_and_groups(self):
        development_rows = set(self.development_indices)

        for (
            fold_number,
            train_indices,
            validation_indices,
        ) in self.folds:
            train_rows = set(train_indices)
            validation_rows = set(validation_indices)

            self.assertTrue(
                train_rows.isdisjoint(validation_rows),
                msg=(
                    f"Fold {fold_number} zawiera wspólne rekordy "
                    "w treningu i walidacji."
                ),
            )

            self.assertEqual(
                train_rows.union(validation_rows),
                development_rows,
                msg=(
                    f"Fold {fold_number} nie obejmuje dokładnie "
                    "zbioru development."
                ),
            )

            overlap = ensure_no_group_overlap(
                self.df,
                train_indices,
                validation_indices,
                first_name=f"fold_{fold_number}_train",
                second_name=f"fold_{fold_number}_validation",
            )

            self.assertEqual(
                overlap,
                0,
                msg=(
                    f"Fold {fold_number} zawiera wspólne "
                    "grupy żądań."
                ),
            )

    def test_validation_folds_partition_development(self):
        validation_rows = [
            row_index
            for _, _, validation_indices in self.folds
            for row_index in validation_indices
        ]

        self.assertEqual(
            len(validation_rows),
            len(self.development_indices),
        )

        self.assertEqual(
            set(validation_rows),
            set(self.development_indices),
        )


class SafeSplitUsageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_files = read_project_python_files()

    def test_deprecated_split_mechanisms_are_absent(self):
        forbidden_tokens = [
            "train_" + "test_split",
            "create_random_" + "split",
            "load_group_" + "split",
            "get_split_" + "indices",
            "LogisticRegression" + "CV",
        ]

        violations = {}

        for file_name, source in self.source_files:
            detected_tokens = [
                token
                for token in forbidden_tokens
                if token in source
            ]

            if detected_tokens:
                violations[file_name] = detected_tokens

        self.assertFalse(
            violations,
            msg=(
                "Wykryto wycofane mechanizmy podziału "
                f"lub walidacji: {violations}"
            ),
        )

    def test_plain_stratified_kfold_is_only_used_for_rq1(self):
        splitter_name = "Stratified" + "KFold"

        files_using_splitter = sorted(
            file_name
            for file_name, source in self.source_files
            if splitter_name in source
        )

        self.assertEqual(
            files_using_splitter,
            ["porownanie_podzialow_random_forest.py"],
            msg=(
                "Zwykły StratifiedKFold może występować wyłącznie "
                "w kontrolowanym porównaniu random vs group."
            ),
        )

    def test_group_fold_consumers_check_overlap(self):
        fold_loader_name = "get_development_" + "folds"
        overlap_guard_name = "ensure_no_group_" + "overlap"

        violations = [
            file_name
            for file_name, source in self.source_files
            if (
                fold_loader_name in source
                and overlap_guard_name not in source
            )
        ]

        self.assertEqual(
            violations,
            [],
            msg=(
                "Skrypty korzystające z grupowych foldów "
                f"nie kontrolują group_overlap: {violations}"
            ),
        )


if __name__ == "__main__":
    unittest.main()