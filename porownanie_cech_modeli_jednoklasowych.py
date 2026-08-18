from walidacja_modeli_jednoklasowych_cv10 import validate_configs_cv10


def compare_one_class_feature_sets() -> None:
    print(
        "Porównanie modeli jednoklasowych korzysta teraz "
        "z centralnego protokołu i grupowej walidacji CV10."
    )

    validate_configs_cv10()


if __name__ == "__main__":
    compare_one_class_feature_sets()