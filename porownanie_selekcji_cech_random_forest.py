from walidacja_selekcji_cech_random_forest_cv10 import validate_with_cv10


def compare_feature_selection() -> None:
    print(
        "Porównanie selekcji cech korzysta teraz z centralnego "
        "protokołu i grupowej walidacji CV10."
    )

    validate_with_cv10()


if __name__ == "__main__":
    compare_feature_selection()