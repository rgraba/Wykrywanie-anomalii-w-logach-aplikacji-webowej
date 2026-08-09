from przetwarzanie_danych import get_processed_data
from podzial_danych import create_and_save_group_split


def prepare_splits() -> None:
    print("Przygotowywanie podziału grupowego...")

    df = get_processed_data()

    create_and_save_group_split(df)

    print("\nPodział danych przygotowany poprawnie.")


if __name__ == "__main__":
    prepare_splits()