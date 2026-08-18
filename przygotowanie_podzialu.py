import config
from protokol_eksperymentalny import create_and_save_experimental_protocol
from przetwarzanie_danych import get_processed_data


def prepare_splits() -> None:
    print("Przygotowywanie centralnego protokołu eksperymentalnego...")

    df = get_processed_data()
    manifest = create_and_save_experimental_protocol(df)

    development_size = (manifest["partition"] == "development").sum()
    final_test_size = (manifest["partition"] == "final_test").sum()

    print("\nProtokół przygotowany poprawnie.")
    print(f"Development: {development_size} rekordów")
    print(f"Final test: {final_test_size} rekordów")
    print(f"Manifest: {config.EXPERIMENTAL_PROTOCOL_FILE}")
    print(f"Metadane: {config.EXPERIMENTAL_PROTOCOL_METADATA_FILE}")


if __name__ == "__main__":
    prepare_splits()