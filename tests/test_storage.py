from src.storage import Storage


def test_storage_initializes_database(tmp_path) -> None:
    database_path = tmp_path / "test.sqlite"
    storage = Storage(database_path)

    storage.initialize()

    assert database_path.exists()
