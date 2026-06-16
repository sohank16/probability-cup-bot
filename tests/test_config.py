from src.config import Settings


def test_settings_defaults_to_dry_run() -> None:
    settings = Settings()
    assert settings.dry_run is True
