from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "postgresql+psycopg://fl_web:fl_web@localhost:5432/fl_web"

    jwt_secret: str = "dev-insecure-secret-change-me"
    jwt_algorithm: str = "HS256"
    jwt_expires_minutes: int = 1440

    # Public-facing addresses, used to render the docker run command shown to
    # participants after they create a client token. After deploy, these become
    # the real server's address (e.g. http://83.149.250.70:8000 + :9092).
    public_server_url: str = "http://172.25.71.235:8000"
    public_superlink_addr: str = "172.25.71.235:9092"
    fl_client_image: str = "ghcr.io/nxhxl000/fl-client:latest"


@lru_cache
def get_settings() -> Settings:
    return Settings()
