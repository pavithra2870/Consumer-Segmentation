from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    app_name: str = "Consumer Intelligence Platform"
    app_version: str = "1.0.0"
    debug: bool = False

    data_dir: str = "../data"
    models_dir: str = "../models"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    ollama_timeout: int = 120

    dataset_size: int = 100000
    n_clusters: int = 6
    random_state: int = 42

    class Config:
        env_file = ".env"

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir).resolve()

    @property
    def models_path(self) -> Path:
        return Path(self.models_dir).resolve()


settings = Settings()
