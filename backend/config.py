from contextvars import ContextVar
from pydantic_settings import BaseSettings
from pathlib import Path

# Dataset the current request is operating on. Set per request by the middleware in main.py.
current_dataset: ContextVar[str] = ContextVar("current_dataset", default="synthetic")


class Settings(BaseSettings):
    app_name: str = "Consumer Intelligence Platform"
    app_version: str = "1.0.0"
    debug: bool = False

    data_dir: str = "../data"
    models_dir: str = "../models"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "phi4-mini:latest"
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
    def models_root(self) -> Path:
        return Path(self.models_dir).resolve()

    @property
    def workspace_path(self) -> Path:
        """Per-dataset folder for the prepared table and segmentation output."""
        ds = current_dataset.get()
        if ds == "synthetic":
            return self.data_path
        p = self.data_path / "workspaces" / ds
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def models_path(self) -> Path:
        """Per-dataset model folder. The synthetic dataset keeps the original root location."""
        ds = current_dataset.get()
        if ds == "synthetic":
            return self.models_root
        p = self.models_root / ds
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def table_path(self) -> Path:
        return self.workspace_path / "consumers.csv"

    @property
    def segmented_path(self) -> Path:
        return self.workspace_path / "consumers_segmented.csv"


settings = Settings()
