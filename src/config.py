
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import yaml
import pandas as pd
from pydantic import BaseModel

class PathsConfig(BaseModel):
    data_dir: Path
    results_dir: Path = Path("./results")
    prefer_committed_data: bool = True
    fail_duplicate: bool = True

class Config(BaseModel):
    paths: PathsConfig

    @classmethod
    def load(cls, config_path: Path) -> "Config":
        """Loads and validates the configuration from a YAML file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at: {config_path}")

        with open(config_path, "r") as f:
            yaml_data = yaml.safe_load(f) or {}

        # Pydantic automatically unpacks the nested dictionaries, 
        # validates the fields, and casts strings to pathlib.Path objects.
        return cls(**yaml_data)

class DataPath:
    def __init__(self, config):
        self.committed_root = Path(__file__).resolve().parent.parent / "data"
        self.data_dir = config.paths.data_dir
        self.prefer_committed = config.paths.prefer_committed_data
        self.fail_duplicate = config.paths.fail_duplicate
        
        # Ensure the base uncommitted directory exists for potential future writes
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def get_path(self, relative_path: str) -> Tuple[Path, bool]:
        """
        Resolves the path based on existence and configuration preferences.
        Returns a tuple: (resolved_path, file_exists_boolean)
        """
        committed_path = self.committed_root / relative_path
        uncommitted_path = self.data_dir / relative_path

        # 1. Conflict Check
        if committed_path.exists() and uncommitted_path.exists():
            msg = (
                f"\nData Conflict for '{relative_path}'!\n"
                f"Copies exist in both locations:\n"
                f"  - Committed: {committed_path}\n"
                f"  - Local:     {uncommitted_path}"
            )
            if self.fail_duplicate:
                raise FileExistsError(msg)
            else:
                logging.warning(msg)

        # 2. Determine search order
        search_paths = (
            [committed_path, uncommitted_path] 
            if self.prefer_committed 
            else [uncommitted_path, committed_path]
        )

        # 3. Check for existence
        for path in search_paths:
            if path.exists():
                return path, True

        # 4. If neither exists, route the write flow to the uncommitted local path
        return uncommitted_path, False
