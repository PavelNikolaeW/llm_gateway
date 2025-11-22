import os
from pathlib import Path
from pydantic import BaseModel


class BaseSettings(BaseModel):
    """Минимальная заглушка для BaseSettings без внешних зависимостей."""

    class Config:
        env_file = None
        extra = "allow"

    def __init__(self, **values):
        env_values: dict[str, str] = {}
        env_file = getattr(self.Config, "env_file", None)
        if env_file:
            path = Path(env_file)
            if path.exists():
                for line in path.read_text().splitlines():
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    env_values[key.strip()] = val.strip()
        for field in getattr(self, "model_fields", {}):
            if field in os.environ and field not in values:
                env_values[field] = os.environ[field]
        merged = {**env_values, **values}
        super().__init__(**merged)
