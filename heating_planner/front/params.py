from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

FPATH = Path(__file__)
ENV_PATH = FPATH.parent.parent.parent / "local.env"

class AppSettings(BaseSettings):
    DRIAS_REF_PATH: str
    DRIAS_PROJ_PATH: str
    CLAY_PATH: str
    SEA_ELEVATION_PATH: str
    
    model_config = SettingsConfigDict(env_file=ENV_PATH, env_file_encoding='utf-8')


settings = AppSettings()
print(settings.CLAY_PATH)