from functools import lru_cache

from pydantic import Extra
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # vars
    is_local: bool
    is_prod: bool
    debug: bool = True
    open_ai_key: str
    open_ai_org: str
    elevenlabs_key: str

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        extra = Extra.allow


@lru_cache()
def get_settings():
    """
    Returns the settings object.

    :return: The settings object.
    """
    return Settings()


settings: Settings = get_settings()
