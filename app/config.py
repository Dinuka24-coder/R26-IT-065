from typing import Optional
 
from pydantic_settings import BaseSettings
 
class Settings(BaseSettings):
    MONGO_URI: str
    MONGO_DB_NAME: str
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    API_PREFIX: str = "/api/v1"
    APP_ENV: str = "development"
 
    # Optional -- the component3 gatekeeper cascade degrades gracefully (falls
    # back to the local CNN+heuristic) when this isn't set, so a missing key
    # must never crash app startup.
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
 
    class Config:
        env_file = ".env"
 
settings = Settings()