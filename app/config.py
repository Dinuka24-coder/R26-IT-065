from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MONGO_URI: str
    MONGO_DB_NAME: str
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    API_PREFIX: str = "/api/v1"
    APP_ENV: str = "development"

    class Config:
        env_file = ".env"

settings = Settings()