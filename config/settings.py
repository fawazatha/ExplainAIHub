from dotenv import load_dotenv
from dataclasses import dataclass
import os

load_dotenv(override=True)

@dataclass
class Settings:
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

settings = Settings()
