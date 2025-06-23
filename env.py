import os
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

class EnvSchema(BaseModel):
    GEMINI_BASE_URL: str
    GEMINI_API_KEY: str
    
env = EnvSchema(
    GEMINI_BASE_URL=os.getenv("GEMINI_BASE_URL"),
    GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")
)