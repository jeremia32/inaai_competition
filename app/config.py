import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
GEMINI_API_KEY = os.getenv("GeminiAPIKEY")

CHROMA_DB_DIR = "chroma_db"

# Prefer a lightweight multilingual embedding tuned for Indonesian and similar languages.
# Use a 384-dimensional embedding model instead of 768 by default.
# Can be overridden with the EMBEDDING_MODEL env var.
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "intfloat/multilingual-e5-small",
)   