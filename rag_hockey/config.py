import os

#from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DOC_PATH = os.getenv("DOC_PATH", "data/documents/2025-26_iihf_rulebook.pdf")
VS_PATH = os.getenv("VS_PATH", "data/vectorstore")
PARENT_PATH = os.getenv("PARENT_PATH", "data/parentstore")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")
EMBED_DIM = int(os.getenv("EMBED_DIM", 3072))
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openai")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
RETRIEVER_K = int(os.getenv("RETRIEVER_K", 8))
RETRIEVER_FETCH_K = int(os.getenv("RETRIEVER_FETCH_K", 20))
RETRIEVER_LAMBDA = float(os.getenv("RETRIEVER_LAMBDA", 0.5))
CHILD_CHUNK_SIZE = int(os.getenv("CHILD_CHUNK_SIZE", 800))
CHILD_CHUNK_OVERLAP = int(os.getenv("CHILD_CHUNK_OVERLAP", 120))
SOURCE_LABEL = "IIHF Rulebook 2025-26"
CROP_START, CROP_END = 15, 160  # drop TOC/appendix in the iihf 2025 rulebook
TIMEOUT_S = int(os.getenv("LLM_TIMEOUT", 30))
TEMPERATURE = int(os.getenv("LLM_TEMPERATURE", 0))
MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", 3))
USE_MULTI_QUERY = bool(os.getenv("USE_MULTI_QUERY", True))
