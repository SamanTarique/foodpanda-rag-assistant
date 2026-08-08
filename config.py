from pathlib import Path
from dotenv import load_dotenv
import os



load_dotenv()


# ====================Project Root====================================

BASE_DIR = Path(__file__).resolve().parent


# ====================Project Folders====================================


KNOWLEDGE_DIR = BASE_DIR / "knowledge"
FAISS_INDEX_DIR = BASE_DIR / "faiss_index"
LOGS_DIR = BASE_DIR / "logs"


# ====================Files====================================


HASH_FILE = FAISS_INDEX_DIR / "db_hash.txt"


# ====================Supported File Types====================================


SUPPORTED_EXTENSIONS = [".md", ".csv"]


# ====================Chunking Settings====================================

MAX_CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

# ====================Retrieval Settings====================================

TOP_K = 4

#=================================== Embedding Model===================



EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ====================Gemini Model====================================

GEMINI_MODEL = "gemini-3.6-flash"

# ====================API Key====================================


GEMINI_API_KEY = os.getenv("protech4")

# ====================Create Required Folders====================================

for folder in [KNOWLEDGE_DIR, FAISS_INDEX_DIR, LOGS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

