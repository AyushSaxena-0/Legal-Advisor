import os
import json
import logging
from pathlib import Path

# Base directory setup
BASE_DIR = Path(__file__).resolve().parent
DATABASE_DIR = BASE_DIR / "database"
DOCUMENTS_DIR = BASE_DIR / "documents"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
FAISS_INDEX_DIR = BASE_DIR / "faiss_index"
BM25_INDEX_DIR = BASE_DIR / "bm25_index"
MODELS_DIR = BASE_DIR / "models"
RAG_DIR = BASE_DIR / "rag"
UI_DIR = BASE_DIR / "ui"
UTILS_DIR = BASE_DIR / "utils"
PROMPTS_DIR = BASE_DIR / "prompts"
LOGS_DIR = BASE_DIR / "logs"
CACHE_DIR = BASE_DIR / "cache"

# Ensure all directories exist
for directory in [
    DATABASE_DIR, DOCUMENTS_DIR, EMBEDDINGS_DIR, FAISS_INDEX_DIR, 
    BM25_INDEX_DIR, MODELS_DIR, RAG_DIR, UI_DIR, UTILS_DIR, 
    PROMPTS_DIR, LOGS_DIR, CACHE_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)

# Configuration File Path
CONFIG_JSON_PATH = DATABASE_DIR / "settings.json"

# Logging configuration
LOG_FILE = LOGS_DIR / "legal_ai.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LegalAIConfig")

DEFAULT_SETTINGS = {
    "embedding_model": "BAAI/bge-base-en-v1.5",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "top_k_faiss": 30,
    "top_k_bm25": 30,
    "hybrid_weight_faiss": 0.70,
    "temperature": 0.2,
    "context_length": 8192,
    "gpu_enable": True,
    "ollama_model": "qwen3:8b",
    "ollama_base_url": "http://localhost:11434"
}

def load_settings():
    if CONFIG_JSON_PATH.exists():
        try:
            with open(CONFIG_JSON_PATH, "r", encoding="utf-8") as f:
                settings = json.load(f)
            # Ensure all keys exist, fallback to default if missing
            updated = False
            for k, v in DEFAULT_SETTINGS.items():
                if k not in settings:
                    settings[k] = v
                    updated = True
            if updated:
                save_settings(settings)
            return settings
        except Exception as e:
            logger.error(f"Error loading settings from {CONFIG_JSON_PATH}: {e}. Using defaults.")
            return DEFAULT_SETTINGS.copy()
    else:
        save_settings(DEFAULT_SETTINGS)
        return DEFAULT_SETTINGS.copy()

def save_settings(settings):
    try:
        with open(CONFIG_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=4)
        logger.info(f"Settings saved to {CONFIG_JSON_PATH}")
        return True
    except Exception as e:
        logger.error(f"Error saving settings to {CONFIG_JSON_PATH}: {e}")
        return False

# Initialize settings
settings = load_settings()
