# =============================================================================
# config.py -- Central configuration for the Anki Flashcard Generator
# =============================================================================

import os
from dotenv import load_dotenv

# Load environment variables from .env (if present)
load_dotenv()

# -----------------------------------------------------------------------------
# API and Models
# -----------------------------------------------------------------------------

# API key is loaded from the .env file. Never hard-code it here.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# LLM model used to generate flashcard content
CHAT_MODEL = "deepseek/deepseek-v4-flash"

# Embedding model used for semantic chunking
EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"

# Optional headers sent to OpenRouter to identify the app
APP_NAME = "Anki Flashcard Generator"
APP_URL = "https://github.com/mauroforlin/anki-llm-flashcard-generator"

# -----------------------------------------------------------------------------
# Rate Limiting
#
# IMPORTANT: these values are tuned for a paid OpenRouter plan (200 req/min).
# If you are using free-tier models, you must lower these values significantly.
# Free models typically allow 20 req/min or fewer.
# Check your current limits at: https://openrouter.ai/settings/limits
# -----------------------------------------------------------------------------

# Maximum requests per minute sent to OpenRouter (kept slightly below the hard cap)
MAX_REQUESTS_PER_MINUTE = 185

# Maximum number of parallel workers for LLM requests
MAX_CONCURRENT_LLM_REQUESTS = 20

# Maximum number of parallel workers for embedding requests
MAX_CONCURRENT_EMBEDDING_REQUESTS = 10

# Seconds to wait after receiving a 429 rate-limit error before retrying
RATE_LIMIT_BACKOFF_SECONDS = 15

# Maximum number of retry attempts per chunk on failure
MAX_RETRIES = 3

# -----------------------------------------------------------------------------
# Semantic Chunking
# -----------------------------------------------------------------------------

# Cosine similarity percentile threshold used to detect topic boundaries.
# Higher values produce smaller, more tightly focused chunks (range: 60-95).
SIMILARITY_THRESHOLD_PERCENTILE = 85

# Minimum number of sentences per chunk (smaller chunks are merged with the previous one)
MIN_CHUNK_SENTENCES = 4

# Maximum number of sentences per chunk (larger chunks are split further)
MAX_CHUNK_SENTENCES = 30

# Maximum approximate chunk size in characters
MAX_CHUNK_CHARS = 3000

# Batch size for embedding requests (OpenRouter accepts up to 96 inputs per call)
EMBEDDING_BATCH_SIZE = 50

# -----------------------------------------------------------------------------
# Flashcard Generation
# -----------------------------------------------------------------------------

# Generation style. Options: "atomic" (strict SRS, short answers) or "comprehensive" (broad paragraphs)
FLASHCARD_STYLE = "comprehensive"

# Target character density per flashcard. Used to calculate the dynamic quota per chunk.
CHARS_PER_FLASHCARD_ATOMIC = 600
CHARS_PER_FLASHCARD_COMPREHENSIVE = 1200

# -----------------------------------------------------------------------------
# Semantic Deduplication
# -----------------------------------------------------------------------------

# Whether to run a post-generation deduplication step using embeddings.
ENABLE_DEDUPLICATION = False

# Cosine similarity threshold for deduplication (0.0 to 1.0).
# A higher value (e.g. 0.95) makes deduplication stricter (only near-exact semantic matches are dropped).
# A lower value (e.g. 0.85) will drop cards that sound vaguely similar but might cause false positives.
DEDUPLICATION_THRESHOLD = 0.95

# Model temperature: lower values produce more structured, deterministic output
LLM_TEMPERATURE = 0.3

# Maximum tokens allowed in the LLM response
LLM_MAX_TOKENS = 2048

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
SOURCE_DIRECTORY = os.path.join(ROOT_DIRECTORY, "source")
OUTPUT_DIRECTORY = os.path.join(ROOT_DIRECTORY, "output")

# On-disk embedding cache -- avoids recomputing embeddings on repeated runs
EMBEDDING_CACHE_PATH = os.path.join(ROOT_DIRECTORY, ".embedding_cache.pkl")

# Name of the top-level Anki deck
ANKI_DECK_NAME = "Lecture Notes"
