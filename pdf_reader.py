# =============================================================================
# pdf_reader.py -- PDF Text Extraction and Semantic Chunking via Embeddings
# =============================================================================

import os
import re
import pickle
import hashlib
import logging
import asyncio
import numpy as np
import httpx
from pathlib import Path
from typing import List, Tuple

import nltk
from nltk.tokenize import sent_tokenize

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NLTK setup -- download sentence tokenizer data if not already present
# ---------------------------------------------------------------------------

def _ensure_nltk_resources():
    """Download required NLTK tokenizer data if not already present on disk."""
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    ]
    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            logger.info(f"Downloading NLTK resource: {resource_name}")
            nltk.download(resource_name, quiet=True)

_ensure_nltk_resources()


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file using pymupdf4llm (Markdown output, LLM-ready).
    Falls back to pdfplumber if pymupdf4llm fails.
    """
    logger.info(f"Extracting text from: {os.path.basename(pdf_path)}")

    # Attempt 1: pymupdf4llm (best quality, structured Markdown output)
    try:
        import pymupdf4llm
        md_text = pymupdf4llm.to_markdown(pdf_path)
        if md_text and len(md_text.strip()) > 100:
            logger.debug(f"pymupdf4llm OK -- {len(md_text)} characters extracted")
            return _clean_markdown_text(md_text)
    except Exception as e:
        logger.warning(f"pymupdf4llm failed ({e}), falling back to pdfplumber...")

    # Fallback: pdfplumber
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        text = "\n\n".join(text_parts)
        logger.debug(f"pdfplumber OK -- {len(text)} characters extracted")
        return _clean_text(text)
    except Exception as e:
        logger.error(f"pdfplumber also failed ({e})")
        raise RuntimeError(f"Unable to extract text from {pdf_path}") from e


def _clean_markdown_text(text: str) -> str:
    """Clean up Markdown text produced by pymupdf4llm."""
    # Remove excessive Markdown headings while keeping the text content
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Remove empty code fences
    text = re.sub(r'```\s*```', '', text)
    # Normalize multiple blank lines and repeated spaces
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def _clean_text(text: str) -> str:
    """Basic cleanup for raw text extracted by pdfplumber."""
    # Remove non-printable control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Normalize whitespace
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ---------------------------------------------------------------------------
# Embedding cache (avoids recomputing embeddings on repeated runs)
# ---------------------------------------------------------------------------

def _load_embedding_cache() -> dict:
    """Load the on-disk embedding cache. Returns an empty dict if missing or corrupt."""
    if os.path.exists(config.EMBEDDING_CACHE_PATH):
        try:
            with open(config.EMBEDDING_CACHE_PATH, "rb") as f:
                cache = pickle.load(f)
            logger.debug(f"Embedding cache loaded: {len(cache)} entries")
            return cache
        except Exception as e:
            logger.warning(f"Embedding cache is corrupt, ignoring it: {e}")
    return {}


def _save_embedding_cache(cache: dict):
    """Persist the embedding cache to disk."""
    try:
        with open(config.EMBEDDING_CACHE_PATH, "wb") as f:
            pickle.dump(cache, f)
    except Exception as e:
        logger.warning(f"Could not save embedding cache: {e}")


def _cache_key(text: str) -> str:
    """Generate a unique cache key for a given text and embedding model combination."""
    return hashlib.md5(f"{config.EMBEDDING_MODEL}:{text}".encode()).hexdigest()


# ---------------------------------------------------------------------------
# Embeddings via OpenRouter
# ---------------------------------------------------------------------------

async def _get_embeddings_batch(
    sentences: List[str],
    client: httpx.AsyncClient,
    cache: dict,
    semaphore: asyncio.Semaphore,
) -> List[List[float]]:
    """
    Compute embeddings for a list of sentences, using the on-disk cache
    to avoid duplicate API calls. Processes sentences in batches of
    EMBEDDING_BATCH_SIZE to stay within API limits.
    """
    # Separate cached sentences from those that need to be fetched
    results = [None] * len(sentences)
    uncached_indices = []
    uncached_sentences = []

    for i, sentence in enumerate(sentences):
        key = _cache_key(sentence)
        if key in cache:
            results[i] = cache[key]
        else:
            uncached_indices.append(i)
            uncached_sentences.append(sentence)

    if not uncached_sentences:
        return results

    logger.debug(f"Computing embeddings for {len(uncached_sentences)} sentences (not cached)")

    batch_size = config.EMBEDDING_BATCH_SIZE
    for batch_start in range(0, len(uncached_sentences), batch_size):
        batch = uncached_sentences[batch_start : batch_start + batch_size]
        batch_indices = uncached_indices[batch_start : batch_start + batch_size]

        async with semaphore:
            for attempt in range(config.MAX_RETRIES):
                try:
                    response = await client.post(
                        f"{config.OPENROUTER_BASE_URL}/embeddings",
                        headers={
                            "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
                            "HTTP-Referer": config.APP_URL,
                            "X-Title": config.APP_NAME,
                        },
                        json={
                            "model": config.EMBEDDING_MODEL,
                            "input": batch,
                        },
                        timeout=60.0,
                    )
                    response.raise_for_status()
                    data = response.json()

                    # Map results back to their original positions
                    embeddings_data = sorted(data["data"], key=lambda x: x["index"])
                    for j, emb_obj in enumerate(embeddings_data):
                        embedding = emb_obj["embedding"]
                        orig_idx = batch_indices[j]
                        results[orig_idx] = embedding
                        # Store in cache
                        cache[_cache_key(uncached_sentences[batch_start + j])] = embedding

                    break  # success, exit retry loop

                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429:
                        wait = config.RATE_LIMIT_BACKOFF_SECONDS * (attempt + 1)
                        logger.warning(f"Rate limit hit on embeddings -- waiting {wait}s")
                        await asyncio.sleep(wait)
                    else:
                        logger.error(f"HTTP error on embedding request: {e}")
                        if attempt == config.MAX_RETRIES - 1:
                            raise
                except Exception as e:
                    logger.error(f"Embedding error (attempt {attempt+1}): {e}")
                    if attempt == config.MAX_RETRIES - 1:
                        raise
                    await asyncio.sleep(2 ** attempt)

    return results


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute the cosine similarity between two embedding vectors."""
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


# ---------------------------------------------------------------------------
# Semantic chunking
# ---------------------------------------------------------------------------

async def semantic_chunk_text(text: str) -> List[str]:
    """
    Split text into semantically coherent chunks using embedding similarity.

    Pipeline:
      1. Tokenize the text into sentences (NLTK)
      2. Compute an embedding for each sentence via OpenRouter
      3. Calculate cosine similarity between consecutive sentence embeddings
      4. Detect topic boundaries where similarity drops significantly
      5. Group sentences into chunks at those boundaries
      6. Post-process: merge chunks that are too small, split chunks that are too large
    """
    # Step 1: sentence tokenization
    sentences = _split_into_sentences(text)
    logger.info(f"  -> {len(sentences)} sentences extracted")

    if len(sentences) < config.MIN_CHUNK_SENTENCES:
        logger.info("  -> Text too short: returning as a single chunk")
        return [text]

    # Step 2: compute embeddings
    cache = _load_embedding_cache()
    embedding_semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_EMBEDDING_REQUESTS)

    async with httpx.AsyncClient() as client:
        embeddings = await _get_embeddings_batch(sentences, client, cache, embedding_semaphore)

    _save_embedding_cache(cache)

    # Verify all embeddings were computed successfully
    valid_mask = [e is not None for e in embeddings]
    if not all(valid_mask):
        n_failed = valid_mask.count(False)
        logger.warning(f"  -> {n_failed} sentences missing embeddings, using fallback chunking")
        return _fallback_chunk(sentences)

    # Step 3: cosine similarity between consecutive sentences
    similarities = []
    for i in range(len(sentences) - 1):
        sim = _cosine_similarity(embeddings[i], embeddings[i + 1])
        similarities.append(sim)

    # Step 4: detect breakpoints using percentile threshold
    breakpoints = _detect_breakpoints(similarities)
    logger.info(f"  -> {len(breakpoints)} semantic breakpoints detected")

    # Step 5: build chunks
    chunks = _build_chunks_from_breakpoints(sentences, breakpoints)

    # Step 6: post-process
    chunks = _postprocess_chunks(chunks)

    logger.info(f"  -> {len(chunks)} final chunks produced")
    return chunks


def _split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using NLTK.
    Tries Italian language model first; falls back to the default tokenizer.
    """
    try:
        sentences = sent_tokenize(text, language="italian")
    except Exception:
        sentences = sent_tokenize(text)

    # Filter out very short fragments likely caused by extraction artifacts
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    return sentences


def _detect_breakpoints(similarities: List[float]) -> List[int]:
    """
    Identify sentence indices where a new chunk should begin.

    Uses a percentile-based approach: a breakpoint is created at every position
    where the similarity score falls below the (100 - threshold)-th percentile,
    indicating a significant topic shift.
    """
    if not similarities:
        return []

    arr = np.array(similarities)
    threshold = np.percentile(arr, 100 - config.SIMILARITY_THRESHOLD_PERCENTILE)
    breakpoints = [i for i, sim in enumerate(similarities) if sim < threshold]
    return breakpoints


def _build_chunks_from_breakpoints(
    sentences: List[str], breakpoints: List[int]
) -> List[str]:
    """Group sentences into chunks, splitting at each detected breakpoint."""
    chunks = []
    start = 0
    breakpoint_set = set(breakpoints)

    for i in range(len(sentences)):
        if i in breakpoint_set:
            chunk_text = " ".join(sentences[start : i + 1])
            chunks.append(chunk_text)
            start = i + 1

    # Append the final chunk
    if start < len(sentences):
        chunk_text = " ".join(sentences[start:])
        chunks.append(chunk_text)

    return [c for c in chunks if c.strip()]


def _postprocess_chunks(chunks: List[str]) -> List[str]:
    """
    Post-process chunks to enforce size constraints:
    - Merge chunks that are too small into the previous one.
    - Split chunks that exceed the maximum character limit.
    """
    processed = []

    for chunk in chunks:
        sentences_in_chunk = _split_into_sentences(chunk)

        # Merge with the previous chunk if this one is too short
        if len(sentences_in_chunk) < config.MIN_CHUNK_SENTENCES and processed:
            processed[-1] = processed[-1] + " " + chunk
            continue

        # Split if the chunk exceeds the character limit
        if len(chunk) > config.MAX_CHUNK_CHARS:
            sub_chunks = _split_large_chunk(sentences_in_chunk)
            processed.extend(sub_chunks)
        else:
            processed.append(chunk)

    return [c.strip() for c in processed if c.strip()]


def _split_large_chunk(sentences: List[str]) -> List[str]:
    """Break an oversized chunk into smaller sub-chunks that fit within MAX_CHUNK_CHARS."""
    sub_chunks = []
    current_sentences = []
    current_length = 0

    for sentence in sentences:
        if (
            current_length + len(sentence) > config.MAX_CHUNK_CHARS
            and len(current_sentences) >= config.MIN_CHUNK_SENTENCES
        ):
            sub_chunks.append(" ".join(current_sentences))
            current_sentences = []
            current_length = 0

        current_sentences.append(sentence)
        current_length += len(sentence)

    if current_sentences:
        sub_chunks.append(" ".join(current_sentences))

    return sub_chunks


def _fallback_chunk(sentences: List[str]) -> List[str]:
    """Fixed-size chunking fallback used when embedding computation fails."""
    chunks = []
    target_size = (config.MIN_CHUNK_SENTENCES + config.MAX_CHUNK_SENTENCES) // 2

    for i in range(0, len(sentences), target_size):
        chunk = " ".join(sentences[i : i + target_size])
        if chunk.strip():
            chunks.append(chunk.strip())

    return chunks


# ---------------------------------------------------------------------------
# Public entry point: read and chunk all PDFs in the source folder
# ---------------------------------------------------------------------------

async def process_all_pdfs() -> List[Tuple[str, List[str]]]:
    """
    Read every PDF in the configured source directory and chunk it semantically.

    Returns:
        A list of (pdf_name, list_of_chunks) tuples, one entry per PDF file.
    """
    source_dir = Path(config.SOURCE_DIRECTORY)
    if not source_dir.exists():
        raise FileNotFoundError(
            f"Source folder not found: {source_dir}\n"
            "Create the 'source/' directory and place your PDF files inside it."
        )

    pdf_files = sorted(source_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {source_dir}")

    logger.info(f"Found {len(pdf_files)} PDF file(s) in '{source_dir.name}/'")

    results = []
    for pdf_path in pdf_files:
        pdf_name = pdf_path.stem  # filename without extension
        logger.info(f"\n[PDF] Processing: {pdf_path.name}")

        try:
            text = extract_text_from_pdf(str(pdf_path))
            if not text.strip():
                logger.warning(f"  [!] No text extracted from {pdf_path.name}, skipping.")
                continue

            logger.info(f"  -> {len(text):,} characters extracted")
            chunks = await semantic_chunk_text(text)
            results.append((pdf_name, chunks))

        except Exception as e:
            logger.error(f"  [X] Failed to process {pdf_path.name}: {e}")
            continue

    return results
