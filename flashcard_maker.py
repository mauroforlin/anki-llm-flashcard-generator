# =============================================================================
# flashcard_maker.py -- Flashcard Generation with Parallelism and Rate Limiting
# =============================================================================

import asyncio
import json
import logging
import re
import time
from collections import deque
from typing import Dict, List, Tuple

import httpx

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rate limiter (sliding window over 60 seconds)
# ---------------------------------------------------------------------------

class RateLimiter:
    """
    Limits outgoing requests to MAX_REQUESTS_PER_MINUTE using a sliding
    60-second window. Safe for concurrent use with asyncio.Lock.
    """

    def __init__(self, max_per_minute: int):
        self.max_per_minute = max_per_minute
        self._timestamps: deque = deque()
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Block until a new request can be sent without exceeding the rate limit."""
        async with self._lock:
            now = time.monotonic()
            window_start = now - 60.0

            # Drop timestamps that have fallen outside the 60-second window
            while self._timestamps and self._timestamps[0] < window_start:
                self._timestamps.popleft()

            if len(self._timestamps) >= self.max_per_minute:
                # Calculate how long to wait until the oldest request expires
                oldest = self._timestamps[0]
                wait_time = 60.0 - (now - oldest) + 0.1
                if wait_time > 0:
                    logger.debug(f"Rate limit: waiting {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
                    # Recompute after the sleep
                    now = time.monotonic()
                    window_start = now - 60.0
                    while self._timestamps and self._timestamps[0] < window_start:
                        self._timestamps.popleft()

            self._timestamps.append(time.monotonic())


# Global rate limiter instance
_rate_limiter = RateLimiter(config.MAX_REQUESTS_PER_MINUTE)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert academic educator specialized in creating high-quality Anki "
    "flashcards for university-level study, including medicine, biomedical sciences, "
    "and other scientific disciplines.\n\n"
    "Your task is to analyze text extracted from university lecture notes and produce "
    "flashcards that are ideal for active recall and spaced-repetition review in Anki.\n\n"
    "IMPORTANT: Always write the flashcard content in the SAME LANGUAGE as the source "
    "text. If the lecture notes are in Italian, write Italian flashcards. If they are "
    "in English, write English flashcards. Never translate the content.\n\n"
    "Always respond with a valid JSON array and nothing else."
)


def _build_user_prompt(chunk_text: str, pdf_name: str) -> str:
    return f"""Analyze the following text extracted from the lecture \"{pdf_name}\" and \
create between {config.FLASHCARDS_MIN_PER_CHUNK} and {config.FLASHCARDS_MAX_PER_CHUNK} \
Anki flashcards.

Write all flashcard content in the SAME LANGUAGE as the source text below.

=============================================================
RULES FOR HIGH-QUALITY FLASHCARDS:
=============================================================

1. ATOMIC PRINCIPLE: each flashcard tests exactly ONE key concept. Do not combine
   multiple distinct ideas into a single question.

2. QUESTION TYPES to use (vary the type across cards):
   - Definition: "What is [term]?" / "How is [concept] defined?"
   - Mechanism: "What is the mechanism of [physiological/pathological process]?"
   - Classification: "How are [entities] classified?" / "What are the types of [X]?"
   - Numerical values: "What is the normal/threshold value of [parameter]?" -- always
     include the unit of measurement in the answer.
   - Pathogenesis: "What is the pathogenesis of [disease]?"
   - Differential diagnosis: "How does [X] differ from [Y]?"
   - Cause and effect: "What causes [event]?" / "What are the consequences of [X]?"
   - Treatment: "How is [condition] treated?" / "What is the therapy for [X]?"

3. QUESTION QUALITY:
   - Be specific and unambiguous: each question must have exactly one correct answer.
   - Avoid trivial questions with obvious answers.
   - Only use information that is explicitly stated in the provided text.

4. ANSWER QUALITY:
   - Keep answers concise but complete: 1 to 4 sentences at most.
   - Use a bullet list (with dashes) only when there are 3 or more items to enumerate.
   - Always include numerical values, units, and clinical thresholds when relevant.
   - Summarize clearly -- do not copy entire paragraphs verbatim.

5. EXCLUSIONS: do not create flashcards based on introductory context, purely
   illustrative examples, or incomplete/very short sentences in the text.

=============================================================
RESPONSE FORMAT (pure JSON array, no extra text before or after):
=============================================================

[
  {{"question": "Clear and specific question?", "answer": "Concise and complete answer."}},
  {{"question": "Another question?", "answer": "Another answer."}}
]

=============================================================
TEXT TO ANALYZE:
=============================================================

{chunk_text}"""


# ---------------------------------------------------------------------------
# LLM response parsing and validation
# ---------------------------------------------------------------------------

def _parse_flashcards_response(raw_response: str) -> List[Dict[str, str]]:
    """
    Extract and validate flashcards from a raw LLM response string.
    Handles slightly malformed JSON via multiple fallback strategies.
    """
    # Strip markdown code fences if present (e.g., ```json ... ```)
    clean = re.sub(r"```(?:json)?\s*", "", raw_response, flags=re.DOTALL)
    clean = re.sub(r"```\s*$", "", clean, flags=re.MULTILINE).strip()

    # Attempt 1: direct JSON parsing
    try:
        data = json.loads(clean)
        if isinstance(data, list):
            return _validate_flashcards(data)
    except json.JSONDecodeError:
        pass

    # Attempt 2: extract the JSON array from somewhere inside the response
    match = re.search(r"\[\s*\{.*?\}\s*\]", clean, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return _validate_flashcards(data)
        except json.JSONDecodeError:
            pass

    # Attempt 3: extract individual JSON objects using regex
    objects = re.findall(
        r'\{\s*"question"\s*:\s*"[^"]+"\s*,\s*"answer"\s*:\s*"[^"]+"\s*\}', clean
    )
    if objects:
        try:
            data = [json.loads(obj) for obj in objects]
            return _validate_flashcards(data)
        except json.JSONDecodeError:
            pass

    logger.warning("Could not parse LLM response as valid JSON")
    return []


def _validate_flashcards(data: list) -> List[Dict[str, str]]:
    """Filter out any items that are missing a non-empty question or answer."""
    valid = []
    for item in data:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if question and answer and len(question) > 5 and len(answer) > 5:
            valid.append({"question": question, "answer": answer})
    return valid


# ---------------------------------------------------------------------------
# Single-chunk LLM call with retry
# ---------------------------------------------------------------------------

async def _generate_flashcards_for_chunk(
    chunk_text: str,
    pdf_name: str,
    chunk_index: int,
    total_chunks: int,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    progress_callback=None,
) -> List[Dict[str, str]]:
    """
    Generate flashcards for a single text chunk with rate limiting and automatic retry.
    """
    for attempt in range(config.MAX_RETRIES):
        try:
            # Wait for permission from the rate limiter before sending
            await _rate_limiter.acquire()

            async with semaphore:
                response = await client.post(
                    f"{config.OPENROUTER_BASE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
                        "HTTP-Referer": config.APP_URL,
                        "X-Title": config.APP_NAME,
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": config.CHAT_MODEL,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": _build_user_prompt(chunk_text, pdf_name)},
                        ],
                        "temperature": config.LLM_TEMPERATURE,
                        "max_tokens": config.LLM_MAX_TOKENS,
                    },
                    timeout=120.0,
                )
                response.raise_for_status()
                data = response.json()

            raw_content = data["choices"][0]["message"]["content"]
            flashcards = _parse_flashcards_response(raw_content)

            if flashcards:
                if progress_callback:
                    progress_callback()
                return flashcards
            else:
                logger.warning(
                    f"  [!] Chunk {chunk_index+1}/{total_chunks} of '{pdf_name}': "
                    f"empty or unparseable response (attempt {attempt+1})"
                )

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                wait = config.RATE_LIMIT_BACKOFF_SECONDS * (attempt + 1)
                logger.warning(f"Rate limit 429 -- waiting {wait}s before retry")
                await asyncio.sleep(wait)
            else:
                logger.error(
                    f"  [X] HTTP {e.response.status_code} on chunk {chunk_index+1} "
                    f"of '{pdf_name}': {e}"
                )
                if attempt == config.MAX_RETRIES - 1:
                    return []
        except Exception as e:
            logger.error(
                f"  [X] Error on chunk {chunk_index+1}/{total_chunks} of '{pdf_name}' "
                f"(attempt {attempt+1}): {e}"
            )
            if attempt < config.MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                return []

    return []


# ---------------------------------------------------------------------------
# Main parallel orchestration
# ---------------------------------------------------------------------------

async def generate_all_flashcards(
    pdf_chunks: List[Tuple[str, List[str]]],
    progress_callback=None,
) -> Dict[str, List[Dict[str, str]]]:
    """
    Generate flashcards for all PDFs in parallel, respecting the rate limit.

    Args:
        pdf_chunks:        List of (pdf_name, list_of_chunks) from pdf_reader.
        progress_callback: Optional callable invoked after each completed chunk (for tqdm).

    Returns:
        A dict mapping each pdf_name to its list of generated flashcards.
    """
    total_chunks = sum(len(chunks) for _, chunks in pdf_chunks)
    logger.info(f"\n[START] Parallel generation: {total_chunks} total chunks")
    logger.info(f"  Model:           {config.CHAT_MODEL}")
    logger.info(f"  Rate limit:      {config.MAX_REQUESTS_PER_MINUTE} req/min")
    logger.info(f"  Parallel workers:{config.MAX_CONCURRENT_LLM_REQUESTS}")

    semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_LLM_REQUESTS)

    # Build a flat list of all tasks with their metadata
    tasks_info = []  # (pdf_name, chunk_idx, total_pdf_chunks, chunk_text)
    for pdf_name, chunks in pdf_chunks:
        for i, chunk in enumerate(chunks):
            tasks_info.append((pdf_name, i, len(chunks), chunk))

    # Initialize the results dict
    results: Dict[str, List[Dict[str, str]]] = {
        pdf_name: [] for pdf_name, _ in pdf_chunks
    }

    async with httpx.AsyncClient() as client:
        coroutines = [
            _generate_flashcards_for_chunk(
                chunk_text=chunk_text,
                pdf_name=pdf_name,
                chunk_index=chunk_idx,
                total_chunks=total_pdf_chunks,
                client=client,
                semaphore=semaphore,
                progress_callback=progress_callback,
            )
            for pdf_name, chunk_idx, total_pdf_chunks, chunk_text in tasks_info
        ]

        # Run all coroutines concurrently
        all_results = await asyncio.gather(*coroutines, return_exceptions=True)

    # Assemble results by PDF
    for i, result in enumerate(all_results):
        pdf_name = tasks_info[i][0]
        if isinstance(result, Exception):
            logger.error(f"  [X] Task failed for '{pdf_name}': {result}")
        elif result:
            results[pdf_name].extend(result)

    # Final summary
    total_flashcards = sum(len(cards) for cards in results.values())
    logger.info(f"\n[DONE] Generation complete: {total_flashcards} total flashcards")
    for pdf_name, cards in results.items():
        logger.info(f"  * {pdf_name}: {len(cards)} flashcards")

    return results
