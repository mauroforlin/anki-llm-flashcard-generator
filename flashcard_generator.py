#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# flashcard_generator.py -- Main Entry Point
# Anki Flashcard Generator v2.0 -- Powered by OpenRouter
# =============================================================================

import asyncio
import io
import logging
import sys
import time
from pathlib import Path

from tqdm import tqdm

import config

# ---------------------------------------------------------------------------
# Windows encoding fix -- prevents crashes with emoji/unicode on cp1252 consoles
# ---------------------------------------------------------------------------
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from pdf_reader import process_all_pdfs
from flashcard_maker import generate_all_flashcards
from anki_exporter import export_to_apkg, print_summary


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool = False):
    """
    Configure logging: INFO level to the console, DEBUG level to a log file.
    Pass --verbose or -v at the command line to enable DEBUG on the console too.
    """
    log_level = logging.DEBUG if verbose else logging.INFO

    # Console handler (clean, message-only format)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    # File handler (detailed format for debugging)
    log_file = Path(config.ROOT_DIRECTORY) / "flashcard_generator.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Silence noisy third-party libraries
    for noisy_lib in ["httpx", "httpcore", "urllib3"]:
        logging.getLogger(noisy_lib).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------

def validate_config():
    """Check that the configuration is valid before starting any work."""
    errors = []

    if not config.OPENROUTER_API_KEY or config.OPENROUTER_API_KEY == "YOUR-OPENROUTER-API-KEY":
        errors.append(
            "[X] OPENROUTER_API_KEY is not set in your .env file.\n"
            "    Copy .env.example to .env and insert your OpenRouter API key."
        )

    source_dir = Path(config.SOURCE_DIRECTORY)
    if not source_dir.exists():
        errors.append(
            f"[X] Source folder not found: {config.ROOT_DIRECTORY}/source/\n"
            "    Create the 'source/' directory and place your PDF files inside it."
        )
    else:
        pdf_count = len(list(source_dir.glob("*.pdf")))
        if pdf_count == 0:
            errors.append(
                "[X] No PDF files found in the 'source/' folder.\n"
                "    Add at least one PDF file and try again."
            )

    if errors:
        print("\n" + "=" * 60)
        print("  CONFIGURATION ERRORS")
        print("=" * 60)
        for error in errors:
            print(f"\n{error}")
        print("\n" + "=" * 60 + "\n")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Startup banner
# ---------------------------------------------------------------------------

def print_banner():
    """Print a startup banner summarizing the active configuration."""
    print("\n" + "=" * 60)
    print("  ANKI FLASHCARD GENERATOR v2.0")
    print("  Powered by OpenRouter  |  Semantic Chunking")
    print("=" * 60)
    print(f"  LLM model       : {config.CHAT_MODEL}")
    print(f"  Embedding model : {config.EMBEDDING_MODEL}")
    print(f"  Rate limit      : {config.MAX_REQUESTS_PER_MINUTE} req/min")
    print(f"  LLM workers     : {config.MAX_CONCURRENT_LLM_REQUESTS}")
    print(f"  Flashcard style : {config.FLASHCARD_STYLE} (dynamic density)")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    """Main pipeline: PDF files -> semantic chunks -> flashcards -> .apkg"""
    setup_logging(verbose="--verbose" in sys.argv or "-v" in sys.argv)
    logger = logging.getLogger(__name__)

    print_banner()
    validate_config()

    start_time = time.time()

    # -- Phase 1: PDF reading and semantic chunking ---------------------------
    print("PHASE 1/3 -- Reading PDFs and running semantic chunking...")
    print("-" * 60)

    pdf_chunks = await process_all_pdfs()

    if not pdf_chunks:
        print("\n[X] No PDFs were processed successfully. Check the 'source/' folder.")
        sys.exit(1)

    total_chunks = sum(len(chunks) for _, chunks in pdf_chunks)
    print(f"\n  [OK] {len(pdf_chunks)} PDF(s) processed -> {total_chunks} semantic chunks\n")

    # -- Phase 2: Parallel flashcard generation -------------------------------
    print("PHASE 2/3 -- Generating flashcards (parallel)...")
    print("-" * 60)
    print(f"  Processing {total_chunks} chunks in parallel...\n")

    with tqdm(
        total=total_chunks,
        desc="  Chunks processed",
        unit="chunk",
        bar_format="  {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ncols=60,
    ) as pbar:
        flashcards_by_pdf = await generate_all_flashcards(
            pdf_chunks,
            progress_callback=pbar.update,
        )

    total_flashcards = sum(len(cards) for cards in flashcards_by_pdf.values())
    print(f"\n  [OK] {total_flashcards} flashcards generated\n")

    # -- Phase 3: Anki .apkg export -------------------------------------------
    print("PHASE 3/3 -- Exporting Anki deck (.apkg)...")
    print("-" * 60)

    try:
        output_path = export_to_apkg(flashcards_by_pdf)
    except ValueError as e:
        print(f"\n[X] {e}")
        sys.exit(1)

    # -- Final summary --------------------------------------------------------
    elapsed = time.time() - start_time
    minutes, seconds = divmod(int(elapsed), 60)

    print_summary(flashcards_by_pdf, output_path)
    print(f"  Total time: {minutes}m {seconds}s\n")


if __name__ == "__main__":
    # Use WindowsSelectorEventLoopPolicy for compatibility with httpx on Windows
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())
