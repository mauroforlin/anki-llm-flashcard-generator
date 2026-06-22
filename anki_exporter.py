# =============================================================================
# anki_exporter.py -- Anki .apkg file creation using genanki
# =============================================================================

import hashlib
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import genanki

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Card HTML/CSS templates
# ---------------------------------------------------------------------------

CARD_FRONT_TEMPLATE = """
<div class="card front">
  <div class="question-label">Question</div>
  <div class="question">{{Question}}</div>
  <div class="topic-badge">{{Topic}}</div>
</div>
"""

CARD_BACK_TEMPLATE = """
{{FrontSide}}
<hr class="divider">
<div class="card back">
  <div class="answer-label">Answer</div>
  <div class="answer">{{Answer}}</div>
</div>
"""

CARD_CSS = """
/* =====================================================
   Anki Flashcard Stylesheet
   ===================================================== */

.card {
  font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
  font-size: 16px;
  line-height: 1.6;
  color: #1a1a2e;
  background: #f0f4ff;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 20px;
  box-sizing: border-box;
}

.front, .back {
  max-width: 680px;
  width: 100%;
  padding: 32px 36px;
  border-radius: 16px;
  box-shadow: 0 4px 24px rgba(67, 97, 238, 0.12);
  background: #ffffff;
}

.question-label, .answer-label {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  margin-bottom: 12px;
}

.question-label {
  color: #4361ee;
}

.answer-label {
  color: #2ec4b6;
}

.question {
  font-size: 19px;
  font-weight: 600;
  color: #1a1a2e;
  line-height: 1.5;
  margin-bottom: 16px;
}

.answer {
  font-size: 16px;
  color: #333366;
  line-height: 1.7;
}

.topic-badge {
  display: inline-block;
  margin-top: 18px;
  padding: 4px 12px;
  background: #eef1ff;
  color: #4361ee;
  border-radius: 20px;
  font-size: 12px;
  font-weight: 500;
  letter-spacing: 0.3px;
}

.divider {
  border: none;
  border-top: 2px solid #e8eeff;
  margin: 24px 0;
}

/* Lists inside answers */
.answer ul, .answer ol {
  padding-left: 20px;
  margin: 8px 0;
}

.answer li {
  margin-bottom: 6px;
}

/* Anki night mode */
.night_mode .card {
  background: #1a1a2e;
  color: #e8eeff;
}

.night_mode .front,
.night_mode .back {
  background: #16213e;
  box-shadow: 0 4px 24px rgba(0, 0, 0, 0.4);
}

.night_mode .question {
  color: #e8eeff;
}

.night_mode .answer {
  color: #b8c0ff;
}

.night_mode .topic-badge {
  background: #0f3460;
  color: #a5b4fc;
}

.night_mode .divider {
  border-top-color: #1f2d5a;
}
"""


# ---------------------------------------------------------------------------
# Stable ID generation (prevents duplicates on repeated imports into Anki)
# ---------------------------------------------------------------------------

def _stable_id(name: str, offset: int = 0) -> int:
    """
    Generate a stable Anki numeric ID (between 1<<30 and 1<<31) derived from
    a hash of the given name. Using a stable ID means re-importing the same
    deck updates existing notes rather than creating duplicates.
    """
    h = int(hashlib.md5(name.encode()).hexdigest(), 16)
    base = (h % (1 << 30)) + (1 << 30)
    return base + offset


# ---------------------------------------------------------------------------
# Anki model definition
# ---------------------------------------------------------------------------

def _create_anki_model() -> genanki.Model:
    """Create the Anki note model with fields and the HTML/CSS card template."""
    return genanki.Model(
        _stable_id("AnkiFlashcardGenerator_ModelV2"),
        "Lecture Flashcard",
        fields=[
            {"name": "Question"},
            {"name": "Answer"},
            {"name": "Topic"},
        ],
        templates=[
            {
                "name": "Question -> Answer",
                "qfmt": CARD_FRONT_TEMPLATE,
                "afmt": CARD_BACK_TEMPLATE,
            }
        ],
        css=CARD_CSS,
    )


# ---------------------------------------------------------------------------
# .apkg export
# ---------------------------------------------------------------------------

def export_to_apkg(
    flashcards_by_pdf: Dict[str, List[Dict[str, str]]],
) -> str:
    """
    Create an Anki .apkg file containing one sub-deck per PDF source file.

    Args:
        flashcards_by_pdf: {pdf_name: [{"question": ..., "answer": ...}, ...]}

    Returns:
        Absolute path of the generated .apkg file.
    """
    output_dir = Path(config.OUTPUT_DIRECTORY)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"flashcards_{timestamp}.apkg"

    model = _create_anki_model()
    all_decks = []
    total_notes = 0
    skipped_pdfs = []

    for pdf_name, flashcards in flashcards_by_pdf.items():
        if not flashcards:
            logger.warning(f"  [!] No flashcards for '{pdf_name}', skipping sub-deck.")
            skipped_pdfs.append(pdf_name)
            continue

        # Sub-deck name: "Lecture Notes::PDF Name"
        deck_name = f"{config.ANKI_DECK_NAME}::{pdf_name}"
        deck_id = _stable_id(deck_name)
        deck = genanki.Deck(deck_id, deck_name)

        for card in flashcards:
            question = card.get("question", "").strip()
            answer = card.get("answer", "").strip()

            if not question or not answer:
                continue

            # Convert newlines to HTML line breaks for Anki rendering
            answer_html = answer.replace("\n- ", "\n* ").replace("\n", "<br>")

            note = genanki.Note(
                model=model,
                fields=[question, answer_html, pdf_name],
                guid=genanki.guid_for(deck_name, question),  # stable GUID for safe re-import
            )
            deck.add_note(note)
            total_notes += 1

        logger.info(f"  [OK] Sub-deck '{pdf_name}': {len(flashcards)} flashcards")
        all_decks.append(deck)

    if not all_decks:
        raise ValueError("No flashcards were generated. The .apkg file will not be created.")

    package = genanki.Package(all_decks)
    package.write_to_file(str(output_path))

    logger.info(f"\n[DONE] .apkg file written: {output_path}")
    logger.info(f"  Decks: {len(all_decks)} | Total flashcards: {total_notes}")
    if skipped_pdfs:
        logger.warning(f"  [!] PDFs with no flashcards (skipped): {', '.join(skipped_pdfs)}")

    return str(output_path)


def print_summary(flashcards_by_pdf: Dict[str, List[Dict[str, str]]], output_path: str):
    """Print a human-readable summary of the generation results."""
    total = sum(len(cards) for cards in flashcards_by_pdf.values())
    print("\n" + "=" * 60)
    print("  FLASHCARD GENERATION SUMMARY")
    print("=" * 60)
    for pdf_name, cards in flashcards_by_pdf.items():
        status = f"{len(cards):3d} flashcards" if cards else "  [EMPTY]"
        print(f"  * {pdf_name:<40} {status}")
    print("-" * 60)
    print(f"  TOTAL:  {total} flashcards across {len(flashcards_by_pdf)} topic(s)")
    print(f"  OUTPUT: {output_path}")
    print("=" * 60)
    print("\n  Import into Anki: File -> Import -> select the .apkg file\n")
