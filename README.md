# Anki Flashcard Generator

By [Mauro Forlin](https://github.com/mauroforlin)

A simple but effective command-line tool that automatically generates Anki flashcards from PDF lecture notes using large language models via [OpenRouter](https://openrouter.ai).

The idea is straightforward: drop your lecture PDFs into a folder, run one command, and get a ready-to-import `.apkg` Anki deck with the cards organized by topic. The tool handles everything in between -- extracting text from the PDFs, splitting it into semantically coherent chunks using embedding similarity, sending those chunks to an LLM in parallel, and packaging the results into a proper Anki deck with one sub-deck per source file.

It was built as a personal study tool and is intentionally kept simple. There is no web interface, no database, no complex configuration. Just Python, a few libraries, and an OpenRouter API key.

---

## Features

| Feature | Details |
|---|---|
| **LLM backend** | Any model available on OpenRouter (default: `deepseek/deepseek-v4-flash`) |
| **Semantic chunking** | Splits PDFs into topic-coherent sections using embedding cosine similarity |
| **Parallel generation** | Up to 20 concurrent LLM requests with a built-in sliding-window rate limiter |
| **Anki output** | A single `.apkg` file with one sub-deck per PDF source |
| **Embedding cache** | Embeddings are cached on disk so repeated runs skip recomputation |
| **Multilingual** | Flashcards are generated in the same language as the source text |
| **Auto-retry** | Up to 3 automatic retries per chunk on API errors |

---

## Requirements

- Python 3.10 or later
- An [OpenRouter](https://openrouter.ai) account and API key

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your API key

Copy the example environment file and fill in your key:

```bash
cp .env.example .env
```

Then open `.env` and replace the placeholder with your actual OpenRouter API key:

```
OPENROUTER_API_KEY=sk-or-...
```

You can get a key at [openrouter.ai/keys](https://openrouter.ai/keys). The `.env` file is listed in `.gitignore` and will never be committed.

### 3. (Optional) Change the LLM model

The default model is `deepseek/deepseek-v4-flash`. You can swap it for any chat model available on OpenRouter by editing `config.py`:

```python
CHAT_MODEL = "google/gemini-2.0-flash-001"
CHAT_MODEL = "anthropic/claude-3.5-haiku"
CHAT_MODEL = "meta-llama/llama-3.3-70b-instruct"
```

---

## Usage

### 1. Add your PDF files

Place all your lecture PDFs inside the `source/` folder. The filename of each PDF becomes the name of its corresponding sub-deck in Anki, so name them descriptively.

### 2. Run the generator

```bash
python flashcard_generator.py
```

For verbose debug output (useful for troubleshooting):

```bash
python flashcard_generator.py --verbose
```

### 3. Import into Anki

The output file is saved to `output/flashcards_YYYYMMDD_HHMMSS.apkg`. Open Anki and go to **File -> Import**, then select the file.

---

## Rate Limits

> [!IMPORTANT]
> The default rate limit settings (`MAX_REQUESTS_PER_MINUTE = 185`, `MAX_CONCURRENT_LLM_REQUESTS = 20`) are tuned for a **paid OpenRouter plan**, which allows up to 200 requests per minute.
>
> If you are using **free-tier models**, you must significantly lower these values. Free models on OpenRouter are typically limited to around 20 requests per minute or fewer, depending on the specific model.
>
> You can check your current plan and rate limits at [openrouter.ai/settings/limits](https://openrouter.ai/settings/limits).

To adjust the limits, open `config.py` and change:

```python
# Example values for a free-tier plan
MAX_REQUESTS_PER_MINUTE = 18       # stay safely below the free limit
MAX_CONCURRENT_LLM_REQUESTS = 5    # reduce parallel workers accordingly
```

---

## Project Structure

```
Anki_FlashCard_Generator/
├── flashcard_generator.py   # entry point -- run this
├── config.py                # all settings (API key, models, parameters)
├── pdf_reader.py            # PDF extraction and semantic chunking
├── flashcard_maker.py       # parallel LLM calls and rate limiting
├── anki_exporter.py         # .apkg file creation with genanki
├── requirements.txt
├── source/                  # place your PDF files here
└── output/                  # generated .apkg files appear here
```

---

## Deck Structure in Anki

If your `source/` folder contains:

```
lecture 01 - anemia.pdf
lecture 02 - thalassemia.pdf
lecture 03 - iron metabolism.pdf
```

The generated `.apkg` will contain:

```
Lecture Notes
├── lecture 01 - anemia         (N flashcards)
├── lecture 02 - thalassemia    (N flashcards)
└── lecture 03 - iron metabolism (N flashcards)
```

---

## Configuration Reference

All settings live in `config.py`. The most useful ones are:

| Parameter | Default | Description |
|---|---|---|
| `CHAT_MODEL` | `deepseek/deepseek-v4-flash` | LLM used to generate flashcards |
| `EMBEDDING_MODEL` | `qwen/qwen3-embedding-8b` | Embedding model for semantic chunking |
| `MAX_REQUESTS_PER_MINUTE` | `185` | Rate limit (adjust for your OpenRouter plan) |
| `MAX_CONCURRENT_LLM_REQUESTS` | `20` | Parallel LLM workers |
| `SIMILARITY_THRESHOLD_PERCENTILE` | `85` | Chunking sensitivity (range 60-95; higher = smaller chunks) |
| `FLASHCARD_STYLE` | `"atomic"` | Generation style: `"atomic"` or `"comprehensive"` |
| `CHARS_PER_FLASHCARD_ATOMIC` | `600` | Target character density for atomic cards |
| `CHARS_PER_FLASHCARD_COMPREHENSIVE` | `1200` | Target character density for comprehensive cards |
| `ANKI_DECK_NAME` | `Lecture Notes` | Name of the top-level Anki deck |

---

## Troubleshooting

**Getting 429 errors repeatedly:** lower `MAX_REQUESTS_PER_MINUTE` and `MAX_CONCURRENT_LLM_REQUESTS` in `config.py`. The tool retries automatically, but a more conservative rate is safer.

**Scanned PDFs (image-only):** the tool does not support OCR. Use a tool such as Adobe Acrobat or [Tesseract](https://github.com/tesseract-ocr/tesseract) to convert the PDF to a text-based format first.

**Chunks feel too small or too large:** adjust `SIMILARITY_THRESHOLD_PERCENTILE` in `config.py`. Lower values (e.g. 70) produce larger, broader chunks; higher values (e.g. 92) produce smaller, tightly focused ones.

---

## License

MIT -- see [LICENSE](LICENSE) for details.