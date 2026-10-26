# Tanya Cukai

Ask Malaysian income tax questions in BM, English or rojak, and get an answer built only
from LHDN's own guides, with the exact passage cited. The AI model runs on your own
computer: no question is sent to OpenAI, Google or anyone else, and it works with Wi-Fi off.

This is the setup banks, GLCs and law firms in Malaysia ask for when they want AI but can't
let customer data leave the building.

## How it works

```
question ──► bge-m3 turns it into a vector ──► find the 6 closest LHDN passages (SQLite + numpy)
                                                        │
answer with [1][2] citations ◄── Qwen3-4B writes ◄──────┘  "answer ONLY from these passages"
```

1. **Fetch** (`scripts/fetch.py`): downloads the LHDN pages and PDFs listed in `sources.json`
   (tax reliefs, tax rates, rebates, deadlines, penalties, refunds, residency, the FAQ, Form BE 2025,
   the e-Invoice guideline).
2. **Ingest** (`scripts/ingest.py`): pulls the readable text out of each page (one section per
   heading, one row per table line), cuts it into ~900-character passages, and turns each passage
   into a vector with **bge-m3**, a multilingual embedding model. Because it is multilingual, a
   question in Malay lands near an English passage that means the same thing.
3. **Search** (`app/rag.py`): the question becomes a vector the same way; the closest passages
   are the ones whose vectors point in the most similar direction (cosine similarity).
4. **Answer**: the passages are numbered and handed to **Qwen3-4B-Instruct** with strict rules:
   use only these, cite them, reply in the user's language, say "I don't know" instead of guessing.
5. **Page** (`app/static/index.html`): streams the answer as it's written; each `[n]` opens the
   passage it came from, with a link to hasil.gov.my.

Both models run under **llama.cpp** (`llama-server`) on `127.0.0.1` only.

## Why retrieval, not fine-tuning, for the facts

Fine-tuning changes *how* a model writes. It is a poor way to teach it *facts*: the model can't
tell you where a figure came from, it mixes up years, and every Budget would mean retraining.
Retrieval keeps the facts in documents you can update and cite. Phase 2 of this project adds a
small LoRA fine-tune for the part fine-tuning is good at (tone and answer format) and compares the
two.

## Measured, not eyeballed

`eval/questions.json` holds 24 questions (8 BM, 9 English, 7 rojak), each tied to the LHDN passage
that contains the answer. `eval/run_eval.py` reports how often that passage is in the top 1, 3 and 6
results, split by language.

## Run it

Needs macOS or Linux, [uv](https://docs.astral.sh/uv/) and [llama.cpp](https://github.com/ggml-org/llama.cpp)
(`brew install llama.cpp`), and about 3.2 GB for the two models:

```bash
mkdir -p ~/.cache/llm-models && cd ~/.cache/llm-models
curl -LO https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf
curl -LO https://huggingface.co/ggml-org/bge-m3-Q8_0-GGUF/resolve/main/bge-m3-q8_0.gguf
cd -
./run.sh          # first run also downloads the LHDN sources and builds the index
```

Open http://127.0.0.1:8000. Measure retrieval with `uv run eval/run_eval.py` while `run.sh` is up.

## Limits

- General information from LHDN's public guides, not personal tax advice.
- A 4B model is much smaller than ChatGPT; the citations are there so every answer can be checked.
- LHDN's documents are not included in this repo; `fetch.py` downloads them from hasil.gov.my.
