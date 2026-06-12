# Local-First RAG Learning Assistant

A fully offline document intelligence system. Upload a PDF, ask questions, or start an adaptive teaching session grounded in the document. Nothing leaves your machine.

Every layer uses best-in-class methods: structure-preserving parsing (Docling), heading-aware chunking with table protection, hybrid dense+sparse retrieval (BGE-M3), cross-encoder reranking, local LLM generation (Ollama + Llama 3.2), and a LangGraph teaching agent that adapts to the learner. Every answer cites the exact section it came from.

---

## Demo

**Ask mode**
```
>> Upload: tools.pdf (16 pages)
   32 chunks extracted, 5468 words

>> Question: "What is a Pozidriv screwdriver?"

   A Pozidriv screwdriver is similar to Phillips but with additional
   ribs between the arms. Common in European-made furniture and equipment.

   Source: tools.pdf › 1.2 Fastening and Turning Tools
```

**Teach mode**
```
>> Topic: Phillips screwdriver | Level: beginner | Style: example

   Agent: Let me show you with an example. Imagine you're assembling
   flat-pack furniture — every screw has a cross-shaped slot. That slot
   was designed for a Phillips screwdriver...

>> Learner: got it, lets jump to Pozidriv screwdrivers

   Agent: [NEW_TOPIC detected] A Pozidriv looks like a Phillips at first
   glance, but has a second set of ribs between the arms...
```

The system never guesses. If the answer is not in the document, it says so.

---

## Architecture

### RAG Pipeline

```
PDF Upload
    |
    v
Docling Parser .............. structure-preserving markdown extraction
    |                         (headings as #/##, tables as |---|, images tagged)
    v
Structure-Aware Chunker ..... splits at heading boundaries, not word count
    |                         tables stay intact, heading trail prepended
    v
[Optional] Contextual        local Llama writes a situating sentence per chunk
Retrieval ................... uses smart context (doc start + neighborhood + end)
    |
    v
BGE-M3 Encoder ............. 1024-dim dense vectors + sparse lexical weights
    |
    v
ChromaDB + JSON Store ...... dense vectors (cosine) + sparse weights (persisted)


Question
    |
    v
BGE-M3 ..................... encode question (dense + sparse)
    |
    v
Dense Search ............... top 10 candidates from ChromaDB
    |
    v
Hybrid Scoring ............. 0.7 x dense_similarity + 0.3 x sparse_dot_product
    |
    v
Reranker ................... bge-reranker-v2-m3 reads question + chunk together
    |                        selects top 2 from 5 best hybrid candidates
    v
Ollama + Llama 3.2 3B ..... generates cited answer, fully local
    |
    v
Answer + Citations ......... filename + section path for every source
```

### Teaching Agent (LangGraph)

```
/teach/start  {topic, level, pace, strategy}
    |
    v
retrieve ................... BGE-M3 + hybrid search + reranker → top chunks
    |
    v
plan ....................... choose strategy (example / analogy / definition)
    |                        rotates on re-explain, honors learner requests
    v
generate ................... Llama reads chunks + learner profile + history
    |                        builds on prior explanation, never repeats
    v
[interrupt — wait for learner reply]
    |
    v
classify_intent ............ Llama classifies reply into one of five labels:
    |                        GOT_IT / CONFUSED / WANT_STRATEGY /
    |                        WORD_QUESTION / NEW_TOPIC
    v
route
    |-- GOT_IT        → end session
    |-- CONFUSED      → plan (new strategy) → generate
    |-- WANT_STRATEGY → plan (honor request) → generate
    |-- WORD_QUESTION → clarify word → generate (continue)
    |-- NEW_TOPIC     → retrieve (fresh topic) → plan → generate
```

---

## Why Each Piece Exists

| Component | Purpose | What breaks without it |
|---|---|---|
| Docling | Extracts markdown with real headings and table structure | Tables become jumbled text, headings lost, downstream chunking is blind |
| Structure-aware chunking | Splits at heading boundaries, keeps tables whole | Chunks slice through tables and merge unrelated topics |
| Heading trail | Prepends section path to every chunk | Chunks are orphaned from their document context |
| Contextual Retrieval | LLM writes a situating sentence per chunk | Chunks with vague headings ("Properties", "Results") embed poorly |
| BGE-M3 | Produces dense and sparse vectors in one pass | Dense-only search misses rare/exact terms |
| Hybrid search | Merges semantic (dense) and keyword (sparse) signals | Rare terms diluted in dense embeddings, exact matches lost |
| Reranker | Reads question and chunk side by side as a pair | Embedding comparison misses negation, qualifiers, subtle distinctions |
| Ollama + Llama 3.2 | Fully local generation, zero external API calls | Sensitive data (medical, financial, academic) would leave the machine |
| LangGraph | Stateful multi-turn agent with conditional routing | A chain runs once — can't loop, adapt strategy, or handle five intents |
| RAGAS evaluation | Automated scoring of faithfulness, relevancy, correctness | No way to measure if a change helped or hurt |

---

## Evaluation

Every answer is measurable. The project includes a RAGAS-style evaluation harness that scores answers on three dimensions using local Llama as judge:

- **Faithfulness**: is every claim grounded in the retrieved chunks? (catches hallucination)
- **Relevancy**: does the answer address the question that was asked? (catches tangential responses)
- **Correctness**: does the answer match the known ground truth? (end-to-end quality check)

Baseline results on a 16-page tools reference document, 10 golden questions:

| Configuration | Faithfulness | Relevancy | Correctness |
|---|---|---|---|
| Contextual Retrieval OFF | 1.00 | 0.75 | 0.71 |
| Contextual Retrieval ON | 1.00 | 0.80 | 0.77 |

Zero hallucination across all configurations. Contextual Retrieval improved relevancy and correctness with no loss in faithfulness.

---

## Getting Started

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed with Llama 3.2 pulled: `ollama pull llama3.2`
- 8GB+ RAM (tested on M2 MacBook Air, 8GB)

### Install

```bash
git clone git@github.com:gsoumyar/raglocal.git
cd raglocal
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
# Terminal 1 — API (ensure Ollama is running first)
uvicorn main:app
```

```bash
# Terminal 2 — Streamlit UI
streamlit run teach_app.py
```

> **Note:** Do not use `--reload` with uvicorn. Docling and HuggingFace write cache files into `.venv` during model loading, which triggers WatchFiles to restart the server in a loop.

### Usage

**Streamlit UI** (recommended): open `http://localhost:8501`
1. Upload a PDF in the sidebar
2. Use the **ASK** tab to ask questions and get cited answers
3. Use the **TEACH** tab to start an adaptive learning session — set your topic, level, pace, and explanation style

**Swagger UI**: open `http://127.0.0.1:8000/docs`

### Evaluate

```bash
# With the server running and a document uploaded:
python3 eval_ragas.py
```

Results append to `eval_results.json` for comparison across configurations.

---

## API

| Method | Endpoint | Description |
|---|---|---|
| POST | `/upload` | Upload a PDF. Query param: `contextual_retrieval` (bool, default false) |
| POST | `/ask` | Ask a question. Body: `{"text": "..."}`. Returns: answer, sources, retrieved_context |
| POST | `/teach/start` | Start a teaching session. Body: `{"topic", "level", "pace", "strategy"}`. Returns: thread_id, message, sources |
| POST | `/teach/reply` | Continue a session. Body: `{"thread_id", "reply"}`. Returns: intent, message, sources, done |
| GET | `/inspect` | List all stored chunks, sections, and metadata |
| DELETE | `/clear` | Wipe ChromaDB and sparse store |
| GET | `/` | Health check |
| GET | `/about` | Project info |

---

## Project Structure

```
raglocal/
    main.py                 FastAPI app — RAG pipeline + LangGraph teaching agent
    teach_app.py            Streamlit UI — sidebar upload, ASK tab, TEACH tab
    eval_ragas.py           RAGAS evaluation harness
    golden_test.json        10 question-answer pairs for evaluation
    requirements.txt        Python dependencies
    utils.py                Chunker, context helpers, hybrid scoring
    parse_test.py           Standalone: Docling vs PyMuPDF comparison
    bge_test.py             Standalone: BGE-M3 embedding validation
    ollama_test.py          Standalone: Ollama connection test
    rerank_test.py          Standalone: reranker validation
    static/                 Simple web UI
    chroma_db/              ChromaDB storage (auto-generated)
    sparse_store.json       Sparse vectors (auto-generated)
    eval_results.json       Evaluation history (auto-generated)
```

---

## Limitations

- **Single document at a time.** Each upload clears previous data. Deliberate tradeoff: avoids schema conflicts and stale embeddings at the cost of cross-document search.

- **Text-layer PDFs only.** OCR is disabled. Scanned documents produce empty output.

- **Same model as judge.** RAGAS evaluation uses Llama 3.2 as both generator and judge. Relative comparisons between configurations are reliable; absolute scores are softer than they would be with an independent stronger judge.

- **In-memory sparse store.** Sparse vectors load into memory at startup and persist to JSON on upload. Scales fine for single-document use; a sparse-capable vector store (Qdrant) would be needed for thousands of documents.

- **No streaming.** Answers generate in one shot. Long answers on slower hardware have a noticeable wait.

- **8GB RAM ceiling.** BGE-M3 (~2.2GB) + reranker (~1.1GB) + Llama 3.2 (~2GB) coexist in memory. Tested stable on M2 8GB with documents up to 1300 pages, but headroom is limited.

---

## Roadmap

**Stronger evaluation.** RAGAS with a 7B+ judge model for more reliable absolute scores. Context precision and context recall metrics alongside faithfulness, relevancy, and correctness.

**Multi-document support.** Upload and search across multiple documents with chunk-level document tagging and metadata filtering.

**Hybrid vector store.** Replace ChromaDB + JSON with a database that natively supports dense and sparse vectors (Qdrant, Milvus).

**Streaming responses.** Ollama streaming API for token-by-token delivery, reducing perceived latency.

---

## Tech Stack

| Layer | Tool |
|---|---|
| PDF Parsing | Docling (structure-preserving markdown) |
| Chunking | Custom heading-aware splitter with table protection |
| Embeddings | BAAI/bge-m3 (1024-dim dense + sparse lexical weights) |
| Vector Store | ChromaDB (persistent, cosine distance) |
| Sparse Store | JSON (persisted to disk, loaded at startup) |
| Reranker | BAAI/bge-reranker-v2-m3 (cross-encoder) |
| LLM | Ollama + Llama 3.2 3B |
| Teaching Agent | LangGraph (stateful multi-turn, five-intent router) |
| Backend | FastAPI |
| UI | Streamlit |
| Evaluation | Custom RAGAS harness (local Llama as judge) |