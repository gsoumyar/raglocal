# Local-First RAG Learning Assistant

A fully offline document question-answering system. Upload a PDF textbook, ask questions, get cited answers grounded in the source material. Nothing leaves your machine.

Every layer uses best-in-class methods: structure-preserving parsing (Docling), heading-aware chunking with table protection, hybrid dense+sparse retrieval (BGE-M3), cross-encoder reranking, and local LLM generation (Ollama + Llama 3.2). Every answer cites the exact section it came from.

---

## Demo

```
>> Upload: tools.pdf (16 pages)
   32 chunks extracted, 5468 words

>> Question: "What is a Pozidriv screwdriver?"

   A Pozidriv screwdriver is similar to Phillips but with additional
   ribs between the arms. Common in European-made furniture and equipment.

   Source: tools.pdf, Section 1.2 Fastening and Turning Tools
```

The system never guesses. If the answer is not in the document, it says so.

---

## Architecture

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

Zero hallucination across all configurations. Contextual Retrieval improved relevancy and correctness with no loss in faithfulness. The toggle exists because CR adds ingestion time and the benefit varies by document structure.

---

## Getting Started

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed with Llama 3.2 pulled: `ollama pull llama3.2`
- 8GB+ RAM (tested on M2 MacBook Air, 8GB)

### Install

```bash
git clone https://github.com/gsoumyar/rag-document-assistant-langchain.git
cd rag-document-assistant-langchain
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
# Ensure Ollama is running (open the app or run: ollama serve)
uvicorn main:app
```

> **Note:** Do not use `--reload`. Docling and HuggingFace write cache files into `.venv` during model loading, which triggers WatchFiles to restart the server in a loop. Use plain `uvicorn main:app`.

### Usage

1. Open `http://127.0.0.1:8000/docs` (Swagger UI)
2. **POST /upload** with a PDF file. Toggle `contextual_retrieval=true` for richer chunk embeddings (slower ingestion, better retrieval on documents with vague headings).
3. **POST /ask** with a question. Response includes the answer, source citations, and raw retrieved context.
4. **GET /inspect** to view all stored chunks and their section metadata.
5. **DELETE /clear** to wipe all stored data.

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
| POST | `/ask` | Ask a question. Returns: answer, sources, retrieved_context |
| GET | `/inspect` | List all stored chunks, sections, and metadata |
| DELETE | `/clear` | Wipe ChromaDB and sparse store |
| GET | `/` | Health check |
| GET | `/about` | Project info |
| GET | `/ui` | Simple web UI |

---

## Project Structure

```
rag-document-assistant-langchain/
    main.py                 FastAPI app, full RAG pipeline
    golden_test.json        10 question-answer pairs for evaluation
    eval_ragas.py           RAGAS evaluation harness
    requirements.txt        Python dependencies
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

- **Text-layer PDFs only.** OCR is disabled. Scanned documents produce empty output. OCR can be enabled via config but adds significant processing time and memory.

- **Same model as judge.** RAGAS evaluation uses Llama 3.2 as both generator and judge. Relative comparisons between configurations are reliable; absolute scores are softer than they would be with an independent stronger judge.

- **In-memory sparse store.** Sparse vectors load into memory at startup and persist to JSON on upload. Scales fine for single-document use; a sparse-capable vector store (Qdrant) would be needed for thousands of documents.

- **No streaming.** Answers generate in one shot. Long answers on slower hardware have a noticeable wait.

- **8GB RAM ceiling.** BGE-M3 (~2.2GB) + reranker (~1.1GB) + Llama 3.2 (~2GB) coexist in memory. Tested stable on M2 8GB with documents up to 1300 pages, but headroom is limited.

---

## Roadmap

### Next

**Agentic Teaching Flow (LangGraph).** Replace the single question-answer loop with a multi-step teaching agent:
- Assess what the learner already knows
- Retrieve relevant content
- Plan an explanation strategy (analogy-first, definition-first, example-first)
- Generate the explanation
- Check understanding ("can you explain this back to me?")
- Re-explain with a different angle if the learner is stuck
- Advance to the next concept when they've got it

This is the difference between a search engine and a tutor.

**Streamlit UI.** File upload with progress bar, chat-style Q&A, visible citations with section references, toggle controls. Built after the teaching agent since the agent changes what the interface needs to show.

### Later

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
| Backend | FastAPI |
| Evaluation | Custom RAGAS harness (local Llama as judge) |
