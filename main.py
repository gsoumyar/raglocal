import tempfile
import os
import json
import requests
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, validator
import fitz
import chromadb
from FlagEmbedding import BGEM3FlagModel


# ── App ──
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Sparse vector persistence ──
SPARSE_STORE_PATH = "./sparse_store.json"

def _load_sparse_store():
    """Load sparse vectors from disk if they exist (survives server restarts)."""
    if os.path.exists(SPARSE_STORE_PATH):
        with open(SPARSE_STORE_PATH, "r") as f:
            return json.load(f)
    return {}

def _save_sparse_store():
    """Persist sparse vectors to disk as JSON."""
    with open(SPARSE_STORE_PATH, "w") as f:
        json.dump(_sparse_store, f)

_sparse_store = _load_sparse_store()

# ── Docling converter (built ONCE at startup) ──
_pipeline_options = PdfPipelineOptions()
_pipeline_options.do_ocr = False
_pipeline_options.do_table_structure = True
_docling_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=_pipeline_options)
    }
)

# ── Embedding model (built ONCE at startup) ──
embedding_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

# ── Reranker model (built ONCE at startup) ──
_rerank_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
_rerank_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3")
_rerank_model.eval()

# ── ChromaDB (cosine distance, NOT default L2) ──
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)


# ══════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════

def extract_txt_pdf(content: bytes) -> str:
    """Extract structured markdown from PDF bytes using Docling.
    Bridge: bytes -> temp file -> Docling convert -> markdown string.
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        result = _docling_converter.convert(tmp_path)
        return result.document.export_to_markdown()
    finally:
        os.remove(tmp_path)


def chunk_markdown(md: str, max_words: int = 250):
    """Structure-aware chunking for Docling markdown.
    - Splits at heading boundaries (each section = one chunk)
    - Keeps tables intact (never splits a | block)
    - Prepends the heading trail so each chunk knows where it lives
    - Returns list of dicts: {"text": ..., "trail": ...}
    - Falls back to word-splitting ONLY if a single section is too big
    """
    lines = md.splitlines()
    chunks = []
    heading_trail = []
    current_lines = []

    def flush():
        if not current_lines:
            return
        body = "\n".join(current_lines).strip()
        if not body:
            return
        trail = " > ".join(heading_trail)

        if len(body.split()) <= max_words:
            text = f"{trail}\n\n{body}" if trail else body
            chunks.append({"text": text, "trail": trail})
        else:
            # Oversized section: separate table blocks from prose, protect tables
            blocks = []
            current_block = []
            in_table = False

            for line in body.splitlines():
                line_is_table = line.strip().startswith("|")
                if line_is_table and not in_table:
                    if current_block:
                        blocks.append(("prose", "\n".join(current_block)))
                        current_block = []
                    in_table = True
                elif not line_is_table and in_table:
                    if current_block:
                        blocks.append(("table", "\n".join(current_block)))
                        current_block = []
                    in_table = False
                current_block.append(line)

            if current_block:
                blocks.append(("table" if in_table else "prose", "\n".join(current_block)))

            for block_type, block_text in blocks:
                if block_type == "table":
                    text = f"{trail}\n\n{block_text}" if trail else block_text
                    chunks.append({"text": text, "trail": trail})
                else:
                    words = block_text.split()
                    if len(words) <= max_words:
                        text = f"{trail}\n\n{block_text}" if trail else block_text
                        chunks.append({"text": text, "trail": trail})
                    else:
                        for i in range(0, len(words), max_words):
                            piece = " ".join(words[i:i + max_words])
                            text = f"{trail}\n\n{piece}" if trail else piece
                            chunks.append({"text": text, "trail": trail})

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            flush()
            current_lines = []
            level = len(stripped) - len(stripped.lstrip("#"))
            title = stripped.lstrip("#").strip()
            heading_trail = heading_trail[:level - 1]
            heading_trail.append(title)
        else:
            current_lines.append(line)

    flush()
    return chunks


def get_smart_context(full_doc: str, chunk_text: str, budget: int = 15000) -> str:
    """Build a smart context window for Contextual Retrieval.
    Includes: document start (title/TOC) + local neighborhood + document end.
    This gives the LLM enough to situate any chunk without exceeding the budget.
    """
    doc_start = full_doc[:5000]
    doc_end = full_doc[-2000:] if len(full_doc) > 7000 else ""

    # Find where this chunk lives in the original document
    search_key = chunk_text[:200] if len(chunk_text) > 200 else chunk_text
    chunk_pos = full_doc.find(search_key)

    local_context = ""
    if chunk_pos >= 0:
        local_start = max(0, chunk_pos - 1500)
        local_end = min(len(full_doc), chunk_pos + len(chunk_text) + 1500)
        local_context = full_doc[local_start:local_end]

    # Assemble within budget
    context = doc_start
    if local_context:
        context += "\n...\n" + local_context
    if doc_end:
        context += "\n...\n" + doc_end

    return context[:budget]


def contextualize_chunk(chunk_text: str, full_doc: str) -> str:
    """Contextual Retrieval: ask local Llama to write a short sentence
    situating this chunk within the full document. Uses smart context window.
    """
    smart_ctx = get_smart_context(full_doc, chunk_text)

    prompt = f"""Here is a document excerpt:
<document>
{smart_ctx}
</document>

Here is a chunk from that document:
<chunk>
{chunk_text}
</chunk>

Write a single sentence (under 30 words) that situates this chunk within
the document — what topic it covers, what section it belongs to, and what
makes it distinct. Return ONLY that sentence, nothing else."""

    try:
        response = requests.post("http://localhost:11434/api/chat", json={
            "model": "llama3.2",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        })
        context_sentence = response.json()["message"]["content"].strip()
        return f"{context_sentence}\n\n{chunk_text}"
    except Exception:
        return chunk_text


def rerank_chunks_with_metadata(question: str, chunks: list, metadatas: list, top_n: int = 2) -> list:
    """Reranker: scores each (question, chunk) pair side by side.
    Returns top_n results with metadata preserved for citations.
    """
    if not chunks:
        return []
    pairs = [[question, chunk] for chunk in chunks]
    with torch.no_grad():
        inputs = _rerank_tokenizer(
            pairs, padding=True, truncation=True,
            return_tensors="pt", max_length=512
        )
        scores = _rerank_model(**inputs).logits.view(-1).float().tolist()

    scored = sorted(zip(scores, chunks, metadatas), key=lambda x: x[0], reverse=True)
    return [{"text": chunk, "metadata": meta} for _, chunk, meta in scored[:top_n]]


# ══════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════

class Question(BaseModel):
    text: str
    @validator('text')
    def mandatory_field(cls, value):
        if not value.strip():
            raise ValueError('Mandatory Field')
        return value.strip()


# ══════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════

@app.get("/ui")
def serve_ui():
    return FileResponse("static/index.html")

@app.get("/")
def home():
    return {"message": "Hello! RAG Project API is alive!"}

@app.get("/about")
def about(language: str = "english"):
    translations = {
        "english": "RAG Project",
        "spanish": "Proyecto RAG",
        "hindi": "RAG परियोजना"
    }
    name = translations.get(language, "RAG project")
    return {"name": name, "version": "2.0"}


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    contextual_retrieval: bool = False
):
    """Upload a PDF, extract with Docling, chunk, embed, and store.
    Drops all existing data first (no incremental complexity).
    Toggle contextual_retrieval=true for LLM-generated chunk context.
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="only PDF files are accepted!")

    # Drop everything and reload — no incremental complexity
    existing = collection.get()
    if existing['ids']:
        collection.delete(ids=existing['ids'])
    _sparse_store.clear()

    content = await file.read()
    full_txt = extract_txt_pdf(content)
    chunks = chunk_markdown(full_txt)

    # Contextual Retrieval (optional, toggle via query param)
    if contextual_retrieval:
        print(f"Running Contextual Retrieval on {len(chunks)} chunks...")
        for i, chunk in enumerate(chunks):
            chunks[i]["text"] = contextualize_chunk(chunk["text"], full_txt)
            if (i + 1) % 10 == 0:
                print(f"  Contextualized {i + 1}/{len(chunks)} chunks")
        print("Contextual Retrieval complete.")

    # Embed and store each chunk with rich metadata
    for i, chunk in enumerate(chunks):
        encoded = embedding_model.encode([chunk["text"]], return_dense=True, return_sparse=True)
        dense_vec = encoded['dense_vecs'][0].tolist()
        sparse_vec = {str(k): float(v) for k, v in encoded['lexical_weights'][0].items()}

        chunk_id = f"{file.filename}_chunk_{i}"

        collection.add(
            documents=[chunk["text"]],
            embeddings=[dense_vec],
            ids=[chunk_id],
            metadatas=[{
                "filename": file.filename,
                "chunk_index": i,
                "section": chunk["trail"],
                "contextual_retrieval": contextual_retrieval
            }]
        )
        _sparse_store[chunk_id] = sparse_vec

    # Persist sparse store to disk (survives server restarts)
    _save_sparse_store()

    return {
        "filename": file.filename,
        "page_count": len(fitz.open(stream=content, filetype="pdf")),
        "word_count": len(full_txt.split()),
        "total_chunks": len(chunks),
        "contextual_retrieval": contextual_retrieval,
        "message": "Document processed and stored successfully!"
    }


@app.post("/ask")
def ask_question(question: Question):
    """Full retrieval pipeline: embed -> dense search -> hybrid scoring ->
    rerank -> local Llama generation -> cited answer.
    """
    # Step 1: embed the question — dense + sparse
    encoded = embedding_model.encode([question.text], return_dense=True, return_sparse=True)
    q_dense = encoded['dense_vecs'][0].tolist()
    q_sparse = {str(k): float(v) for k, v in encoded['lexical_weights'][0].items()}

    # Step 2: dense search — cast a wide net (top 10 candidates)
    results = collection.query(
        query_embeddings=[q_dense],
        n_results=min(10, collection.count())
    )

    # Step 3: handle no results
    if not results['documents'][0]:
        return {
            "question": question.text,
            "answer": "No relevant content found. Please upload a document first.",
            "chunks_used": 0,
            "sources": []
        }

    # Step 4: hybrid scoring — merge dense + sparse signals
    candidate_ids = results['ids'][0]
    dense_distances = results['distances'][0]

    hybrid_scores = []
    for idx, chunk_id in enumerate(candidate_ids):
        dense_score = 1 - dense_distances[idx]     # cosine distance -> similarity

        chunk_sparse = _sparse_store.get(chunk_id, {})
        sparse_score = sum(
            q_sparse.get(token, 0.0) * chunk_sparse.get(token, 0.0)
            for token in set(q_sparse.keys()) & set(chunk_sparse.keys())
        )

        final_score = (0.7 * dense_score) + (0.3 * sparse_score)
        hybrid_scores.append((idx, final_score))

    # Step 5: take top 5 from hybrid for reranking
    hybrid_scores.sort(key=lambda x: x[1], reverse=True)
    top_hybrid_indices = [s[0] for s in hybrid_scores[:5]]
    top_hybrid_chunks = [results['documents'][0][i] for i in top_hybrid_indices]
    top_hybrid_metadata = [results['metadatas'][0][i] for i in top_hybrid_indices]

    # Step 6: rerank — model reads question + each chunk side by side
    reranked = rerank_chunks_with_metadata(question.text, top_hybrid_chunks, top_hybrid_metadata, top_n=2)
    relevant_chunks = [r["text"] for r in reranked]
    sources = [
        {"filename": r["metadata"]["filename"], "section": r["metadata"].get("section", "")}
        for r in reranked
    ]
    context = "\n\n".join(relevant_chunks)

    # Step 7: send to local Llama with citation instruction
    source_labels = "\n".join(
        f"[Source {i+1}: {s['filename']} — {s['section']}]"
        for i, s in enumerate(sources) if s['section']
    )

    response = requests.post("http://localhost:11434/api/chat", json={
        "model": "llama3.2",
        "messages": [
            {
                "role": "system",
                "content": f"""You are a helpful study assistant. Answer the question based only on the provided context.
If the answer is not in the context, say 'I could not find this in the document.'
Always end your answer by citing which section the information came from.

Available sources:
{source_labels}"""
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question.text}"
            }
        ],
        "stream": False
    })
    answer = response.json()["message"]["content"]

    return {
        "question": question.text,
        "answer": answer,
        "chunks_used": len(relevant_chunks),
        "sources": sources,
        "retrieved_context": relevant_chunks     # exposed for evaluation harness
    }


@app.get("/inspect")
def inspect_database():
    """Return metadata about all stored chunks: filenames, sections, IDs."""
    all_data = collection.get()
    filenames = list(set(
        [m['filename'] for m in all_data['metadatas']]
    )) if all_data['metadatas'] else []

    sections = list(set(
        [m.get('section', '') for m in all_data['metadatas']]
    )) if all_data['metadatas'] else []

    return {
        "total_chunks": len(all_data['ids']),
        "documents_loaded": filenames,
        "sections": sections,
        "chunk_ids": all_data['ids'],
        "metadata": all_data['metadatas']
    }


@app.delete("/clear")
def clear_database():
    """Wipe all data: ChromaDB embeddings + sparse store on disk."""
    existing = collection.get()
    if existing['ids']:
        collection.delete(ids=existing['ids'])
    _sparse_store.clear()
    _save_sparse_store()
    return {"message": "Database cleared successfully!"}