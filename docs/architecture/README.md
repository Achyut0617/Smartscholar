# Architecture

SmartScholar is a Retrieval-Augmented Generation (RAG) application. Ingestion and querying are separate flows that share a Qdrant collection.

## Ingestion flow

`core/preprocessing_pipeline.py` lists files in a docset folder and skips any file whose name already has chunks in the collection. For each new file, `document_processing/files_processor.py` extracts text (`.txt`, `.docx` or `.pdf`), and `document_processing/chunker.py` splits it into chunks with metadata (document id, chunk order, token count, SHA-256 hash). `embeddings/embedder.py` adds an embedding to each chunk. `storage/vector_store.py` writes the points to Qdrant.

The collection name is the basename of the docset folder. Each point's payload holds `document_id`, `chunk_order`, `text`, `token_count` and `chunk_hash`.

## Query flow

`core/rag_pipeline.py` (`RAGPipeline.query`) does the following:

1. Calls `query_processing/query_classifier.py`. This currently always returns `"general"`.
2. Calls `query_processing/query_analyzer.py` (`FileNameExtractor`) to read any `[[file]]` tags.
3. If tags are present, retrieves all chunks for those files with `retrieval/retriever.py`. Otherwise it runs a semantic search over the collection with a 0.6 score threshold.
4. If no chunks are found, calls `generation/generator2.py` without retrieval.
5. Otherwise `generation/augmenter.py` re-ranks the chunks with a cross-encoder and builds a context of at most 4096 tokens.
6. `generation/generator2.py` sends the context and sources to Ollama (`phi`) and returns `{"query", "answer", "model"}` or `{"query", "error"}`.

`app/main.py` and `app/main_gui.py` store each exchange with `sessions/session_manager.py`, which writes to SQLite.

## Module responsibilities

| Package | Modules | Responsibility |
|---|---|---|
| `app` | `main.py`, `main_gui.py` | Command-line and GUI entry points |
| `core` | `rag_pipeline.py`, `preprocessing_pipeline.py` | Orchestration of query and ingestion |
| `document_processing` | `files_processor.py`, `chunker.py`, `folder_monitor.py` | Text extraction, chunking, folder watching (unused) |
| `embeddings` | `embedder.py` | BGE model loading and embedding |
| `retrieval` | `retriever.py` | Semantic and file-based retrieval |
| `storage` | `vector_store.py` | Qdrant collection, insert, search and file lookup |
| `generation` | `augmenter.py`, `generator2.py` (active), `generator.py` (inactive Gemini) | Re-ranking, context building and LLM calls |
| `query_processing` | `query_analyzer.py`, `query_classifier.py`, `query_filetags_extractor.py` | `[[file]]` tag parsing and the disabled classifier |
| `sessions` | `session_manager.py` | SQLite sessions and chat history |
| `config.py` | | Project-root-relative paths |

## Persistence

| Artifact | Location | Notes |
|---|---|---|
| Qdrant database | `qdrant_vdb/` | Local, file-locked. Only one process can open it at a time. |
| Model weights | `local/` | Downloaded on first use. Git-ignored. |
| Sessions | `sessions.db` | SQLite. Preserve it. |
| Query classifier | `local/query_classifier_model.pkl` | Written by the trainer. Not loaded at runtime. |
