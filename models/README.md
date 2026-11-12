# Models

Model weights are not stored in this repository. They are large, machine-specific, and downloaded on first use.

| Model | Used by | Downloaded to |
|---|---|---|
| `BAAI/bge-base-en-v1.5` (sentence embeddings, 768-dim) | `smartscholar/embeddings/embedder.py` | `local/bge-base-en-v1.5/` |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` (re-ranker) | `smartscholar/generation/augmenter.py` | `local/cross-encoder/` |
| `phi` (generator, served by Ollama) | `smartscholar/generation/generator2.py` | Managed by Ollama (`ollama pull phi`) |
| `bert-base-uncased` (classifier training only) | `scripts/training/train_query_classifier.py` | `local/` |

The `local/` folder is git-ignored. Set `SMARTSCHOLAR_MODELS_DIR` to store weights elsewhere.

Moving model storage into this folder is a possible future change. It would require updating `src/smartscholar/config.py`.
