"""Lazy imports for vectorization (heavy deps: FlagEmbedding, chromadb)"""

def __getattr__(name):
    if name == "BGEEmbedder":
        from .embedder import BGEEmbedder
        return BGEEmbedder
    elif name == "VectorStore":
        from .store import VectorStore
        return VectorStore
    elif name == "Reranker":
        from .reranker import Reranker
        return Reranker
    raise AttributeError(f"module has no attribute {name}")
