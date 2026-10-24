from .metadata import ChunkMetadata

def __getattr__(name):
    if name == "DocumentParser":
        from .parser import DocumentParser
        return DocumentParser
    elif name == "GeologicalChunker":
        from .chunker import GeologicalChunker
        return GeologicalChunker
    raise AttributeError(f"module has no attribute {name}")
