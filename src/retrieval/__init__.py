from .rrf import rrf_fusion

def __getattr__(name):
    if name == "HybridRetriever":
        from .hybrid import HybridRetriever
        return HybridRetriever
    elif name == "AdaptiveQueryEngine":
        from .query_engine import AdaptiveQueryEngine
        return AdaptiveQueryEngine
    raise AttributeError(f"module has no attribute {name}")
