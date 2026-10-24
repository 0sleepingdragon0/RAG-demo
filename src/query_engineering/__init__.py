"""Lazy imports for query_engineering (needs openai)"""

def __getattr__(name):
    if name == "multi_query_rewrite":
        from .multi_query import multi_query_rewrite
        return multi_query_rewrite
    elif name == "hyde_transform":
        from .hyde import hyde_transform
        return hyde_transform
    elif name == "step_back_transform":
        from .step_back import step_back_transform
        return step_back_transform
    elif name == "decompose_query":
        from .decompose import decompose_query
        return decompose_query
    raise AttributeError(f"module has no attribute {name}")
