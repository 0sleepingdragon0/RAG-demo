from .schema import Entity, Relation, Triple, GeoSchema, EntityLabel, RelationType

def __getattr__(name):
    """Lazy import: 重依赖模块按需加载"""
    if name == "TripleExtractor":
        from .extractor import TripleExtractor
        return TripleExtractor
    elif name == "Neo4jStore":
        from .neo4j_store import Neo4jStore
        return Neo4jStore
    raise AttributeError(f"module has no attribute {name}")
