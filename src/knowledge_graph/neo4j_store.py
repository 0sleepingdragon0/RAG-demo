"""
Neo4j 知识图谱存储：负责批量写入三元组和 Cypher 查询。
"""
import logging
from typing import Optional

from neo4j import GraphDatabase

from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from src.knowledge_graph.schema import Triple, Entity, RelationType

logger = logging.getLogger(__name__)


class Neo4jStore:
    """
    Neo4j Community 存储封装。

    节点：:Deposit {node_id, name, ...properties}
    关系：-[:OCCURS_IN {confidence, evidence}]->
    """

    def __init__(self, uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info(f"Neo4j 连接成功: {uri}")

    def close(self):
        self.driver.close()

    # === 建库 ===
    def init_schema(self):
        """创建索引和约束（幂等）"""
        with self.driver.session() as session:
            # 全局 node_id 唯一约束
            session.run("""
                CREATE CONSTRAINT entity_node_id IF NOT EXISTS
                FOR (e:EntityBase) REQUIRE e.node_id IS UNIQUE
            """)
            # 各类型 name 索引
            from src.knowledge_graph.schema import EntityLabel
            for label in EntityLabel:
                session.run(f"""
                    CREATE INDEX {label.value}_name_idx IF NOT EXISTS
                    FOR (e:{label.value}) ON (e.name)
                """)
        logger.info("Neo4j Schema 初始化完成")

    # === 写入 ===
    def upsert_triples(self, triples: list[Triple], source_id: str = ""):
        """
        批量写入三元组。使用 MERGE 避免重复节点/关系。
        source_id 用于增量溯源（哪个 chunk 贡献了这些三元组）。
        """
        if not triples:
            return

        with self.driver.session() as session:
            for triple in triples:
                # MERGE 节点
                self._merge_node(session, triple.head, source_id)
                self._merge_node(session, triple.tail, source_id)

                # MERGE 关系
                rel_type = triple.relation.type.value
                params = {
                    "head_id": triple.head.node_id,
                    "tail_id": triple.tail.node_id,
                    "props": {
                        **triple.relation.properties,
                        "confidence": triple.confidence,
                        "evidence": triple.evidence,
                        "source_id": source_id,
                    },
                }
                query = f"""
                    MATCH (h:EntityBase {{node_id: $head_id}})
                    MATCH (t:EntityBase {{node_id: $tail_id}})
                    MERGE (h)-[r:{rel_type}]->(t)
                    SET r += $props
                """
                session.run(query, **params)

            logger.info(f"写入 {len(triples)} 条三元组")

    def _merge_node(self, session, entity: Entity, source_id: str):
        """创建或更新节点"""
        props = {
            "node_id": entity.node_id,
            "name": entity.name,
            "label": entity.label.value,
            "source_id": source_id,
            **entity.properties,
        }
        # 确保 node_id 上也有 EntityBase 标记，方便索引和查询
        query = f"""
            MERGE (e:{entity.label.value} {{node_id: $node_id}})
            ON CREATE SET e += $props, e:EntityBase
            ON MATCH SET e += $props
        """
        session.run(query, **props)

    # === 查询 ===
    def find_entity(self, name: str) -> list[dict]:
        """按名称模糊查找实体"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:EntityBase)
                WHERE e.name CONTAINS $name
                RETURN e.name AS name, e.label AS label,
                       e.node_id AS node_id, properties(e) AS props
                LIMIT 20
            """, name=name)
            return [dict(r) for r in result]

    def get_neighbors(self, entity_name: str, depth: int = 2) -> dict:
        """
        获取实体的 N-hop 邻居子图。
        返回 {"nodes": [...], "edges": [...]}
        """
        with self.driver.session() as session:
            query = f"""
                MATCH path = (e:EntityBase)-[r*1..{depth}]-(other:EntityBase)
                WHERE e.name CONTAINS $name
                WITH relationships(path) AS rels, nodes(path) AS ns
                UNWIND rels AS rel
                UNWIND ns AS node
                WITH collect(DISTINCT node) AS nodes, collect(DISTINCT rel) AS rels
                RETURN [n IN nodes | {{name: n.name, label: n.label}}] AS nodes,
                       [r IN rels | {{
                           from: startNode(r).name, to: endNode(r).name, type: type(r)
                       }}] AS edges
            """
            result = session.run(query, name=entity_name)
            record = result.single()
            if not record:
                return {"nodes": [], "edges": []}
            return {"nodes": record["nodes"], "edges": record["edges"]}

    def get_stats(self) -> dict:
        """统计：节点数、关系数"""
        with self.driver.session() as session:
            node_count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
            return {"nodes": node_count, "relationships": rel_count}
