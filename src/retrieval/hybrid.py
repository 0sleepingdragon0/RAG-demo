"""
混合检索器：整合向量检索（dense）+ 图谱检索（LightRAG/Neo4j），RRF 融合后 Rerank 精排。
"""
import logging
from typing import Optional

from src.vectorization.embedder import BGEEmbedder
from src.vectorization.store import VectorStore
from src.vectorization.reranker import Reranker
from src.retrieval.rrf import rrf_fusion

from config.settings import SIMILARITY_TOP_K, RERANK_TOP_N

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    三路混合检索：
    1. Dense 向量检索（bge-m3 + Chroma）
    2. Sparse 关键词匹配（bge-m3 sparse，Demo中作为辅助）
    3. 图谱检索（通过 LightRAG / Neo4j）

    融合：RRF 合并 → Reranker 精排 → Top-N 输出。
    """

    def __init__(
        self,
        embedder: BGEEmbedder,
        vector_store: VectorStore,
        reranker: Reranker,
        graph_retriever=None,  # LightRAG 实例，可选
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.reranker = reranker
        self.graph_retriever = graph_retriever

    def retrieve(
        self,
        query: str,
        top_k: int = SIMILARITY_TOP_K,
        use_graph: bool = True,
        metadata_filter: Optional[dict] = None,
    ) -> list[dict]:
        """
        执行混合检索，返回精排后的 top-N 文档块。
        每个结果含: id, text, metadata, score, rerank_score, rrf_score
        """
        ranked_lists = []

        # === 1. Dense 向量检索 ===
        query_vec = self.embedder.encode_query(query)
        dense_results = self.vector_store.search(
            query_embedding=query_vec,
            top_k=top_k,
            where=metadata_filter,
        )
        ranked_lists.append(dense_results)
        logger.debug(f"Dense 检索: {len(dense_results)} 条")

        # === 2. 图谱检索 ===
        if use_graph and self.graph_retriever:
            graph_results = self._graph_retrieve(query, top_k=top_k)
            if graph_results:
                ranked_lists.append(graph_results)
                logger.debug(f"图谱检索: {len(graph_results)} 条")

        # === 3. RRF 融合 ===
        if len(ranked_lists) > 1:
            fused = rrf_fusion(ranked_lists)
        else:
            fused = ranked_lists[0]

        # === 4. Reranker 精排 ===
        reranked = self.reranker.rerank(query, fused, top_n=RERANK_TOP_N)

        logger.info(f"混合检索完成: 融合 {len(fused)} → 精排 {len(reranked)}")
        return reranked

    def _graph_retrieve(self, query: str, top_k: int) -> list[dict]:
        """
        通过 LightRAG 执行图谱检索。
        返回与文档格式兼容的结果列表。
        """
        try:
            # LightRAG 的 query 返回自然语言答案 + 引用上下文
            import asyncio
            from lightrag import QueryParam

            result = asyncio.run(
                self.graph_retriever.aquery(
                    query,
                    param=QueryParam(mode="hybrid", top_k=top_k)
                )
            )
            # 将图谱结果转为标准文档格式
            if isinstance(result, str) and result.strip():
                return [{
                    "id": f"graph_{hash(query) % 10000:04d}",
                    "text": result,
                    "metadata": {"source": "知识图谱", "doc_type": "graph_context"},
                    "score": 0.8,
                }]
            return []
        except Exception as e:
            logger.warning(f"图谱检索失败（降级跳过）: {e}")
            return []
