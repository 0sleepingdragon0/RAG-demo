"""
bge-reranker-v2-m3 精排模型封装。
"""
import logging
from typing import Optional

from config.settings import RERANKER_MODEL_NAME, RERANK_TOP_N

logger = logging.getLogger(__name__)


class Reranker:
    """
    Cross-encoder Reranker：对粗召回结果做精排。

    用法：
        reranker = Reranker()
        reranked = reranker.rerank(query, candidate_docs, top_n=5)
    """

    def __init__(self, model_name: str = RERANKER_MODEL_NAME, top_n: int = RERANK_TOP_N):
        self.model_name = model_name
        self.top_n = top_n
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from FlagEmbedding import FlagReranker
            logger.info(f"加载 Reranker: {self.model_name}")
            self._model = FlagReranker(self.model_name, use_fp16=True)
            logger.info("Reranker 加载完成")
        return self._model

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_n: Optional[int] = None,
        text_key: str = "text",
    ) -> list[dict]:
        """
        对候选文档列表精排。

        参数：
        - query: 用户查询
        - documents: [{"text": ..., "score": ..., ...}, ...]
        - top_n: 返回前 N 个

        返回：重排后的 documents 子集，附带 rerank_score 字段。
        """
        if not documents:
            return []

        top_n = top_n or self.top_n
        texts = [d[text_key] for d in documents]

        # 构造 pairs
        pairs = [[query, t] for t in texts]
        scores = self.model.compute_score(pairs, normalize=True)

        # 合并分数并排序
        for d, s in zip(documents, scores):
            d["rerank_score"] = float(s)

        reranked = sorted(documents, key=lambda d: d["rerank_score"], reverse=True)
        return reranked[:top_n]
