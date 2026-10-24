"""
bge-m3 Embedding 封装：支持 dense + sparse 双模式编码。
"""
import logging
import numpy as np
from typing import Optional

from config.settings import EMBED_MODEL_NAME, EMBED_USE_FP16

logger = logging.getLogger(__name__)


class BGEEmbedder:
    """
    基于 FlagEmbedding 的 bge-m3 编码封装。

    输出：
    - dense_vecs: (N, 1024) float32 数组，用于语义相似度检索
    - sparse_vecs: dict[int, float] per doc，用于关键词精确匹配（lexical）
    """

    def __init__(self, model_name: str = EMBED_MODEL_NAME, use_fp16: bool = EMBED_USE_FP16):
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel
            logger.info(f"加载 Embedding 模型: {self.model_name}")
            self._model = BGEM3FlagModel(self.model_name, use_fp16=self.use_fp16)
            logger.info("模型加载完成")
        return self._model

    def encode_dense(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """编码为 dense 向量 (N, 1024)"""
        outputs = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=8192,
            return_dense=True,
            return_sparse=False,
        )
        dense_vecs = outputs["dense_vecs"]
        logger.info(f"Dense 编码完成: {dense_vecs.shape}")
        return dense_vecs.astype(np.float32)

    def encode_sparse(self, texts: list[str], batch_size: int = 32) -> list[dict]:
        """编码为 sparse 向量（词表形式，适合 BM25-like 检索）"""
        outputs = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=8192,
            return_dense=False,
            return_sparse=True,
        )
        sparse_vecs = outputs["lexical_weights"]
        # 转为 list[dict[token_id: float]]
        result = []
        for sv in sparse_vecs:
            result.append({int(k): float(v) for k, v in sv.items() if v > 0.01})
        logger.info(f"Sparse 编码完成: {len(result)} docs")
        return result

    def encode_both(self, texts: list[str], batch_size: int = 32) -> tuple[np.ndarray, list[dict]]:
        """同时获取 dense + sparse"""
        outputs = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=8192,
            return_dense=True,
            return_sparse=True,
        )
        dense = outputs["dense_vecs"].astype(np.float32)
        sparse = [{int(k): float(v) for k, v in sv.items() if v > 0.01} for sv in outputs["lexical_weights"]]
        return dense, sparse

    def encode_query(self, query: str) -> np.ndarray:
        """编码查询文本（加 instruction 前缀提升检索精度）"""
        # bge-m3 推荐检索查询加 instruction
        instruct_query = f"为地质文本检索目的生成向量表示：{query}"
        return self.encode_dense([instruct_query])[0]
