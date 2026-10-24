"""
Chroma 向量存储封装：支持 dense 检索 + metadata 过滤。
"""
import logging
from pathlib import Path
from typing import Optional

import chromadb
import numpy as np

from config.settings import CHROMA_DIR, CHROMA_COLLECTION, EMBED_DIM

logger = logging.getLogger(__name__)


class VectorStore:
    """
    基于 Chroma 的向量存储。

    功能：
    - add_documents: 批量插入文本 + embedding + metadata
    - search: 向量相似度检索（cosine），支持 metadata 过滤
    - delete_collection / reset
    """

    def __init__(self, persist_dir: Optional[Path] = None, collection_name: str = CHROMA_COLLECTION):
        self.persist_dir = persist_dir or CHROMA_DIR
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine", "embed_dim": EMBED_DIM},
        )
        logger.info(f"Chroma 就绪: {self.persist_dir}, 已有 {self.collection.count()} 条向量")

    def add_documents(
        self,
        texts: list[str],
        embeddings: np.ndarray,
        ids: list[str],
        metadatas: list[dict],
    ):
        """批量添加文档块"""
        if len(texts) == 0:
            return

        # Chroma 接受 list[list[float]]
        emb_list = embeddings.tolist()

        # 分批插入（Chroma 单次有大小限制）
        batch_size = 500
        for i in range(0, len(texts), batch_size):
            end = min(i + batch_size, len(texts))
            self.collection.add(
                documents=texts[i:end],
                embeddings=emb_list[i:end],
                ids=ids[i:end],
                metadatas=metadatas[i:end],
            )
        logger.info(f"插入 {len(texts)} 条向量，当前总量: {self.collection.count()}")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 20,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """
        向量检索，返回 [{"id":..., "text":..., "score":..., "metadata":...}]
        """
        kwargs = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        result = self.collection.query(**kwargs)

        docs = []
        for i in range(len(result["ids"][0])):
            docs.append({
                "id": result["ids"][0][i],
                "text": result["documents"][0][i] if result["documents"] else "",
                "metadata": result["metadatas"][0][i] if result["metadatas"] else {},
                # cosine distance → similarity score
                "score": 1.0 - result["distances"][0][i],
            })
        return docs

    def delete_all(self):
        """清空 collection（重新构建时使用）"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine", "embed_dim": EMBED_DIM},
        )
        logger.info("向量库已清空")

    @property
    def count(self) -> int:
        return self.collection.count()
