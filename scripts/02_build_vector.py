"""
scripts/02_build_vector.py — 将 chunks 向量化并写入 Chroma
用法: python scripts/02_build_vector.py
"""
import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import *
from src.vectorization.embedder import BGEEmbedder
from src.vectorization.store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("pipeline.vector")


def main():
    # 1. 加载 chunks
    chunks_file = CHUNKS_DIR / "all_chunks.json"
    if not chunks_file.exists():
        logger.error(f"未找到 {chunks_file}，请先运行 01_parse_docs.py")
        return

    chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
    logger.info(f"加载 {len(chunks)} 个文本块")

    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # 2. Embedding
    embedder = BGEEmbedder()
    logger.info("开始批量编码...")
    embeddings = embedder.encode_dense(texts, batch_size=16)
    logger.info(f"编码完成: shape={embeddings.shape}")

    # 3. 写入 Chroma
    store = VectorStore()
    store.delete_all()  # 重建
    store.add_documents(texts=texts, embeddings=embeddings, ids=ids, metadatas=metadatas)

    logger.info(f"向量库构建完成: {store.count} 条向量")
    logger.info(f"存储位置: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
