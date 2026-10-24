"""
scripts/03_build_kg.py — 从 chunks 中抽取三元组，写入 Neo4j
用法: python scripts/03_build_kg.py [--batch N] [--limit M]

前置条件：
1. Neo4j 已启动 (docker-compose up -d)
2. .env 中配置了 DEEPSEEK_API_KEY
"""
import sys
import json
import asyncio
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import *
from src.knowledge_graph.extractor import TripleExtractor
from src.knowledge_graph.neo4j_store import Neo4jStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("pipeline.kg")


async def main(batch_size: int = 5, limit: int = 0):
    # 1. 加载 chunks
    chunks_file = CHUNKS_DIR / "all_chunks.json"
    if not chunks_file.exists():
        logger.error(f"未找到 {chunks_file}，请先运行 01_parse_docs.py")
        return

    chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
    if limit:
        chunks = chunks[:limit]

    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    logger.info(f"准备抽取 {len(texts)} 个文本块的三元组")

    # 2. 抽取
    extractor = TripleExtractor()
    store = Neo4jStore()
    store.init_schema()

    total_triples = 0
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]

        logger.info(f"处理批次 {i // batch_size + 1}: chunks {i+1}-{min(i+batch_size, len(texts))}")
        batch_results = await extractor.extract_batch(batch_texts, concurrency=3)

        for chunk_id, triples in zip(batch_ids, batch_results):
            if triples:
                store.upsert_triples(triples, source_id=chunk_id)
                total_triples += len(triples)
                low_conf = [t for t in triples if t.confidence < 0.7]
                if low_conf:
                    logger.warning(f"  {chunk_id}: {len(low_conf)} 条低置信度三元组需人工校验")

        logger.info(f"  累计三元组: {total_triples}")

    # 3. 统计
    stats = store.get_stats()
    logger.info(f"知识图谱构建完成: {stats['nodes']} 节点, {stats['relationships']} 关系")
    store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=5, help="每批处理的chunk数")
    parser.add_argument("--limit", type=int, default=0, help="限制处理总数（调试用）")
    args = parser.parse_args()
    asyncio.run(main(batch_size=args.batch, limit=args.limit))
