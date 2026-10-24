"""
scripts/04_test_retrieval.py — 交互式测试检索质量
用法: python scripts/04_test_retrieval.py
"""
import sys
import json
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import *
from src.vectorization.embedder import BGEEmbedder
from src.vectorization.store import VectorStore
from src.vectorization.reranker import Reranker
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.query_engine import AdaptiveQueryEngine


def print_separator():
    print("\n" + "=" * 70)


async def main():
    # 初始化引擎
    embedder = BGEEmbedder()
    store = VectorStore()
    reranker = Reranker()
    retriever = HybridRetriever(embedder=embedder, vector_store=store, reranker=reranker)
    engine = AdaptiveQueryEngine(retriever)

    print_separator()
    print("地质大模型 — 交互式检索测试")
    print("输入问题回车查询，输入 q 退出，输入 s 显示统计")
    print_separator()

    while True:
        q = input("\n> 问题: ").strip()
        if q.lower() == "q":
            break
        if q.lower() == "s":
            print(f"  向量库: {store.count} 条")
            continue
        if not q:
            continue

        result = await engine.query(q)

        print(f"\n  路由: {result.route}")
        if len(result.sub_queries) > 1:
            print(f"  变换: {result.sub_queries}")
        print(f"\n  答案:\n  {result.answer[:800]}")
        print(f"\n  来源 ({len(result.source_chunks)} 条):")
        for i, c in enumerate(result.source_chunks[:3], 1):
            meta = c.get("metadata", {})
            score = c.get("rerank_score", c.get("score", 0))
            print(f"    [{i}] {meta.get('source','?')} (score={score:.3f}): {c['text'][:100]}...")


if __name__ == "__main__":
    asyncio.run(main())
