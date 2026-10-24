"""
scripts/05_run_eval.py — 运行 RAGAS 评测
用法: python scripts/05_run_eval.py [--sample N]
"""
import sys
import json
import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import *
from src.vectorization.embedder import BGEEmbedder
from src.vectorization.store import VectorStore
from src.vectorization.reranker import Reranker
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.query_engine import AdaptiveQueryEngine
from src.evaluation.ragas_eval import run_evaluation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("eval")


async def main(sample_size: int = 0):
    # 初始化
    embedder = BGEEmbedder()
    store = VectorStore()
    reranker = Reranker()
    retriever = HybridRetriever(embedder=embedder, vector_store=store, reranker=reranker)
    engine = AdaptiveQueryEngine(retriever)

    logger.info("开始评测...")
    results = await run_evaluation(engine, sample_size=sample_size or None)

    # 输出
    print("\n" + "=" * 60)
    print("RAGAS 评测结果")
    print("=" * 60)

    summary = results.get("summary", {})
    if summary:
        print(f"  Faithfulness:       {summary.get('avg_faithfulness', 'N/A'):.3f}  (目标 ≥0.85)")
        print(f"  Answer Relevancy:   {summary.get('avg_answer_relevancy', 'N/A'):.3f}  (目标 ≥0.80)")
        print(f"  Context Precision:  {summary.get('avg_context_precision', 'N/A'):.3f}  (目标 ≥0.75)")
        print(f"  Context Recall:     {summary.get('avg_context_recall', 'N/A'):.3f}  (目标 ≥0.70)")
    else:
        print("  (RAGAS 指标计算失败，请检查依赖)")

    print("\n诊断建议:")
    for d in results.get("diagnostics", []):
        print(f"  • {d}")

    # 保存详细结果
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = OUTPUT_DIR / f"eval_{ts}.json"
    out_file.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n详细结果已保存: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0, help="只评测前 N 条（调试用）")
    args = parser.parse_args()
    asyncio.run(main(sample_size=args.sample))
