"""
RAGAS 评测：Faithfulness / Answer Relevancy / Context Precision / Context Recall 四维指标。
"""
import asyncio
import logging
from typing import Optional

from openai import AsyncOpenAI

from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from src.evaluation.test_set import build_test_dataset
from src.retrieval.query_engine import AdaptiveQueryEngine, QueryResult

logger = logging.getLogger(__name__)


class _BGEM3Embeddings:
    """把项目内的 BGEEmbedder 适配成 Langchain Embeddings 接口，供 RAGAS 使用。"""

    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        self._embedder = None

    @property
    def embedder(self):
        if self._embedder is None:
            from src.vectorization.embedder import BGEEmbedder
            self._embedder = BGEEmbedder()
        return self._embedder

    def embed_documents(self, texts):
        return self.embedder.encode_dense(texts, batch_size=self.batch_size).tolist()

    def embed_query(self, text):
        return self.embedder.encode_dense([text], batch_size=1)[0].tolist()


async def run_evaluation(
    engine: AdaptiveQueryEngine,
    test_cases: Optional[list[dict]] = None,
    sample_size: Optional[int] = None,
) -> dict:
    """
    批量评测 RAG 系统。

    返回:
    {
        "per_case": [{"question":..., "faithfulness":..., ...}, ...],
        "summary": {"avg_faithfulness":..., "avg_answer_relevancy":..., ...},
        "diagnostics": [...]
    }
    """
    if test_cases is None:
        test_cases = build_test_dataset()
    if sample_size:
        test_cases = test_cases[:sample_size]

    # 先跑所有查询获取结果
    eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    results: list[QueryResult] = []

    for i, case in enumerate(test_cases):
        logger.info(f"[{i+1}/{len(test_cases)}] 评测: {case['question'][:40]}...")
        result = await engine.query(case["question"])
        results.append(result)

        eval_data["question"].append(case["question"])
        eval_data["answer"].append(result.answer)
        eval_data["contexts"].append([c["text"] for c in result.source_chunks])
        eval_data["ground_truth"].append(case["ground_truth"])

    # 使用 RAGAS 计算指标
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from datasets import Dataset

        dataset = Dataset.from_dict(eval_data)

        # 配置 RAGAS 使用的 LLM
        from langchain_openai import ChatOpenAI
        ragas_llm = ChatOpenAI(
            model=DEEPSEEK_MODEL,
            openai_api_key=DEEPSEEK_API_KEY,
            openai_api_base=DEEPSEEK_BASE_URL,
            temperature=0,
        )
        evaluator_llm = LangchainLLMWrapper(ragas_llm)

        # answer_relevancy 需要 embedding：用本地 bge-m3，避免依赖 API 的中转站
        evaluator_embeddings = LangchainEmbeddingsWrapper(_BGEM3Embeddings())

        scores = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            show_progress=True,
        )

        df = scores.to_pandas()
        summary = {
            "avg_faithfulness": float(df["faithfulness"].mean()),
            "avg_answer_relevancy": float(df["answer_relevancy"].mean()),
            "avg_context_precision": float(df["context_precision"].mean()),
            "avg_context_recall": float(df["context_recall"].mean()),
        }

        # 诊断
        diagnostics = _run_diagnostics(df)

        return {"per_case": df.to_dict("records"), "summary": summary, "diagnostics": diagnostics}

    except ImportError as e:
        logger.error(f"RAGAS 未安装或依赖缺失: {e}")
        return {"per_case": eval_data, "summary": {}, "diagnostics": ["RAGAS 评测不可用"]}


def _run_diagnostics(df) -> list[str]:
    """基于四维指标交叉分析给出诊断建议"""
    suggestions = []
    cr = df["context_recall"].mean()
    cp = df["context_precision"].mean()
    fa = df["faithfulness"].mean()
    ar = df["answer_relevancy"].mean()

    if cr < 0.70:
        suggestions.append(f"[检索不足] Context Recall={cr:.2f} < 0.70 → 增大 Top-K / 减小 chunk_size / 开启 HyDE")
    if cp < 0.75:
        suggestions.append(f"[噪声过多] Context Precision={cp:.2f} < 0.75 → 加强 Reranker / 元数据过滤 / 混合检索")
    if fa < 0.85:
        suggestions.append(f"[幻觉风险] Faithfulness={fa:.2f} < 0.85 → 加强 Prompt 幻觉抑制 / 降 temperature")
    if ar < 0.80:
        suggestions.append(f"[偏题] Answer Relevancy={ar:.2f} < 0.80 → 去除上下文噪声 / 优化 Query 改写")
    if not suggestions:
        suggestions.append("✓ 所有指标达标，系统表现良好")

    return suggestions
