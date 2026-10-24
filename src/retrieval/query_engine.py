"""
自适应查询引擎：根据问题复杂度路由到不同 Query 工程策略 + 检索路径。
"""
import json
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

from openai import AsyncOpenAI

from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from src.retrieval.hybrid import HybridRetriever
from src.query_engineering.multi_query import multi_query_rewrite
from src.query_engineering.hyde import hyde_transform
from src.query_engineering.step_back import step_back_transform
from src.query_engineering.decompose import decompose_query
from src.prompt.query_rewrite import ROUTING_PROMPT
from src.prompt.generation import ANSWER_GENERATION_SYSTEM, build_answer_prompt

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """最终问答结果"""
    question: str
    answer: str
    route: str = ""
    source_chunks: list[dict] = field(default_factory=list)
    sub_queries: list[str] = field(default_factory=list)


class AdaptiveQueryEngine:
    """
    端到端问答引擎：
    1. LLM 路由判断 → 选择最优策略
    2. Query 工程变换
    3. 混合检索 + RRF + Rerank
    4. LLM 生成答案（幻觉抑制 + 来源标注）
    """

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    async def query(self, question: str) -> QueryResult:
        """主入口"""
        # Step 1: 路由判断
        route_info = await self._route(question)
        route = route_info["route"]
        logger.info(f"路由: '{question[:30]}...' → {route}")

        # Step 2: Query 变换
        search_queries = await self._transform(question, route)

        # Step 3: 检索（多query时取各路结果并融合）
        all_contexts = []
        for q in search_queries:
            ctx = self.retriever.retrieve(q, use_graph=(route == "graph"))
            all_contexts.extend(ctx)

        # 去重 + 截取前 10 个
        seen_ids = set()
        unique_ctx = []
        for c in all_contexts:
            if c["id"] not in seen_ids:
                seen_ids.add(c["id"])
                unique_ctx.append(c)
        unique_ctx = unique_ctx[:10]

        # Step 4: 生成答案
        answer = await self._generate(question, unique_ctx)

        return QueryResult(
            question=question,
            answer=answer,
            route=route,
            source_chunks=unique_ctx,
            sub_queries=search_queries,
        )

    async def _route(self, question: str) -> dict:
        """LLM 路由判断"""
        prompt = ROUTING_PROMPT.format(question=question)
        try:
            resp = await self.client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"},
                max_tokens=200,
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            logger.warning(f"路由失败: {e}, 默认 direct")
            return {"route": "direct", "reason": "fallback"}

    async def _transform(self, question: str, route: str) -> list[str]:
        """按路由执行 Query 变换"""
        if route == "multi":
            return await multi_query_rewrite(question, client=self.client)
        elif route == "hyde":
            return await hyde_transform(question, client=self.client)
        elif route == "decompose":
            return await decompose_query(question, client=self.client)
        elif route == "graph":
            # step-back + 原始 query
            abstract_q, orig_q = await step_back_transform(question, client=self.client)
            return [orig_q, abstract_q]
        else:
            # direct
            return [question]

    async def _generate(self, question: str, contexts: list[dict]) -> str:
        """基于检索上下文生成最终答案"""
        user_prompt = build_answer_prompt(question, contexts)

        try:
            resp = await self.client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": ANSWER_GENERATION_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.6,
                max_tokens=1500,
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.error(f"答案生成失败: {e}")
            return f"[错误] 答案生成失败: {e}"
