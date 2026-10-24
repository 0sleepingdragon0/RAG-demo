"""
Query Decomposition：将复杂问题拆分为子问题。
"""
import json
import logging

from openai import AsyncOpenAI

from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from src.prompt.query_rewrite import DECOMPOSE_PROMPT

logger = logging.getLogger(__name__)


async def decompose_query(
    question: str,
    client: AsyncOpenAI = None,
    max_sub_queries: int = 3,
) -> list[str]:
    """
    将复杂问题拆分为最多 3 个子问题。
    返回子问题列表（包含原始问题作为兜底）。
    """
    if client is None:
        client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    prompt = DECOMPOSE_PROMPT.format(question=question)

    try:
        response = await client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        sub_queries = data.get("sub_queries", [])[:max_sub_queries]
        logger.info(f"Decomposition: '{question}' → {len(sub_queries)} 个子问题")
    except Exception as e:
        logger.warning(f"Decomposition 失败: {e}")
        sub_queries = []

    # 始终保留原始问题
    if question not in sub_queries:
        sub_queries.insert(0, question)
    return sub_queries
