"""
Multi-Query 改写：生成 2-3 条同义检索query。
"""
import json
import logging

from openai import AsyncOpenAI

from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from src.prompt.query_rewrite import QUERY_REWRITE_PROMPT

logger = logging.getLogger(__name__)


async def multi_query_rewrite(
    question: str,
    client: AsyncOpenAI = None,
    num_queries: int = 3,
) -> list[str]:
    """
    将原始问题改写为 2-3 条不同表述的检索 query。
    返回包含原始问题 + 改写结果的列表（已去重）。
    """
    if client is None:
        client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    prompt = QUERY_REWRITE_PROMPT.format(question=question)

    try:
        response = await client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        queries = data.get("queries", [])[:num_queries]
    except Exception as e:
        logger.warning(f"Multi-Query 改写失败: {e}, 回退原始问题")
        queries = []

    # 合并去重（原始 query 始终保留）
    all_queries = [question] + [q for q in queries if q != question]
    logger.info(f"Multi-Query: 生成 {len(all_queries)} 条检索词")
    return all_queries
