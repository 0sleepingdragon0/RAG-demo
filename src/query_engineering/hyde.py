"""
HyDE（Hypothetical Document Embeddings）：生成假设性答案用于向量检索。
"""
import logging

from openai import AsyncOpenAI

from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from src.prompt.query_rewrite import HYDE_PROMPT

logger = logging.getLogger(__name__)


async def hyde_transform(
    question: str,
    client: AsyncOpenAI = None,
    include_original: bool = True,
) -> list[str]:
    """
    生成假设性答案文档，用它的 embedding 去检索真实文档。

    返回 [hyde_text, original_question]（或仅 [hyde_text]）。
    """
    if client is None:
        client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    prompt = HYDE_PROMPT.format(question=question)

    try:
        response = await client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,  # HyDE 允许一定创造性
            max_tokens=300,
        )
        hyde_doc = response.choices[0].message.content.strip()
        logger.info(f"HyDE 生成假设文档 ({len(hyde_doc)} 字)")
    except Exception as e:
        logger.warning(f"HyDE 失败: {e}, 回退原始问题")
        hyde_doc = ""

    result = []
    if hyde_doc:
        result.append(hyde_doc)
    if include_original or not hyde_doc:
        result.append(question)
    return result
