"""
Step-back Prompting：从具体问题抽象出高层问题，获取背景知识后再回答。
"""
import json
import logging

from openai import AsyncOpenAI

from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from src.prompt.query_rewrite import STEP_BACK_PROMPT

logger = logging.getLogger(__name__)


async def step_back_transform(
    question: str,
    client: AsyncOpenAI = None,
) -> tuple[str, str]:
    """
    生成抽象化的 step-back 问题。

    返回 (abstract_question, original_question)。
    调用方应该分别检索两个问题，然后合并上下文。
    """
    if client is None:
        client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    prompt = STEP_BACK_PROMPT.format(question=question)

    try:
        response = await client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        abstract_q = data.get("abstract_question", question)
        logger.info(f"Step-back: '{question}' → '{abstract_q}'")
    except Exception as e:
        logger.warning(f"Step-back 失败: {e}")
        abstract_q = question

    return abstract_q, question
