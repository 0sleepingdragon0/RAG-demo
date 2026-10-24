"""
LLM 三元组抽取器：调用 DeepSeek API，按 Schema 从地质文本中抽取结构化三元组。
"""
import json
import asyncio
import logging
from typing import Optional

from openai import AsyncOpenAI

from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from src.knowledge_graph.schema import (
    Entity, EntityLabel, Relation, RelationType, Triple, GeoSchema
)
from src.prompt.extraction import build_extraction_prompt

logger = logging.getLogger(__name__)


class TripleExtractor:
    """
    地质三元组抽取器。

    用法：
        extractor = TripleExtractor()
        triples = extractor.extract_sync("德兴铜矿位于...")
        # 或异步批量
        triples_list = await extractor.extract_batch(texts)
    """

    def __init__(
        self,
        api_key: str = DEEPSEEK_API_KEY,
        base_url: str = DEEPSEEK_BASE_URL,
        model: str = DEEPSEEK_MODEL,
        temperature: float = 0.1,
        max_retries: int = 3,
    ):
        if not api_key:
            raise ValueError("缺少 DEEPSEEK_API_KEY，请在 .env 中配置")
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature  # 抽取任务用低温
        self.max_retries = max_retries

    async def extract(self, text: str) -> list[Triple]:
        """从单段文本抽取三元组"""
        prompt = build_extraction_prompt(text)

        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content
                return self._parse_triples(content)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON 解析失败 (尝试 {attempt+1}): {e}")
            except Exception as e:
                logger.error(f"API 调用失败 (尝试 {attempt+1}): {e}")
                await asyncio.sleep(2 ** attempt)

        logger.error(f"抽取失败，已达最大重试次数")
        return []

    async def extract_batch(self, texts: list[str], concurrency: int = 5) -> list[list[Triple]]:
        """批量并发抽取"""
        semaphore = asyncio.Semaphore(concurrency)

        async def _one(text):
            async with semaphore:
                return await self.extract(text)

        results = await asyncio.gather(*[_one(t) for t in texts])
        total = sum(len(r) for r in results)
        logger.info(f"批量抽取完成: {len(texts)} chunks → {total} triples")
        return results

    def extract_sync(self, text: str) -> list[Triple]:
        """同步调用封装"""
        return asyncio.run(self.extract(text))

    def _parse_triples(self, json_str: str) -> list[Triple]:
        """将 LLM JSON 输出解析为 Triple 对象列表，校验 Schema"""
        data = json.loads(json_str)
        if isinstance(data, dict):
            data = data.get("triples", data.get("result", [data]))

        triples = []
        for item in data:
            try:
                head = self._parse_entity(item["head"])
                relation = self._parse_relation(item["relation"])
                tail = self._parse_entity(item["tail"])
                confidence = float(item.get("confidence", 0.8))
                evidence = item.get("evidence", "")

                triple = Triple(
                    head=head, relation=relation, tail=tail,
                    confidence=confidence, evidence=evidence,
                )

                valid, reason = GeoSchema.validate_triple(triple)
                if not valid:
                    logger.warning(f"跳过非法三元组: {reason}")
                    continue
                triples.append(triple)
            except (KeyError, ValueError) as e:
                logger.warning(f"解析单条三元组失败: {e}")

        return triples

    @staticmethod
    def _parse_entity(data: dict) -> Entity:
        label_str = data.get("label", "")
        # 支持中文/英文标签
        label = None
        for el in EntityLabel:
            if el.value == label_str:
                label = el
                break
        if label is None:
            raise ValueError(f"未知实体类型: {label_str}")
        return Entity(
            name=data["name"].strip(),
            label=label,
            properties=data.get("properties", {}),
        )

    @staticmethod
    def _parse_relation(data: dict) -> Relation:
        identifier = data.get("identifier", "")
        rtype = None
        for rt in RelationType:
            if rt.value == identifier or rt.cn_name == identifier:
                rtype = rt
                break
        if rtype is None:
            raise ValueError(f"未知关系类型: {identifier}")
        return Relation(type=rtype, properties=data.get("properties", {}))
