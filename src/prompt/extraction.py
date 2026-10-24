"""
三元组抽取 Prompt 模板。
"""
import json
from src.knowledge_graph.schema import GeoSchema


TRIPLE_EXTRACTION_SYSTEM = """你是一个地质领域知识图谱构建专家。请从以下地质文本中抽取知识三元组。

## 允许的实体类型（严格遵守，不可自创）：
{entity_types}

## 允许的关系类型（严格遵守）：
{relation_types}

## 输出格式（JSON）：
{{"triples": [
  {{
    "head": {{"name": "实体名", "label": "实体类型", "properties": {{}}}},
    "relation": {{"type": "关系中文名", "identifier": "REL_ID", "properties": {{}}}},
    "tail": {{"name": "实体名", "label": "实体类型", "properties": {{}}}},
    "confidence": 0.0-1.0,
    "evidence": "原文中的证据句子"
  }}
]}}

## 规则：
1. 实体名称做归一化："德兴铜矿" = "德兴铜矿床" = "德兴"，统一用全称
2. confidence < 0.7 的三元组也输出，但标注低置信度
3. 不要遗漏文本中的任何地质关系
4. 如果文本中没有可抽取的三元组，返回 {{"triples": []}}"""


TRIPLE_EXTRACTION_FEW_SHOT = """## 示例输入文本：
"德兴铜矿位于江西省德兴市，是中国最大的斑岩型铜矿床。矿体赋存于震旦系下统双桥山群千枚岩中，受燕山早期（约171Ma）花岗闪长斑岩侵入控制。主要矿石矿物为黄铜矿和斑铜矿，伴有强烈的硅化和绢云母化蚀变。"

## 示例输出：
{"triples": [
  {
    "head": {"name": "德兴铜矿", "label": "Deposit", "properties": {"矿种": ["Cu"], "矿床类型": "斑岩型铜矿", "规模": "超大型"}},
    "relation": {"type": "产出于", "identifier": "OCCURS_IN", "properties": {"层位": "震旦系下统双桥山群"}},
    "tail": {"name": "震旦系下统", "label": "Stratum", "properties": {"年代": "新元古代", "岩性": "千枚岩"}},
    "confidence": 0.95,
    "evidence": "矿体赋存于震旦系下统双桥山群千枚岩中"
  },
  {
    "head": {"name": "德兴铜矿", "label": "Deposit", "properties": {}},
    "relation": {"type": "形成于", "identifier": "FORMED_IN", "properties": {"同位素年龄": "171Ma"}},
    "tail": {"name": "燕山早期", "label": "Geological_Time", "properties": {"绝对年龄_Ma": 171}},
    "confidence": 0.92,
    "evidence": "受燕山早期（约171Ma）花岗闪长斑岩侵入控制"
  },
  {
    "head": {"name": "花岗闪长斑岩", "label": "Rock", "properties": {"岩石类型": "岩浆岩-侵入岩"}},
    "relation": {"type": "控制", "identifier": "CONTROLS", "properties": {"控矿方式": "岩浆侵入"}},
    "tail": {"name": "德兴铜矿", "label": "Deposit", "properties": {}},
    "confidence": 0.88,
    "evidence": "受燕山早期花岗闪长斑岩侵入控制"
  },
  {
    "head": {"name": "黄铜矿", "label": "Mineral", "properties": {"化学式": "CuFeS2"}},
    "relation": {"type": "伴生", "identifier": "ASSOCIATED_WITH", "properties": {"关系": "共生"}},
    "tail": {"name": "斑铜矿", "label": "Mineral", "properties": {"化学式": "Cu5FeS4"}},
    "confidence": 0.90,
    "evidence": "主要矿石矿物为黄铜矿和斑铜矿"
  },
  {
    "head": {"name": "德兴铜矿", "label": "Deposit", "properties": {}},
    "relation": {"type": "伴生", "identifier": "ASSOCIATED_WITH", "properties": {"强度": "强烈"}},
    "tail": {"name": "硅化", "label": "Alteration", "properties": {"蚀变类型": "热液蚀变"}},
    "confidence": 0.85,
    "evidence": "伴有强烈的硅化和绢云母化蚀变"
  }
]}"""


def build_extraction_prompt(text: str) -> str:
    """组装完整的抽取 prompt"""
    system = TRIPLE_EXTRACTION_SYSTEM.format(
        entity_types=GeoSchema.entity_types_str(),
        relation_types=GeoSchema.relation_types_str(),
    )
    user_msg = f"{TRIPLE_EXTRACTION_FEW_SHOT}\n\n## 待抽取文本：\n\"\"\"\n{text[:3000]}\n\"\"\"\n\n请输出JSON："
    return f"{system}\n\n{user_msg}"
