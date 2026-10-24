"""
Chunk 元数据定义
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ChunkMetadata:
    """每个文本块的元数据，用于检索过滤和溯源"""
    source: str = ""              # 来源文件名，如 "庐枞盆地勘查报告.pdf"
    doc_type: str = "技术文档"     # "勘查报告" / "论文" / "地质图" / "结构化数据"
    category: str = ""            # 内容分类："矿床地质" / "地球化学" / "构造地质"
    location: str = ""            # 地理/矿区标注："庐枞盆地" / "德兴铜矿"
    section: str = ""             # 所属章节标题（最内层），如 "3.1 铜厂矿体"
    page: int = 0                 # 原始页码
    chunk_index: int = 0          # 块序号
    content_type: str = "text"    # "text" / "table" / "figure_caption"

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v}

    @classmethod
    def from_dict(cls, d: dict) -> "ChunkMetadata":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})
