"""
地质领域知识图谱 Schema 定义。

参考：GB/T 9649《地质矿产术语分类代码》
实体类型 13 种 + 关系类型 14 种（与规格书 §4 一致）。
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Any


# === 实体类型 ===
class EntityLabel(str, Enum):
    DEPOSIT = "Deposit"                      # 矿床
    OREBODY = "OreBody"                      # 矿体
    STRATUM = "Stratum"                      # 地层
    ROCK = "Rock"                            # 岩石
    MINERAL = "Mineral"                      # 矿物
    STRUCTURE = "Structure"                  # 构造
    ALTERATION = "Alteration"                # 蚀变
    METALLOGENIC_BELT = "Metallogenic_Belt"  # 成矿带
    EXPLORATION_METHOD = "Exploration_Method" # 勘查方法
    GEOCHEM_ANOMALY = "Geochem_Anomaly"      # 地球化学异常
    GEOPHYS_ANOMALY = "Geophys_Anomaly"      # 地球物理异常
    GEOLOGICAL_TIME = "Geological_Time"      # 地质年代
    REGION = "Region"                        # 区域


# === 关系类型 ===
class RelationType(str, Enum):
    OCCURS_IN = ("产出于", "OCCURS_IN")
    FORMED_IN = ("形成于", "FORMED_IN")
    BELONGS_TO = ("属于", "BELONGS_TO")
    ASSOCIATED_WITH = ("伴生", "ASSOCIATED_WITH")
    CONTROLS = ("控制", "CONTROLS")
    WALLROCK_OF = ("围岩", "WALLROCK_OF")
    ALTERED_TO = ("蚀变", "ALTERED_TO")
    CONTAINS_ORE = ("含矿", "CONTAINS_ORE")
    INDICATES = ("指示", "INDICATES")
    EXPLORED_BY = ("勘查方法", "EXPLORED_BY")
    UNCONFORMABLE_WITH = ("不整合接触", "UNCONFORMABLE_WITH")
    CONFORMABLE_WITH = ("整合接触", "CONFORMABLE_WITH")
    INTRUDES = ("侵入", "INTRUDES")
    LOCATED_IN = ("位于", "LOCATED_IN")

    def __new__(cls, cn_name: str, identifier: str):
        obj = str.__new__(cls, identifier)
        obj._value_ = identifier
        obj.cn_name = cn_name
        return obj


# === 合法的 头-关系-尾 组合 ===
# 格式: (head_label, relation_type) -> [allowed tail_labels]
VALID_TRIPLES = {
    (EntityLabel.DEPOSIT, RelationType.OCCURS_IN): [EntityLabel.STRATUM, EntityLabel.REGION],
    (EntityLabel.OREBODY, RelationType.OCCURS_IN): [EntityLabel.STRATUM],
    (EntityLabel.DEPOSIT, RelationType.FORMED_IN): [EntityLabel.GEOLOGICAL_TIME],
    (EntityLabel.DEPOSIT, RelationType.BELONGS_TO): [EntityLabel.METALLOGENIC_BELT],
    (EntityLabel.DEPOSIT, RelationType.ASSOCIATED_WITH): [EntityLabel.ALTERATION],
    (EntityLabel.MINERAL, RelationType.ASSOCIATED_WITH): [EntityLabel.MINERAL],
    (EntityLabel.STRUCTURE, RelationType.CONTROLS): [EntityLabel.DEPOSIT, EntityLabel.OREBODY],
    (EntityLabel.ROCK, RelationType.WALLROCK_OF): [EntityLabel.DEPOSIT, EntityLabel.OREBODY],
    (EntityLabel.ROCK, RelationType.ALTERED_TO): [EntityLabel.ALTERATION],
    (EntityLabel.STRATUM, RelationType.CONTAINS_ORE): [EntityLabel.MINERAL],
    (EntityLabel.ROCK, RelationType.CONTAINS_ORE): [EntityLabel.MINERAL],
    (EntityLabel.GEOCHEM_ANOMALY, RelationType.INDICATES): [EntityLabel.DEPOSIT, EntityLabel.MINERAL],
    (EntityLabel.GEOPHYS_ANOMALY, RelationType.INDICATES): [EntityLabel.DEPOSIT],
    (EntityLabel.DEPOSIT, RelationType.EXPLORED_BY): [EntityLabel.EXPLORATION_METHOD],
    (EntityLabel.STRATUM, RelationType.UNCONFORMABLE_WITH): [EntityLabel.STRATUM],
    (EntityLabel.STRATUM, RelationType.CONFORMABLE_WITH): [EntityLabel.STRATUM],
    (EntityLabel.ROCK, RelationType.INTRUDES): [EntityLabel.STRATUM, EntityLabel.ROCK],
    (EntityLabel.DEPOSIT, RelationType.LOCATED_IN): [EntityLabel.REGION],
    (EntityLabel.GEOCHEM_ANOMALY, RelationType.LOCATED_IN): [EntityLabel.REGION],
}


# === 数据结构 ===
@dataclass
class Entity:
    name: str
    label: EntityLabel
    properties: dict = field(default_factory=dict)

    @property
    def node_id(self) -> str:
        """唯一标识：类型前缀 + 名称，避免同名不同实体合并"""
        return f"{self.label.value}:{self.name}"


@dataclass
class Relation:
    type: RelationType
    properties: dict = field(default_factory=dict)


@dataclass
class Triple:
    head: Entity
    relation: Relation
    tail: Entity
    confidence: float = 1.0
    evidence: str = ""

    def is_valid(self) -> bool:
        """校验三元组是否符合 Schema 约束"""
        key = (self.head.label, self.relation.type)
        allowed_tails = VALID_TRIPLES.get(key)
        if allowed_tails is None:
            return False
        return self.tail.label in allowed_tails


class GeoSchema:
    """地质 Schema 辅助类：提供给 Prompt 的 schema 字符串"""

    ENTITY_DESCRIPTIONS = {
        EntityLabel.DEPOSIT: "Deposit（矿床）：名称, 矿种, 矿床类型, 规模, 品位, 发现年份",
        EntityLabel.OREBODY: "OreBody（矿体）：编号, 形态, 产状, 长度, 延深, 厚度, 平均品位",
        EntityLabel.STRATUM: "Stratum（地层）：名称, 年代, 岩性描述, 厚度, 接触关系",
        EntityLabel.ROCK: "Rock（岩石）：名称, 岩石类型, 矿物组合, 结构构造",
        EntityLabel.MINERAL: "Mineral（矿物）：名称, 化学式, 晶系, 硬度, 颜色",
        EntityLabel.STRUCTURE: "Structure（构造）：名称, 类型, 走向, 倾向, 性质",
        EntityLabel.ALTERATION: "Alteration（蚀变）：名称, 蚀变类型, 蚀变矿物组合, 强度",
        EntityLabel.METALLOGENIC_BELT: "Metallogenic_Belt（成矿带）：名称, 位置, 主要矿种, 成矿时代",
        EntityLabel.EXPLORATION_METHOD: "Exploration_Method（勘查方法）：名称, 方法类型, 适用场景",
        EntityLabel.GEOCHEM_ANOMALY: "Geochem_Anomaly（地球化学异常）：名称, 元素组合, 异常面积, 峰值",
        EntityLabel.GEOPHYS_ANOMALY: "Geophys_Anomaly（地球物理异常）：名称, 方法, 异常特征",
        EntityLabel.GEOLOGICAL_TIME: "Geological_Time（地质年代）：名称, 宙/代/纪/世, 绝对年龄(Ma)",
        EntityLabel.REGION: "Region（区域）：名称, 级别, 坐标范围",
    }

    @classmethod
    def entity_types_str(cls) -> str:
        return "\n".join(f"- {desc}" for desc in cls.ENTITY_DESCRIPTIONS.values())

    @classmethod
    def relation_types_str(cls) -> str:
        return ", ".join(f"{r.cn_name}({r.value})" for r in RelationType)

    @classmethod
    def validate_triple(cls, triple: Triple) -> tuple[bool, str]:
        """返回 (是否合法, 错误原因)"""
        if not triple.is_valid():
            key = (triple.head.label, triple.relation.type)
            allowed = VALID_TRIPLES.get(key)
            return False, f"非法组合: ({triple.head.label.value})-[{triple.relation.type.value}]->({triple.tail.label.value}), 允许: {allowed}"
        return True, ""
