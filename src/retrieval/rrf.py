"""
RRF（Reciprocal Rank Fusion）多路结果融合。
公式: score(d) = Σ 1/(k + rank_i(d)), k=60
"""
import logging
from collections import defaultdict

from config.settings import RRF_K

logger = logging.getLogger(__name__)


def rrf_fusion(
    ranked_lists: list[list[dict]],
    k: int = RRF_K,
    id_key: str = "id",
) -> list[dict]:
    """
    将多路检索的排序结果融合为一个统一排序。

    参数：
    - ranked_lists: 每路检索的排序结果 [{"id":..., "text":..., ...}, ...]
    - k: RRF 常数，默认 60
    - id_key: 用于识别同一文档的字段名

    返回：融合后的排序结果，附带 rrf_score 字段。
    """
    scores = defaultdict(float)
    doc_map = {}  # id → doc data（取第一次出现的内容）

    for rank_list in ranked_lists:
        for rank, doc in enumerate(rank_list, start=1):
            doc_id = doc[id_key]
            scores[doc_id] += 1.0 / (k + rank)
            if doc_id not in doc_map:
                doc_map[doc_id] = doc

    # 按 rrf_score 降序排列
    sorted_ids = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)
    result = []
    for doc_id in sorted_ids:
        doc = {**doc_map[doc_id], "rrf_score": scores[doc_id]}
        result.append(doc)

    logger.info(f"RRF 融合: {len(ranked_lists)} 路 → {len(result)} 条唯一结果")
    return result
