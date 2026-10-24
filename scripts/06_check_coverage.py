"""
06_check_coverage.py — 语料/测试集覆盖率校验

用途：在跑 RAGAS 评测之前，先确认"测试集问的东西语料里真的有"。
否则 Context Recall 低分反映的是语料缺口，而不是检索质量，评测结论会失真。

检查两个方向：
  A. 测试集 → 语料（无空考）
     - 每条用例声明的 source 文档必须存在于 data/raw/
     - 每条用例声明的 entity 必须至少出现在一份语料文档中
  B. 语料 → 测试集（无浪费语料）
     - 每份 data/raw/ 文档必须至少被一条用例引用

用法：
    python scripts/06_check_coverage.py
退出码：0 = 全部通过；1 = 存在缺口
"""
import io
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from config.settings import RAW_DIR  # noqa: E402
from src.evaluation.test_set import (  # noqa: E402
    CATEGORIES,
    GEO_TEST_CASES,
    category_summary,
)

# 通用地学/方法学术语，不需要语料里逐字出现，跳过实体校验
ENTITY_SKIP = {
    "勘查", "地球化学", "异常组合", "品位", "成矿规律", "找矿方向", "铜矿", "钨矿", "铀矿",
    "蚀变分带", "矿化分带", "斑岩型", "矽卡岩型", "火山岩型", "石英脉型", "热液脉型",
    "层控", "层状", "斜深", "喷发旋回", "成矿系统", "成岩成矿时差", "激电", "磁法", "重力",
    "遥感", "锆石", "SHRIMP", "Re-Os", "Ar-Ar", "WO₃", "εHf", "南岭", "赣杭", "九瑞",
}


def load_corpus() -> dict[str, str]:
    """读取 data/raw/ 下所有文档，返回 {不带扩展名的文件名: 全文}"""
    corpus: dict[str, str] = {}
    for p in sorted(RAW_DIR.iterdir()):
        if p.suffix.lower() not in (".md", ".txt"):
            continue
        corpus[p.stem] = io.open(p, encoding="utf-8", errors="ignore").read()
    return corpus


def main() -> int:
    corpus = load_corpus()
    print("=" * 66)
    print("语料 / 测试集覆盖率校验")
    print("=" * 66)

    total_chars = sum(len(v) for v in corpus.values())
    print(f"\n[语料] data/raw/ 共 {len(corpus)} 份文档，合计 {total_chars} 字符")
    for name, text in corpus.items():
        print(f"    - {name:<28} {len(text):>6} 字符")

    print(f"\n[测试集] 共 {len(GEO_TEST_CASES)} 条用例")
    for c in CATEGORIES:
        print(f"    - {c:<8} {category_summary()[c]:>3} 条")

    problems: list[str] = []

    # ---- A. 测试集 -> 语料 ----
    print("\n[A] 测试集引用的文档是否存在")
    missing_docs: list[str] = []
    for tc in GEO_TEST_CASES:
        for s in tc.source:
            if s not in corpus:
                missing_docs.append(f"{s}  <- {tc.question[:24]}")
    if missing_docs:
        problems.append("引用了不存在的语料文档")
        for m in sorted(set(missing_docs)):
            print(f"    [缺失] {m}")
    else:
        print(f"    OK  全部 {len({s for tc in GEO_TEST_CASES for s in tc.source})} 个引用均有对应文件")

    print("\n[A2] 测试集考察的实体是否出现在语料中")
    missing_entities: list[tuple[str, str]] = []
    for tc in GEO_TEST_CASES:
        for ent in tc.entities:
            if ent in ENTITY_SKIP:
                continue
            hit = any(ent in text for text in corpus.values())
            if not hit:
                missing_entities.append((ent, tc.question))
    if missing_entities:
        problems.append("有实体在语料中检索不到")
        seen = set()
        for ent, q in missing_entities:
            if ent in seen:
                continue
            seen.add(ent)
            print(f"    [缺失] {ent}   <- {q[:34]}")
    else:
        print("    OK  全部实体可在语料中命中")

    # ---- B. 语料 -> 测试集 ----
    print("\n[B] 每份语料是否至少被一条用例覆盖")
    used = {s for tc in GEO_TEST_CASES for s in tc.source}
    orphans = [name for name in corpus if name not in used]
    if orphans:
        problems.append("有语料文档未被任何用例覆盖")
        for o in orphans:
            print(f"    [未覆盖] {o}")
    else:
        print(f"    OK  {len(corpus)} 份文档全部被覆盖")

    # 每份文档被多少条用例引用
    print("\n[B2] 各文档被引用次数")
    for name in corpus:
        cnt = sum(1 for tc in GEO_TEST_CASES if name in tc.source)
        flag = "  <-- 偏低" if cnt < 2 else ""
        print(f"    {name:<28} {cnt:>3} 条{flag}")

    print("\n" + "=" * 66)
    if problems:
        print(f"结果：发现 {len(problems)} 类缺口")
        for p in problems:
            print(f"    - {p}")
        print("\n修复建议：补写语料文档，或修改测试集用例，使其与语料一致。")
        return 1
    print("结果：测试集与语料完全对齐，可以跑评测。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
