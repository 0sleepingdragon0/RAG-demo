"""
GeologicalChunker 行为测试。

可直接运行（无需 pytest）：
    python tests/test_chunker.py
装了 pytest 后也可以：
    pytest tests/test_chunker.py

这套测试是为了修掉 2026-10-24 实测出来的四个问题而写的：
  1. 没有字符级兜底 —— 单句 2041 字照样出一个 2041 字的块
  2. 章节标题被孤立留在上一块末尾，正文掉进下一块
  3. overlap 是纯字符切片，块首常常从半句中间开始
  4. markdown 表格没走表格通道，被当普通文本切碎
每条测试都用手算出来的字面量做期望值，不用被测代码反推。
"""
import io
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.data_processing.chunker import GeologicalChunker  # noqa: E402

SENTENCE_ENDS = "。！？；\n"


# ==========================================================================
# 1. 尺寸契约：块（含重叠前缀）不得超过 chunk_size + chunk_overlap
# ==========================================================================
def test_oversized_single_sentence_is_split():
    """350 字、只在末尾有一个句号的句子，必须被硬切，不能出一个超长块。"""
    c = GeologicalChunker(chunk_size=100, chunk_overlap=20)
    text = "德兴铜矿是亚洲最大的斑岩型铜矿床" * 12 + "。"   # 15*12 = 180 + 1 = 181 字
    chunks = c.chunk_document(text, source="t.md")

    assert len(chunks) >= 2, f"181 字应当切成多块，实际只有 {len(chunks)} 块"
    for ch in chunks:
        assert len(ch.text) <= 120, f"块超长：{len(ch.text)} 字（上限 100+20）"


def test_hard_cut_preserves_first_and_last_content():
    """硬切不能吞掉内容：原文开头和结尾都要落在块里。"""
    c = GeologicalChunker(chunk_size=100, chunk_overlap=0)
    head = "甲乙丙丁戊己庚辛壬癸"          # 10 字，作首哨兵
    tail = "子丑寅卯辰巳午未申酉"          # 10 字，作尾哨兵
    text = head + "填" * 300 + tail
    chunks = c.chunk_document(text, source="t.md")

    assert chunks[0].text.startswith(head), "第一块开头丢了原文首部内容"
    assert chunks[-1].text.endswith(tail), "最后一块结尾丢了原文尾部内容"


def test_hard_cut_prefers_punctuation_boundary():
    """字符级兜底要优先落在标点后面，不该把 '铜铅锌金银、' 这种单元劈开。"""
    c = GeologicalChunker(chunk_size=100, chunk_overlap=0)
    unit = "铜铅锌金银、"                 # 6 字
    text = unit * 30                      # 180 字
    chunks = c.chunk_document(text, source="t.md")

    # 手算：100 // 6 = 16 个单元 = 96 字，余 14 个单元 = 84 字
    assert [ch.text for ch in chunks] == [unit * 16, unit * 14], (
        "未在标点处切分或切分位置错误："
        + str([(len(ch.text), ch.text[-4:]) for ch in chunks])
    )


# ==========================================================================
# 2. 章节边界：标题不能孤悬在上一块末尾，正文要跟着自己的标题
# ==========================================================================
def test_heading_stays_with_its_body():
    """'## 第二章 构造' 所在的那一块，必须同时包含该章的第一句正文。"""
    c = GeologicalChunker(chunk_size=100, chunk_overlap=0)
    p1 = "区内发育北东向断裂。" * 5              # 10*5 = 50 字
    heading = "## 第二章 构造"
    p2 = "北西向断裂切割背斜轴部。" * 5          # 12*5 = 60 字
    text = f"{p1}\n\n{heading}\n\n{p2}"

    chunks = c.chunk_document(text, source="t.md")
    with_heading = [ch for ch in chunks if heading in ch.text]
    assert with_heading, "标题不见了"
    assert any("北西向断裂切割背斜轴部。" in ch.text for ch in with_heading), (
        "标题与它的正文被切开了：标题成了上一块的尾巴"
    )


def test_chunk_carries_section_metadata():
    """每块要记住自己属于哪个章节，检索结果才能报出出处层级。"""
    c = GeologicalChunker(chunk_size=100, chunk_overlap=0)
    text = (
        "## 第三章 矿床地质特征\n\n"
        "矿体呈透镜状产出。\n\n"
        "### 3.1 铜厂矿体\n\n"
        "主矿体长一千二百米。"
    )
    chunks = c.chunk_document(text, source="t.md")

    sections = [ch.metadata.section for ch in chunks]
    assert sections[0] == "第三章 矿床地质特征", f"第一块章节标注错：{sections[0]!r}"
    assert "3.1 铜厂矿体" in sections[-1], f"末块章节标注错：{sections[-1]!r}"


# ==========================================================================
# 3. 重叠区必须对齐句子边界，不能把上一块的后半句劈开带来
# ==========================================================================
def test_overlap_starts_at_sentence_boundary():
    """带重叠的块，其开头在原文里必须紧跟一个句子终止符。"""
    c = GeologicalChunker(chunk_size=20, chunk_overlap=8)
    p1 = "第一块甲句。第一块乙句。第一块丙句。"          # 18 字
    p2 = "第二块丁句。第二块戊句。第二块己句。"          # 18 字
    text = p1 + "\n\n" + p2

    chunks = c.chunk_document(text, source="t.md")
    assert len(chunks) >= 2, "这个 fixture 应该至少切成两块"

    for ch in chunks[1:]:
        frag = ch.text[:6]
        pos = text.find(frag)
        assert pos != -1, f"块首 {frag!r} 在原文里找不到，重叠把内容改写了"
        assert pos == 0 or text[pos - 1] in SENTENCE_ENDS, (
            f"块首 {ch.text!r} 从半句中间开始："
            f"原文中它前面是 {text[pos - 1]!r}，不是句子终止符"
        )


def test_overlap_stays_within_budget():
    """块正文 ≤ chunk_size，加重叠后总长 ≤ chunk_size + chunk_overlap。"""
    c = GeologicalChunker(chunk_size=20, chunk_overlap=8)
    # 情形 A：句子短，重叠窗口里就能找到句号
    short = "矿区内地层以千枚岩为主。" * 8          # 11*8 = 88 字，一个段落
    # 情形 B：整段只有一个句号，硬切出来的块末尾 8 字里没有句子边界。
    #         如果实现是"往前扩到整句开头"，这里会撑出 20+20=40 字
    long = "矿区内地层以千枚岩为主" * 8 + "。"        # 89 字单句
    for text in (short, long):
        for ch in c.chunk_document(text, source="t.md"):
            assert len(ch.text) <= 28, f"块超长：{len(ch.text)}（上限 20+8）：{ch.text[:40]!r}"


# ==========================================================================
# 4. 术语保护要作用在真正会切开的地方（字符级兜底）
# ==========================================================================
def test_hard_cut_keeps_geology_term_intact():
    """切点落在 '花岗闪长斑岩' 中间时，必须把切点退到术语开头，而不是劈开术语。"""
    c = GeologicalChunker(chunk_size=20, chunk_overlap=0)
    text = "甲乙丙丁戊己庚辛" * 2 + "花岗闪长斑岩" + "壬癸" * 2   # 16+6+4 = 26 字，无标点

    chunks = c.chunk_document(text, source="t.md")
    assert [ch.text for ch in chunks] == ["甲乙丙丁戊己庚辛" * 2, "花岗闪长斑岩壬癸壬癸"], (
        "术语被切断或切点位置不对：" + str([ch.text for ch in chunks])
    )


# ==========================================================================
# 5. 正文里的 markdown 表格必须独立成块，不能被当普通文本切碎
# ==========================================================================
def test_markdown_table_becomes_its_own_block():
    """夹在正文中的表格要单独成块，行不能被劈开、也不渗进相邻文本块。"""
    c = GeologicalChunker(chunk_size=100, chunk_overlap=20)
    intro = "区内主要矿床类型汇总如下表。"
    table = (
        "| 矿床类型 | 代表矿床 | 成矿时代 |\n"
        "| --- | --- | --- |\n"
        "| 斑岩型 | 德兴铜矿 | 171~163Ma |\n"
        "| 矽卡岩型 | 武山铜矿 | 145Ma |"
    )
    outro = "上表数据来源于江西省区域地质调查资料。"
    text = "\n\n".join([intro, table, outro])

    chunks = c.chunk_document(text, source="t.md")
    tables = [ch for ch in chunks if ch.metadata.content_type == "table"]
    assert len(tables) == 1, f"表格应独立成 1 块，实际 {len(tables)} 块"

    body = table.replace("\n", "")
    assert body in tables[0].text.replace("\n", ""), "表格内容有缺失或被改写"
    for ch in chunks:
        if ch is tables[0]:
            continue
        assert "成矿时代" not in ch.text, f"表格行渗进了非表格块：{ch.text[:40]!r}"
    assert any(intro in ch.text for ch in chunks), "表格前的正文块丢了"
    assert any(outro in ch.text for ch in chunks), "表格后的正文块丢了"


def test_table_does_not_donate_overlap():
    """重叠尾巴只能来自上一块正文，不能是表格的最后几行。"""
    c = GeologicalChunker(chunk_size=100, chunk_overlap=20)
    table = "| 项目 | 数值 |\n| --- | --- |\n| 铜储量 | 500万吨 |"
    after = "矿区构造以断裂为主，岩浆岩发育。"
    text = table + "\n\n" + after

    chunks = c.chunk_document(text, source="t.md")
    body = [ch for ch in chunks if ch.metadata.content_type != "table"]
    assert body and body[0].text.startswith(after), (
        f"表格后的块被表格尾巴污染：{body[0].text[:40]!r}"
    )


def test_zero_overlap_adds_no_prefix():
    """overlap=0 时块首必须就是原文，不能把上一块整块重复一遍（[-0:] 会取全文）。"""
    c = GeologicalChunker(chunk_size=20, chunk_overlap=0)
    p1 = "第一块甲句。第一块乙句。第一块丙句。"
    p2 = "第二块丁句。第二块戊句。第二块己句。"
    chunks = c.chunk_document(p1 + "\n\n" + p2, source="t.md")

    assert len(chunks) >= 2
    assert chunks[1].text.startswith("第二块丁句。"), repr(chunks[1].text)


def test_long_table_split_repeats_header():
    """超长表格被分组时，每组都要自带表头，否则后半组读不出列名。"""
    c = GeologicalChunker(chunk_size=100, chunk_overlap=0)
    rows = ["| 德兴铜矿 | 171Ma |", "| 武山铜矿 | 145Ma |", "| 永平铜矿 | 160Ma |",
            "| 银山铅矿 | 172Ma |", "| 西华山钨 | 155Ma |"]
    table = "\n".join(["| 矿床 | 时代 |", "|---|---|"] + rows)
    text = "## 表 1 矿集区一览表\n\n" + table

    chunks = c.chunk_document(text, source="t.md")
    tables = [ch for ch in chunks if ch.metadata.content_type == "table"]
    assert len(tables) >= 2, f"115 字的表在 size=100 下应被分组，实际 {len(tables)} 块"
    for ch in tables:
        assert "矿床" in ch.text and "时代" in ch.text, f"分组后表头丢失：{ch.text!r}"
        assert len(ch.text) <= 100, f"表格块超长：{len(ch.text)}"
    for r in rows:
        n = sum(1 for ch in tables if r in ch.text)
        assert n == 1, f"数据行丢失或重复（出现 {n} 次）：{r!r}"


def test_overlap_tail_starts_at_clean_sentence_start():
    """上一块含软换行（列表、PDF 换行常见）时，重叠尾巴不能把换行带到块首。"""
    c = GeologicalChunker(chunk_size=20, chunk_overlap=12)
    p1 = "- 甲岩。乙岩。\n- 丙岩。丁岩。"        # 17 字，块内含 "。\n"
    p2 = "戊岩呈层状。己岩呈脉状。"              # 12 字
    chunks = c.chunk_document(p1 + "\n\n" + p2, source="t.md")

    assert len(chunks) >= 2, "这个 fixture 应切成两块"
    for ch in chunks:
        assert ch.text == ch.text.strip(), f"块首尾有空白噪声：{ch.text!r}"


def test_heading_never_left_alone_after_downgrade():
    """标题 + 超长正文被降级切分时，不能剩一个只有标题的块。"""
    c = GeologicalChunker(chunk_size=100, chunk_overlap=0)
    heading = "## 第五章 深部找矿方向"
    body = "深部及覆盖区隐伏岩体是下一步找矿的重点，" * 8 + "。"   # 20*8+1 = 161 字
    chunks = c.chunk_document(heading + "\n\n" + body, source="t.md")

    for ch in chunks:
        lines = [l for l in ch.text.split("\n") if l.strip()]
        assert not (len(lines) == 1 and GeologicalChunker.HEADING_RE.match(lines[0])), (
            f"出现了只有标题的孤立块：{ch.text!r}")
    assert any(heading in ch.text and "深部及覆盖区" in ch.text for ch in chunks), (
        "标题和它的正文被切开了：" + str([ch.text[:20] for ch in chunks]))


def test_heading_without_blank_line_is_not_swallowed():
    """'# 标题\\n正文' 紧贴写法（MinerU 转出的 markdown 常见）也要认出标题行。"""
    c = GeologicalChunker(chunk_size=200, chunk_overlap=0)
    text = "# 区域地质报告\n本报告用于 Demo 验证。\n\n## 第一章 概况\n区内出露志留系地层。"
    chunks = c.chunk_document(text, source="t.md")

    assert chunks[0].metadata.section == "区域地质报告", (
        f"正文被并进了章节名：{chunks[0].metadata.section!r}")
    assert "本报告用于 Demo 验证。" in chunks[0].text
    assert chunks[-1].metadata.section == "第一章 概况", (
        f"末块章节名错：{chunks[-1].metadata.section!r}")


def test_explicit_tables_still_get_chunked():
    """MinerU 解析出的结构化表格（tables 参数）仍各自成块，序号接在正文块之后。"""
    c = GeologicalChunker(chunk_size=50, chunk_overlap=0)
    text = "区内发育花岗闪长斑岩体，接触带控制矿体。"
    tables = [{"caption": "表1 样品分析结果",
               "columns": ["样品编号", "Cu/%"],
               "rows": [["H-1", "0.85"], ["H-2", "1.20"]]}]

    chunks = c.chunk_document(text, tables=tables, source="t.md")
    tt = [ch for ch in chunks if ch.metadata.content_type == "table"]
    assert len(tt) == 1, f"传入 1 张结构化表格应产出 1 个表格块，实际 {len(tt)}"
    assert "表1 样品分析结果" in tt[0].text and "Cu/%" in tt[0].text and "0.85" in tt[0].text
    body = [ch for ch in chunks if ch.metadata.content_type != "table"]
    assert tt[0].metadata.chunk_index == len(body), "表格块的序号没有接在正文之后"


# ==========================================================================
# 6. 真实语料集成体检
# ==========================================================================
def test_real_corpus_invariants():
    """对 data/raw/*.md 逐文档检查四条硬性质。"""
    docs = sorted((PROJECT_ROOT / "data" / "raw").glob("*.md"))
    assert docs, "data/raw 下没有语料，集成测试失去意义"
    c = GeologicalChunker()
    budget = c.chunk_size + c.chunk_overlap
    table_chunks = 0

    for doc in docs:
        text = doc.read_text(encoding="utf-8")
        chunks = c.chunk_document(text, source=doc.name)
        assert chunks, f"{doc.name} 切出了 0 块"
        joined = [ch.text for ch in chunks]

        for ch in chunks:
            assert len(ch.text) <= budget, (
                f"{doc.name} 块 {ch.chunk_id} 超长 {len(ch.text)} > {budget}")
            if ch.metadata.content_type == "table":
                table_chunks += 1
            else:
                assert not re.search(r"^\s*\|.*\|\s*$", ch.text, re.M), (
                    f"{doc.name} 块 {ch.chunk_id} 里混进了表格行")

        # 原标题不能孤悬在块尾
        for ch in chunks:
            last = [l for l in ch.text.split("\n") if l.strip()]
            if last and GeologicalChunker.HEADING_RE.match(last[-1]):
                raise AssertionError(f"{doc.name} 块 {ch.chunk_id} 以孤立标题结尾：{last[-1]!r}")

        # 原文每一行都得能在某个块里找到（切块不能丢内容）
        for line in text.split("\n"):
            line = line.strip()
            if len(line) >= 8:
                assert any(line in t for t in joined), f"{doc.name} 内容丢失：{line[:40]!r}"

    assert table_chunks >= 1, "语料里有 markdown 表格，却一个表格块都没产出"


# ==========================================================================
# 独立运行器（无 pytest 时可用）
# ==========================================================================
def _run() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            print(f"  FAIL  {name}\n          {e}")
            failed.append(name)
        except Exception as e:
            print(f"  ERROR {name}\n          {type(e).__name__}: {e}")
            failed.append(name)
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
