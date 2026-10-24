"""
地质文本切块器：
- 按 章节 > 段落 > 句子 > 标点 > 字符 逐级降级切分，块长有硬上限
- 地质专业术语保护（不把 GEO_TERMS 中的术语从中间切断）
- markdown 表格与外部传入的结构化表格都独立成块
- 重叠区对齐句子边界；标题留在自己块开头，并写入 metadata.section
"""
import re
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, GEO_TERMS
from src.data_processing.metadata import ChunkMetadata

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """切块后的文本片段"""
    chunk_id: str
    text: str
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)


class GeologicalChunker:
    """
    地质文本切块器。

    特性：
    1. 默认 chunk_size=512, overlap=64，单位是字符数
    2. 逐级降级切分：章节标题 > 段落(\\n\\n) > 句子(。！？；) > 标点 > 字符
    3. 标题强制开新块，标题永远留在自己块开头，并记进 metadata.section
    4. 不会把 GEO_TERMS 中的术语从中间切断（含字符级兜底）
    5. 表格数据独立成块，附带表格标题作为上下文
    """

    # Markdown 标题行
    HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
    # 紧贴正文的标题行：用补空行的方式把它单独拆成一个段落
    HEADING_LINE_RE = re.compile(r"(?m)^(#{1,6} .+)$")
    # Markdown 表格的表头分隔行，如 |---|---| 与 | :--- | ---: |
    SEPARATOR_RE = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")
    # 句子终止符：重叠区和降级切分都以它为准
    SENTENCE_ENDS = "。！？；\n"
    # 字符级兜底时允许断在其后的标点（含空白，兼容英文文献与数字串）
    CUT_PUNCT = ["。", "！", "？", "；", "，", "、", "：", " ", "\n"]

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # 预编译术语匹配（按长度降序，同一位置优先匹配长术语）
        sorted_terms = sorted(GEO_TERMS, key=len, reverse=True)
        self.term_pattern = re.compile("|".join(re.escape(t) for t in sorted_terms))
        self.max_term_len = max(len(t) for t in GEO_TERMS)

    def chunk_document(
        self,
        text: str,
        tables: Optional[list] = None,
        source: str = "unknown",
        doc_type: str = "技术文档",
        category: str = "",
        location: str = "",
    ) -> list[TextChunk]:
        """
        对单个文档执行切块。
        返回 TextChunk 列表，每个附带完整元数据。
        """
        chunks = []
        base_meta = ChunkMetadata(
            source=source,
            doc_type=doc_type,
            category=category,
            location=location,
        )

        # 1. 主文本切块：切成 (正文, 章节, 类型) 三元组，再补重叠
        blocks = self._apply_overlap(self._recursive_split(text))
        for i, (chunk_text, section, kind) in enumerate(blocks):
            meta = ChunkMetadata(
                **{**base_meta.__dict__,
                   "section": section,
                   "content_type": kind,
                   "chunk_index": i,
                   "page": self._estimate_page(i, len(blocks), base_meta)}
            )
            chunks.append(TextChunk(
                chunk_id=f"{Path(source).stem}_chunk_{i:04d}",
                text=chunk_text,
                metadata=meta,
            ))

        # 2. 表格独立成块
        if tables:
            for ti, table in enumerate(tables):
                table_text = self._format_table(table)
                if table_text.strip():
                    meta = ChunkMetadata(
                        **{**base_meta.__dict__, "chunk_index": len(blocks) + ti, "content_type": "table"}
                    )
                    chunks.append(TextChunk(
                        chunk_id=f"{Path(source).stem}_table_{ti:04d}",
                        text=table_text,
                        metadata=meta,
                    ))

        n_inline_table = sum(1 for b in blocks if b[2] == "table")
        logger.info(
            f"{source}: 切分为 {len(chunks)} 块 "
            f"(正文 {len(blocks) - n_inline_table} + 表格 {n_inline_table + len(tables or [])})"
        )
        return chunks

    def _recursive_split(self, text: str) -> list[tuple[str, str, str]]:
        """
        递归切分：章节 > 段落 > 句子 > 标点 > 字符。
        返回 (块文本, 所属章节, content_type) 列表。

        规则：
        - Markdown 标题强制开新块，标题永远留在自己块开头，并记进 metadata.section
        - 标题后面紧跟标题（如"文档大标题 + 一级章节"）时不孤立封存，标题链和自己的正文呆在一起
        - 表格行自成一块，前后都必须断开；表格不参与重叠
        - 任何即将存下来的超长文本都走 _flush 继续降级切分，
          避免出现"整段只有一个句号 → 一个 2000 字的块"
        """
        # 标题行后面缺空行时补一个，别让 "# 标题 + 正文" 整段被当成正文
        normalized = self.HEADING_LINE_RE.sub(r"\1\n", text)
        paragraphs = [p.strip() for p in normalized.split("\n\n") if p.strip()]
        out: list[tuple[str, str, str]] = []
        current = ""
        current_section = ""
        current_has_body = False
        section = ""

        def flush_body() -> None:
            nonlocal current, current_has_body
            if current:
                for part in self._flush(current):
                    out.append((part, current_section, "text"))
                current, current_has_body = "", False

        for para in paragraphs:
            if self._is_table_block(para):
                if current_has_body:
                    flush_body()
                if current:
                    # 攒着的标题链没有正文就撞上了表格 → 它正是这张表的标题
                    para = current + "\n" + para
                    current, current_has_body = "", False
                for part in self._split_table(para):
                    out.append((part, section, "table"))
                continue

            m = self.HEADING_RE.match(para)
            if m:
                if current_has_body:
                    flush_body()
                section = m.group(2).strip()
                current = f"{current}\n{para}" if current else para
                current_section = section
                continue

            if not current_has_body:
                # current 只攒了标题：正文必须接在后面，长度交给 _flush 兜底
                current = f"{current}\n{para}" if current else para
                current_section = section
                current_has_body = True
            elif len(current) + len(para) + 1 <= self.chunk_size:
                current = current + "\n" + para
            else:
                flush_body()
                current, current_section, current_has_body = para, section, True

        flush_body()
        return out

    def _is_table_block(self, para: str) -> bool:
        """整段都是 '|' 开头的行 → 认为是一张 markdown 表格。"""
        lines = [l for l in para.split("\n") if l.strip()]
        return len(lines) >= 2 and all(l.lstrip().startswith("|") for l in lines)

    def _split_table(self, para: str) -> list[str]:
        """
        表格块（可带标题行）。超过 chunk_size 的长表按数据行分组，
        每组都重复"标题 + 表头"，保证任何半张表都还能读出这一列是什么。
        """
        if len(para) <= self.chunk_size:
            return [para]

        lines = [l.strip() for l in para.split("\n") if l.strip()]
        caption: list[str] = []
        idx = 0
        while idx < len(lines) and not lines[idx].startswith("|"):
            caption.append(lines[idx])
            idx += 1
        rows = lines[idx:]
        header = rows[:2] if len(rows) > 2 and self.SEPARATOR_RE.match(rows[1]) else rows[:1]
        prefix_lines = caption + header
        prefix = "\n".join(prefix_lines)

        blocks: list[str] = []
        group: list[str] = []
        size = len(prefix)
        for row in rows[len(header):]:
            if group and size + len(row) + 1 > self.chunk_size:
                blocks.append(prefix + "\n" + "\n".join(group))
                group, size = [], len(prefix)
            group.append(row)
            size += len(row) + 1
        if group:
            blocks.append(prefix + "\n" + "\n".join(group))
        return blocks or [prefix]

    def _flush(self, text: str) -> list[str]:
        """把任意长度的文本逐级切到不超过 chunk_size。"""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        pieces: list[str] = []
        for sent in self._split_by_sentence(text):
            if len(sent) <= self.chunk_size:
                pieces.append(sent)
            else:
                pieces.extend(self._hard_cut(sent))
        return [p for p in pieces if p.strip()]

    def _split_by_sentence(self, para: str) -> list[str]:
        """按中文句号、问号、感叹号、分号切分超长段落。"""
        # 换行故意不作切点：标题和正文若在换行处断开，标题会被孤零零切成一块
        sentences = re.split(r'(?<=[。！？；])', para)
        result: list[str] = []
        current = ""

        for sent in sentences:
            if not sent.strip():
                continue
            candidate = current + sent
            if len(candidate) > self.chunk_size and current:
                result.append(current.strip())
                current = sent
            else:
                current = candidate

        if current.strip():
            result.append(current.strip())
        return result

    def _hard_cut(self, sentence: str) -> list[str]:
        """
        字符级兜底：优先在标点后断开，实在没有标点才硬切。
        同时保证不从地质术语中间劈开。
        """
        limit = self.chunk_size
        pieces: list[str] = []
        rest = sentence

        while len(rest) > limit:
            window = rest[:limit]
            cut = max(window.rfind(p) + 1 for p in self.CUT_PUNCT)
            if cut <= 0:
                cut = self._term_safe_position(rest, limit)
            else:
                cut = self._term_safe_position(rest, cut)
            if cut <= 0 or cut >= len(rest):
                cut = limit
            pieces.append(rest[:cut])
            rest = rest[cut:]

        if rest:
            pieces.append(rest)
        return pieces

    def _term_safe_position(self, text: str, pos: int) -> int:
        """若 pos 落在某个地质术语内部，把切点前移到该术语开头。"""
        # 跨越 pos 的术语，起点最远只可能在 pos-max_term_len+1 之后，
        # 所以只扫 pos 附近这一小段，不必扫全文
        lo = max(0, pos - self.max_term_len + 1)
        window = text[lo:pos + self.max_term_len]
        for m in self.term_pattern.finditer(window):
            if m.start() + lo < pos < m.end() + lo:
                return m.start() + lo
        return pos

    def _apply_overlap(self, blocks: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
        """
        给正文块补上上一正文块的尾巴。

        重叠区必须对齐句子边界：宁可重叠为 0，也不要让块首以半句开头
        —— 半句会污染 embedding，还会让 reranker 拿到没有主语的碎片。

        表格块原样透传：它既不接收重叠，也不给下一块贡献尾巴，
        否则表格最后几行会被当成正文塞进下一个语义段。
        """
        out: list[tuple[str, str, str]] = []
        prev_body = ""
        for content, section, kind in blocks:
            if kind != "text":
                out.append((content, section, kind))
                continue
            out.append((self._aligned_tail(prev_body) + content, section, kind))
            prev_body = content   # 用未加重叠的原始正文取尾巴，避免逐块累积
        return out

    def _aligned_tail(self, prev: str) -> str:
        """取 prev 末尾 chunk_overlap 个字符里、第一个完整句之后的部分。"""
        # 注意 prev[-0:] 等于整段，overlap=0 必须在这里挡住
        if self.chunk_overlap <= 0 or not prev:
            return ""
        tail = prev[-self.chunk_overlap:]
        for k, ch in enumerate(tail):
            if ch in self.SENTENCE_ENDS and k < len(tail) - 1:
                # lstrip：终止符后面常跟着软换行，别让块首顶着空白
                return tail[k + 1:].lstrip()
        # 片段里没有句子边界 → 放弃这块重叠，而不是切断句子
        return ""

    def _format_table(self, table: dict) -> str:
        """将表格数据格式化为可检索的文本块"""
        caption = table.get("caption", "")
        rows = table.get("rows", [])
        columns = table.get("columns", [])

        parts = []
        if caption:
            parts.append(f"[表格] {caption}")
        if columns:
            parts.append(" | ".join(str(c) for c in columns))
        for row in rows[:20]:  # 限制行数，防止超长
            parts.append(" | ".join(str(cell) for cell in row))
        if len(rows) > 20:
            parts.append(f"... (共 {len(rows)} 行)")
        return "\n".join(parts)

    @staticmethod
    def _estimate_page(chunk_index: int, total_chunks: int, meta: ChunkMetadata) -> int:
        """粗估页码（Demo阶段简化处理）"""
        return 0  # 待后续从 PDF 元数据映射
