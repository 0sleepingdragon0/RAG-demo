"""
scripts/01_parse_docs.py — 批量解析文档 + 切块 + 输出 chunks JSON
用法: python scripts/01_parse_docs.py
"""
import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import *
from src.data_processing.parser import DocumentParser
from src.data_processing.chunker import GeologicalChunker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("pipeline.parse")


def main():
    # 1. 解析
    parser = DocumentParser()
    docs = parser.parse_all()
    parser.save_parsed(docs)

    # 2. 切块
    chunker = GeologicalChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    all_chunks = []

    for doc in docs:
        chunks = chunker.chunk_document(
            text=doc.text,
            tables=doc.tables,
            source=doc.source_file,
            doc_type=doc.doc_type,
            location=doc.metadata.get("location", ""),
        )
        all_chunks.extend(chunks)

    # 3. 保存
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = CHUNKS_DIR / "all_chunks.json"
    data = [
        {
            "chunk_id": c.chunk_id,
            "text": c.text,
            "metadata": c.metadata.to_dict(),
        }
        for c in all_chunks
    ]
    output_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"共输出 {len(all_chunks)} 个文本块 → {output_file}")


if __name__ == "__main__":
    main()
