"""
init_demo.py — 一键初始化 Demo（解析 → 向量化 → 验证检索）
用法: python init_demo.py

前置：
1. pip install -r requirements.txt
2. 复制 .env.example 为 .env，填入 DEEPSEEK_API_KEY
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import *
from src.data_processing.parser import DocumentParser
from src.data_processing.chunker import GeologicalChunker
from src.vectorization.embedder import BGEEmbedder
from src.vectorization.store import VectorStore


def main():
    start = time.time()
    print("=" * 60)
    print("地质大模型 Demo — 一键初始化")
    print("=" * 60)

    # Step 1: 文档解析
    print("\n[1/4] 解析 data/raw/ 下的文档...")
    parser = DocumentParser()
    docs = parser.parse_all()
    parser.save_parsed(docs)
    print(f"  ✓ 解析完成: {len(docs)} 个文档")

    # Step 2: 切块
    print("\n[2/4] 文本切块...")
    chunker = GeologicalChunker()
    all_chunks = []
    for doc in docs:
        chunks = chunker.chunk_document(
            text=doc.text,
            tables=doc.tables,
            source=doc.source_file,
            doc_type=doc.doc_type,
        )
        all_chunks.extend(chunks)

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    (CHUNKS_DIR / "all_chunks.json").write_text(
        json.dumps([{"chunk_id": c.chunk_id, "text": c.text, "metadata": c.metadata.to_dict()} for c in all_chunks],
                   ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"  ✓ 切块完成: {len(all_chunks)} 个文本块")

    # Step 3: 向量化
    print("\n[3/4] 向量化（加载 bge-m3 模型需要 1-2 分钟）...")
    texts = [c.text for c in all_chunks]
    ids = [c.chunk_id for c in all_chunks]
    metadatas = [c.metadata.to_dict() for c in all_chunks]

    embedder = BGEEmbedder()
    embeddings = embedder.encode_dense(texts, batch_size=16)

    store = VectorStore()
    store.delete_all()
    store.add_documents(texts=texts, embeddings=embeddings, ids=ids, metadatas=metadatas)
    print(f"  ✓ 向量库: {store.count} 条向量已写入 {CHROMA_DIR}")

    # Step 4: 快速验证
    print("\n[4/4] 验证检索...")
    test_q = "德兴铜矿的成矿时代是什么？"
    q_vec = embedder.encode_query(test_q)
    results = store.search(q_vec, top_k=3)
    print(f"  测试查询: \"{test_q}\"")
    for r in results[:3]:
        preview = r["text"][:80].replace("\n", " ")
        print(f"  [{r['score']:.3f}] {r['metadata'].get('source','')} → {preview}...")

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"初始化完成！耗时 {elapsed:.1f}s")
    print(f"\n下一步:")
    print(f"  启动问答界面:  streamlit run src/app/streamlit_app.py")
    print(f"  交互测试:       python scripts/04_test_retrieval.py")
    print(f"  构建知识图谱:   python scripts/03_build_kg.py")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
