"""
地质大模型 Demo — Streamlit 问答界面
启动: streamlit run src/app/streamlit_app.py
"""
import sys
import asyncio
from pathlib import Path

# 确保项目根目录在 path 中
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
from config.settings import *
from src.vectorization.embedder import BGEEmbedder
from src.vectorization.store import VectorStore
from src.vectorization.reranker import Reranker
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.query_engine import AdaptiveQueryEngine, QueryResult


st.set_page_config(page_title="地质大模型 Demo", page_icon="⛏️", layout="wide")

# === 样式 ===
st.markdown("""
<style>
    .route-tag {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 12px;
        font-size: 0.85em;
        font-weight: 600;
        margin: 2px 4px;
    }
    .route-direct { background: #e8f5e9; color: #2e7d32; }
    .route-hyde { background: #e3f2fd; color: #1565c0; }
    .route-multi { background: #f3e5f5; color: #7b1fa2; }
    .route-decompose { background: #fff3e0; color: #e65100; }
    .route-graph { background: #e0f7fa; color: #00695c; }
</style>
""", unsafe_allow_html=True)


# === 初始化（缓存） ===
@st.cache_resource
def init_engine() -> AdaptiveQueryEngine:
    """初始化检索引擎（首次加载模型需要较长时间）"""
    with st.spinner("加载 Embedding 模型..."):
        embedder = BGEEmbedder()
    with st.spinner("连接向量数据库..."):
        store = VectorStore()
    with st.spinner("加载 Reranker..."):
        reranker = Reranker()

    retriever = HybridRetriever(
        embedder=embedder,
        vector_store=store,
        reranker=reranker,
        graph_retriever=None,  # 需要 Neo4j 启动后手动配置
    )
    engine = AdaptiveQueryEngine(retriever)
    return engine


@st.cache_resource
def get_vector_store() -> VectorStore:
    return VectorStore()


# === 主界面 ===
st.title("⛏️ 地质大模型 Demo")
st.caption("KG + RAG + GraphRAG — 江西省金银铜多金属找矿知识库")

# 侧边栏
with st.sidebar:
    st.header("系统状态")
    store = get_vector_store()
    st.metric("向量库文档数", store.count)
    st.markdown("---")
    st.header("使用指南")
    st.markdown("""
1. 先运行 `scripts/01_parse_docs.py` 解析文档
2. 再运行 `scripts/02_build_vector.py` 建立索引
3. （可选）`scripts/03_build_kg.py` 构建知识图谱
4. 在此页面输入问题获得答案
    """)

# 输入区
question = st.text_input(
    "请输入地质问题",
    placeholder="例如：德兴铜矿的成矿时代是什么？",
    key="query_input",
)

col1, col2 = st.columns([1, 5])
with col1:
    submit_btn = st.button("查询", type="primary", use_container_width=True)
with col2:
    st.markdown(
        "<div style='padding-top:32px; color:#666;'>"
        "支持单跳事实、数值查询、多跳推理、对比分析、综合总结等类型"
        "</div>",
        unsafe_allow_html=True,
    )

# === 执行查询 ===
if submit_btn and question.strip():
    engine = init_engine()

    if store.count == 0:
        st.warning("向量库为空，请先运行数据管线建立索引。")
        st.stop()

    with st.spinner("正在检索和生成答案..."):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result: QueryResult = loop.run_until_complete(engine.query(question))
        finally:
            loop.close()

    # 显示路由信息
    st.markdown(f"""
    **检索策略:** <span class="route-tag route-{result.route}">{result.route.upper()}</span>
    """, unsafe_allow_html=True)

    if len(result.sub_queries) > 1:
        with st.expander(f"查询变换 ({len(result.sub_queries)} 条检索词)"):
            for i, q in enumerate(result.sub_queries, 1):
                st.text(f"{i}. {q}")

    # 答案
    st.markdown("---")
    st.subheader("回答")
    st.markdown(result.answer)

    # 来源
    st.markdown("---")
    st.subheader("参考来源")
    for i, chunk in enumerate(result.source_chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", "未知")
        doc_type = meta.get("doc_type", "")
        score = chunk.get("rerank_score", chunk.get("rrf_score", chunk.get("score", 0)))

        with st.expander(f"[{i}] {source} — 相关度 {score:.3f}"):
            st.text(chunk["text"][:500])
            if meta.get("location"):
                st.caption(f"矿区: {meta['location']} | 类型: {doc_type}")

elif submit_btn:
    st.error("请输入问题后再点击查询。")

# 历史对话（简单版）
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
