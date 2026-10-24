# 地质大模型 Demo

KG + RAG + GraphRAG — 江西省金银铜多金属找矿知识库

## 项目进度

截至 2026-10-24。

### 已完成

| 模块 | 状态 | 说明 |
|---|---|---|
| 技术选型与规格 | 完成 | 两份规格文档合并为最终版（上级目录 HTML） |
| 项目骨架 | 完成 | 目录结构、配置模块、`.env` 加载 |
| 文档解析 + 切块 | 完成 | 8 份文档 → 83 个 chunk（正文 75 + 表格 8），17 条 chunker 测试通过 |
| 向量化模块 | 完成 | bge-m3（1024 维）+ Chroma 入库 |
| Reranker | 完成 | bge-reranker-v2-m3，Top-20 → 精排 Top-5 |
| 混合检索 | 完成 | 向量 + 稀疏 + RRF 融合 |
| Query 工程 | 完成 | 自适应路由（direct / multi_query / decompose / hyde） |
| 端到端问答 | 完成 | 检索 → rerank → DeepSeek 生成，带引用编号 |
| Streamlit 界面 | 完成 | http://localhost:8501，已实测可问答 |
| 评测模块 | 完成 | RAGAS 四维指标 + 本地 bge-m3 embedding 适配 |
| RAGAS 全量评测 | 完成 | 43 条用例，3/4 指标达标，详见「评测结果」 |
| GPU 环境 | 完成 | torch 2.5.1+cu121，RTX 4060 可用 |
| 模型权重 | 完成 | bge-m3 + reranker 本地缓存，离线加载 |

### 进行中

无。

### 未开始

| 任务 | 阻塞原因 |
|---|---|
| 知识图谱（LightRAG + Neo4j） | 本机未安装 Neo4j / Docker / JDK |
| 语料扩充 | 需从全国地质资料馆 / 中国地质图书馆调取真实报告 |
| MinerU 高精度 PDF 解析 | 未安装（+4-6 GB），当前用 pymupdf 够用 |

## 快速开始

所有依赖装在项目文件夹内，不污染系统环境。

```bat
:: 1. 进入项目目录
cd /d D:\RAG\geology-rag-demo

:: 2. 安装依赖（已有环境可跳过）
setup_env.bat GPU

:: 3. 编辑 .env（API Key 当前走 Sub2API 中转）
notepad .env

:: 4. 建索引（模型已下载，跳过下载步骤）
.conda-env\python.exe init_demo.py

:: 5. 启动界面
.conda-env\python.exe -m streamlit run src\app\streamlit_app.py
```

浏览器打开 http://localhost:8501。首次查询约 40 秒（加载模型到显存），之后正常。

## 当前环境快照

| 组件 | 版本 | 备注 |
|---|---|---|
| Python | 3.10 | 项目内 `.conda-env` |
| torch | 2.5.1+cu121 | cu121 是上限，决定整条版本链 |
| transformers | 4.57.6 | 必须 <5（5.x 要求 torch≥2.6） |
| sentence-transformers | 5.7.0 | 必须 <6 |
| FlagEmbedding | 1.4.2 | |
| chromadb | 1.5.9 | |
| ragas | 0.3.2 | 必须锁这个版本 |
| datasets | 3.6.0 | 必须 <4 |
| huggingface-hub | 0.36.2 | 必须 <1.0 |
| openai | 1.109.1 | |
| langchain-openai | 1.1.9 | |

**版本约束链的原因**：cu121 的 torch 最高只到 2.5.1，而 transformers 5.x 要求 torch≥2.6。放开任何一个上限都会把 torch 顶到需要 cu124+ 的版本，然后 GPU 全挂。升级前先读这条链。

## 已知问题与修复记录

| 问题 | 根因 | 修复 |
|---|---|---|
| FlagEmbedding 导入失败 `NameError: name 'nn' is not defined` | transformers 5.x 与 torch 2.5.1 不兼容 | requirements.txt 锁 `transformers<5` 等整条链 |
| bge-m3 加载报 `require torch>=2.6` | 官方只提供 `pytorch_model.bin`，新版 transformers 因 CVE-2025-32434 禁止 torch 2.5.1 加载 .bin | 本地把 .bin 转成 `model.safetensors`（2.27 GB，391 tensor），走 safetensors 加载路径 |
| 加载时 `LocalEntryNotFoundError` | FlagEmbedding 内部调 `snapshot_download` 联网校验元数据，网络超时 | `config/settings.py` 默认 `HF_HUB_OFFLINE=1` |
| 控制台 `UnicodeEncodeError: gbk` | Windows CMD 默认 GBK，脚本输出含 ✓/✗ | 运行前 `set PYTHONIOENCODING=utf-8` |
| DeepSeek 401 | `.env` 里是占位符 | 接入 Sub2API 中转（OpenAI 兼容），模型名 `deepseek-v4.1-flash` |

## 模型下载

模型缓存位置：`models\hf-cache\`（约 4.6 GB）。已下载完成，正常情况下不需要重下。

需要重新下载时（先设 `HF_HUB_OFFLINE=0`）：

```cmd
.conda-env\python.exe -c "import os; os.environ['HF_ENDPOINT']='https://hf-mirror.com'; os.environ['HF_HOME']='D:/RAG/geology-rag-demo/models/hf-cache'; os.environ['HF_HUB_OFFLINE']='0'; from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-m3', ignore_patterns=['*.DS_Store','imgs/*']); snapshot_download('BAAI/bge-reranker-v2-m3', ignore_patterns=['*.DS_Store','imgs/*'])"
```

注意 bge-m3 下载后还需要转 safetensors（见上表修复记录），否则在 torch 2.5.1 下加载失败。

## 项目结构

```
geology-rag-demo/
├── config/settings.py         # 全局配置 + 地质术语保护表
├── data/raw/                  # 原始文档 (8 份)
├── data/parsed/               # 解析后 JSON
├── data/chunks/               # 切块输出（83 chunk）
├── models/hf-cache/           # 模型权重（自动生成，勿提交）
├── output/chroma_db/          # 向量库（83 条）
├── output/lightrag_data/      # 知识图谱数据（未生成）
├── .conda-env/                # 项目本地 Python 环境
├── src/
│   ├── data_processing/       # 文档解析 + 切块 + 元数据
│   ├── vectorization/         # bge-m3 + Chroma + Reranker
│   ├── knowledge_graph/       # Schema + 三元组抽取 + Neo4j
│   ├── retrieval/             # 混合检索 + RRF + 自适应路由
│   ├── query_engineering/     # Multi-Query / HyDE / Step-back / Decompose
│   ├── prompt/                # 所有 Prompt 模板
│   ├── evaluation/            # RAGAS 评测 + 地质测试集
│   └── app/streamlit_app.py   # Web 问答界面
├── scripts/
│   ├── 01_parse_docs.py       # 解析 + 切块
│   ├── 02_build_vector.py     # 向量化入库
│   ├── 03_build_kg.py         # 构建知识图谱（需 Neo4j）
│   ├── 04_test_retrieval.py   # 终端交互式测试
│   ├── 05_run_eval.py         # RAGAS 评测
│   └── 06_check_coverage.py   # 语料/测试集覆盖率校验
├── tests/test_chunker.py      # 切块器行为测试（17 条）
├── init_demo.py               # 一键：解析 → 切块 → 向量化 → 验证
├── setup_env.bat              # 环境安装
├── activate_env.bat           # 激活环境
└── requirements.txt           # 核心依赖（含版本上限）
```

## 语料与测试集

`data/raw/` 预置 8 份文档，合计约 1.76 万字符，切块后 83 个 chunk（正文 75 + 表格 8）：

| 文档 | 主要实体 | 数据出处 |
|---|---|---|
| 德兴铜矿区域地质报告 | 德兴铜矿、双桥山群、富家坞/铜厂山/朱砂红 | 公开文献 + 部分演示数值 |
| 庐枞盆地火山岩型矿床概述 | 庐枞盆地、梅山铁矿、罗河金矿 | 公开文献 + 部分演示数值 |
| 武山铜矿地质特征 | 武山铜矿、黄龙组、九瑞矿集区 | 丁昕等 2005；矿床地质 2015 等 |
| 银山铅锌矿地质概述 | 银山铅锌矿、德乐火山盆地 | 余明刚等 2022；岩石学报 2003 等 |
| 相山铀矿田区域地质 | 相山铀矿田、碎斑熔岩、鹅湖岭组 | 杨水源等 2010；地质学报 2017/2020 等 |
| 西华山钨矿地质特征 | 西华山钨矿、五层楼分带 | 郭小飞等 2022 |
| 永平铜矿地质特征 | 永平铜矿、藕塘底组、周潭群 | 田明君等 2019；丁昕等 2005 |
| 江西金银铜多金属成矿规律与勘查技术综述 | 跨文档归纳层，支撑综合类问题 | 汇总上述各篇 |

**这些文档不是原始勘查报告**，是为验证 RAG 链路而编写的合成文档：每份开头都有数据声明，具体数值尽量取自文末列出的公开文献。真正的区域地质调查与勘查报告需从全国地质资料馆（ngac.cn）或中国地质图书馆（cgl.org.cn）调取。

`src/evaluation/test_set.py` 有 43 条用例（单跳 12 / 数值 12 / 多跳 8 / 对比 6 / 综合 5）。跑评测前建议先校验对齐：

```bat
.conda-env\python.exe scripts\06_check_coverage.py
```

该脚本检查两件事：测试集问的实体语料里是否真的有，以及每份语料是否至少被一条用例覆盖。

**已知局限**：83 个 chunk 的语料规模下，`SIMILARITY_TOP_K=20` 仍会召回约四分之一的语料，Context Precision 的区分度偏弱；语料扩到 500+ chunk 后评测指标才有实际意义。

## 评测

```bat
:: 全量 43 条（约 20-30 分钟，消耗 API 额度）
.conda-env\python.exe scripts\05_run_eval.py

:: 先跑 2 条验证链路
.conda-env\python.exe scripts\05_run_eval.py --sample 2
```

结果保存到 `output/eval_<时间戳>.json`。

### 评测结果（2026-10-24，43 条全量）

| 指标 | 分数 | 目标 | 判定 |
|---|---|---|---|
| Faithfulness | 0.876 | ≥0.85 | 达标 |
| Answer Relevancy | 0.914 | ≥0.80 | 达标 |
| Context Precision | 0.857 | ≥0.75 | 达标 |
| Context Recall | 0.599 | ≥0.70 | **未达标** |

**Context Recall 为什么低（已诊断）**：

43 条中有 11 条 recall=0、13 条 <0.5。逐条排查后确认**不是检索召回的锅**：

- 语料覆盖完整（`06_check_coverage.py` 全部通过）
- 11 条失败用例里，最匹配标准答案的 chunk 有 4 条在 reranker 排序中排第 1、2 条排第 3-4
- 但 RAGAS 会把标准答案拆成多个 claim 逐一核对，答案信息分散在 2-3 个 chunk（例如「矿床类型」和「地理位置」各在一个 chunk），Top-5 凑不齐所有 claim，单条就是 0 分
- 把 `RERANK_TOP_N` 从 5 提到 8 重测这 11 条，平均 recall 只从 0 升到 0.121，收益有限且会拉低 Precision

**结论**：这是 chunk 粒度 + RAGAS claim 级判定共同导致的结构性问题，不是调参能解决的。真正的解法是减小 `CHUNK_SIZE` 让单个 chunk 信息更聚焦，或扩充语料后重做 chunk 策略。在 83 chunk 规模下，0.599 属于预期范围，不必据此判断系统可用性——另外三个指标都达标，且 top-1 检索命中率高。

**评测注意事项**：
- `answer_relevancy` 需要 embedding，脚本已配本地 bge-m3 适配（`_BGEM3Embeddings`），不依赖中转站的 embedding 接口
- 2 条样本的分数方差极大（当时 Recall 显示 1.000，全量只有 0.599），没有参考价值，至少跑到 20 条以上

## 知识图谱（未开始）

需要先装 Neo4j。三种方式：

1. **Neo4j Desktop**（推荐）— 自带 JRE，GUI 安装
2. **JDK 17 + Community ZIP** — 免安装，需配 `JAVA_HOME`
3. **Docker** — `docker-compose up -d`（本机未装 Docker）

装好后：

```bat
.conda-env\python.exe scripts\03_build_kg.py --limit 5
```

## 技术栈

| 组件 | 选型 |
|---|---|
| Embedding | bge-m3（1024 维，dense + sparse） |
| 向量库 | Chroma |
| Reranker | bge-reranker-v2-m3 |
| GraphRAG | LightRAG + Neo4j |
| LLM API | DeepSeek（当前走 Sub2API 中转，`deepseek-v4.1-flash`） |
| 文档解析 | pymupdf（MinerU 可选未装） |
| 评测 | RAGAS 四维指标 |

详细设计见上级目录的 `地质大模型Demo技术规格书-最终版.html`。

## 后续待办

按优先级排列。

1. **优化 Context Recall** — 当前 0.599。已排除检索召回和语料覆盖问题，根因是 chunk 粒度过粗导致答案跨块分散。下一步试：减小 `CHUNK_SIZE`（512 → 300）重建索引，或按「问答对」粒度补充 chunk 元数据，再做 A/B 对比。`RERANK_TOP_N` 支持环境变量覆盖，可用于对比实验。
2. **扩充语料** — 当前 83 chunk 太少，评测指标区分度不足。从 ngac.cn / cgl.org.cn 下载图幅说明书和典型矿床报告，目标 500+ chunk。
3. **装 Neo4j，打通 GraphRAG** — 跑 `03_build_kg.py` 抽取三元组，实现多跳推理。注意：11 条 recall=0 里有 5 条是对比/综合类，这类问题正是 GraphRAG 的目标场景。
4. **界面增强** — 目前只支持单轮问答，缺对话历史、流式输出、检索参数调节面板。
5. **Query 工程实测调优** — 自适应路由目前靠 LLM 分类，需要验证多跳/对比类问题是否真的走了 decompose 分支且效果更好。
6. **写项目文档** — 大创中期检查需要：技术路线图、实验对比数据、系统演示录屏。
