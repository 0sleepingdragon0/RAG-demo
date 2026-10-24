"""
地质大模型 Demo — 全局配置
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 项目根目录（先算出来，供后面的缓存路径使用）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 先加载 .env，让其中的配置优先于下面的默认值
load_dotenv()

# === 模型缓存固定在项目内部，避免写入 C 盘默认位置 ===
# 必须在任何 import FlagEmbedding / transformers 之前设置。
# 用 setdefault，所以 .env、activate_env.bat 或系统环境变量都能覆盖。
_MODEL_CACHE = str(PROJECT_ROOT / "models" / "hf-cache")
os.environ.setdefault("HF_HOME", _MODEL_CACHE)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(PROJECT_ROOT / "models" / "hf-cache" / "hub"))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / "models" / "torch"))
# 默认走国内镜像（需 VPN 关闭）；想用官方源在 .env 里设 HF_ENDPOINT=https://huggingface.co
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 模型权重已下载到本地缓存，默认离线加载，避免 FlagEmbedding 内部
# snapshot_download 联网校验元数据时因网络超时导致加载失败。
# 需要下载新模型时，在 .env 里设 HF_HUB_OFFLINE=0。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
Path(_MODEL_CACHE).mkdir(parents=True, exist_ok=True)

# === 路径 ===
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PARSED_DIR = DATA_DIR / "parsed"
CHUNKS_DIR = DATA_DIR / "chunks"
OUTPUT_DIR = PROJECT_ROOT / "output"
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", OUTPUT_DIR / "chroma_db"))
LIGHTRAG_DIR = Path(os.getenv("LIGHTRAG_DIR", OUTPUT_DIR / "lightrag_data"))

# === LLM API ===
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")

# === Neo4j ===
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "geology_demo_2026")

# === Embedding 模型 ===
EMBED_MODEL_NAME = "BAAI/bge-m3"
EMBED_DIM = 1024
EMBED_MAX_LENGTH = 8192
EMBED_USE_FP16 = True  # RTX 4060 可用 fp16

# === Reranker 模型 ===
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))

# === 检索参数 ===
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
SIMILARITY_TOP_K = 20
RRF_K = 60

# === Chroma Collection ===
CHROMA_COLLECTION = "geological_docs"

# === 切块保护术语（不被从中间切断）===
GEO_TERMS = [
    "矽卡岩型铜矿床", "花岗闪长斑岩", "双桥山群",
    "SHRIMP锆石U-Pb年龄", "Re-Os同位素等时线",
    "燕山早期", "燕山晚期", "印支期", "华力西期",
    "震旦系下统", "志留系中统", "泥盆系", "石炭系", "二叠系上统",
    "钦杭成矿带", "长江中下游成矿带",
    "绢云母化", "绿泥石化", "高岭土化", "硅化",
    "斑岩型", "热液脉型", "矽卡岩型", "火山岩型", "沉积变质型",
    "玢岩铁矿", "铜金矿", "铅锌矿", "银多金属矿",
    "千枚岩", "片岩", "变粒岩", "板岩",
    "黄铜矿", "斑铜矿", "辉铜矿", "自然金", "闪锌矿", "方铅矿",
    "矿化分带", "蚀变分带", "品位变化",
    # --- 2026-10-24 补齐语料后新增 ---
    "流纹英安岩", "碎斑熔岩", "花岗斑岩", "黑云母花岗岩", "石英二长斑岩",
    "流纹质石英斑岩", "英安斑岩", "闪长玢岩", "安粗岩", "粗安岩",
    "黑钨矿", "白钨矿", "锡石", "辉钼矿", "黄铁矿", "黝铜矿", "沥青铀矿",
    "石英脉型黑钨矿", "层状矽卡岩型", "接触带矽卡岩型", "层控硫化物型",
    "斑岩型铀矿", "火山岩型铀矿", "浅成低温热液型", "玢岩型铁矿",
    "云英岩化", "大理岩化", "矽卡岩化", "青磐岩化", "钠交代", "钾交代",
    "石榴子石矽卡岩", "透辉石矽卡岩", "硅灰石矽卡岩",
    "鹅湖岭组", "双峰岭组", "打鼓顶组", "武夷群", "藕塘底组", "叶家湾组",
    "周潭群", "黄龙组", "茅口组", "五通组", "坟头组", "茅山组",
    "德兴铜矿", "武山铜矿", "银山铅锌矿", "永平铜矿", "西华山钨矿",
    "相山铀矿田", "庐枞盆地", "梅山铁矿", "罗河金矿", "富家坞", "铜厂山",
    "朱砂红", "邹家山", "火烧岗", "赣东北深断裂", "十杭裂谷带",
    "郯庐断裂带", "江山—绍兴断裂带", "萍乡—广丰深断裂", "钦杭结合带",
    "江南造山带", "扬子陆块", "华夏地块", "华南板块",
    "埃达克质", "壳源重熔", "成岩成矿时差", "五层楼", "三层楼",
    "同位素地质年代学", "LA-ICP-MS", "锆石U-Pb", "裂变径迹",
    "高极化率异常", "土壤地球化学测量", "激电法",
]
