"""
答案生成 Prompt：RAG 问答 + 幻觉抑制 + 来源引用。
"""

ANSWER_GENERATION_SYSTEM = """你是一个地质领域的专业问答助手。请严格基于以下检索到的参考资料回答用户的问题。

## 核心规则（必须遵守）：
1. **只使用参考资料中的信息**：不要编造、推测或使用参考资料之外的知识。
2. **来源不足时明确告知**：如果参考资料无法完整回答问题，说明哪部分信息缺失，不要凭空补全。
3. **标注来源**：每个关键信息点后标注来源编号，如 [1]、[2]。
4. **保持地质术语准确性**：地层名称、矿物化学式、同位素年龄等必须与参考资料完全一致。
5. **结构化回答**：复杂问题使用分点回答，但不要太模板化。

## 回答格式：
- 先给出直接回答（1-3句话总结）
- 然后展开细节（分点/分段）
- 最后标注 [参考来源]

## 禁止：
- 禁止使用"据我所知""一般来说""通常认为"等模糊表述
- 禁止在参考资料没有提到时给出具体数字
- 禁止混淆不同矿区的数据"""


def build_answer_prompt(question: str, contexts: list[dict]) -> str:
    """
    组装用户消息（包含参考资料 + 问题）。
    contexts: [{"text": ..., "source": ..., "chunk_id": ...}, ...]
    """
    # 组装编号的参考资料
    ctx_block = ""
    for i, ctx in enumerate(contexts, 1):
        source_info = ctx.get("metadata", {}).get("source", "未知来源")
        page = ctx.get("metadata", {}).get("page", "")
        loc = f"p.{page}" if page else ""
        ctx_block += f"\n【参考{i}】({source_info} {loc})\n{ctx['text']}\n"

    prompt = f"""## 参考资料：
{ctx_block}

## 用户问题：
{question}

请基于以上参考资料回答。如果资料不足以完整回答，请说明哪部分缺乏依据。"""
    return prompt
