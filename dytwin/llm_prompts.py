"""集中管理提示词模板

这里不追求完美prompt，而是提供可迭代的模板：
- 转发预测 -> 生成转发文本（JSON）
- 反思改进 -> 给出画像改进建议（JSON）
- LLM评分 -> 多维度相似度打分（JSON）
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple


def prompt_reflect_and_update_profile(
    simulation_time: str,
    root_post: Dict[str, Any],
    current_profile: str,
    memories: List[Tuple[str, float]],
    few_shot_examples: List[Tuple[str, str]],
    predicted_forward_text: str,
    pred_rationale: str,
    true_forward_text: str,
    llm_score_rationale: str,
) -> str:
    """生成画像更新的提示词
    
    Args:
        simulation_time: 当前模拟的时间
        root_post: 原博文内容
        current_profile: 当前用户画像
        memories: 检索到的用户历史记忆（长期记忆）
        few_shot_examples: few-shot样例（短期记忆）
        predicted_forward_text: 预测的转发文本
        pred_rationale: 预测的理由
        true_forward_text: 真实转发文本
        llm_score_rationale: LLM五维度评分的详细理由
    """
    mem_str = "\n".join([f"- {text}" for text, score in memories]) or "- 无"
    profile_str = current_profile if current_profile else "（当前画像为空）"
    
    # 分析画像相似性
    profile_lines = [line.strip() for line in profile_str.split('\n') if line.strip()] if profile_str != "（当前画像为空）" else []
    profile_line_count = len(profile_lines)
    similarity_analysis = _analyze_profile_similarity(profile_lines)
    
    # 格式化few-shot样例
    if few_shot_examples:
        few_shot_str = "\n".join([f"- 原博文摘要: {post[:50]}... → 用户转发: {fwd}" 
                                  for post, fwd in few_shot_examples])
    else:
        few_shot_str = "- 无"

    return f"""你是一个用户画像反思与改进助手。

任务：基于预测结果和专家评估反馈，反思预测流程，分析预测与真实转发的差异原因，并据此改进用户画像以提升未来预测准确率。

【当前模拟时间】
{simulation_time}

【最重要原则】预测成功时不更新画像：
- 用户画像存在的唯一目的是帮助更准确地预测用户转发语
- 当预测转发与真实转发高度一致时（仅有标点符号、表情符号数量等微小差异），说明当前画像已经足够好
- 此时应返回空的profile_patch，不做任何画像修改

【画像更新操作顺序与隔离原则】：
1) **操作顺序**：严格按照 merge → modify → add → remove 的顺序进行
2) **操作隔离**：同一条画像描述句最多只能被一个操作处理，不允许重复操作
3) **逻辑清晰**：
   - merge：优先合并语义相近的多条描述，简化画像结构
   - modify：对单条描述进行精确修改，提升准确性
   - add：补充画像中缺失的新维度或特征
   - remove：删除明确错误或过时的描述

要求：
1) 仅输出JSON，必须可解析。
2) 返回格式：
   {{
     "reflection": "对预测流程和结果的反思分析：预测时基于什么信息做出了什么判断，为什么与真实结果有差异",
     "improvement_suggestion": "画像改进建议：基于反思，说明如何修改画像能提升未来预测准确率",
     "profile_patch": {{
       "merge": [
         {{"old_sentences": ["画像中的完整旧描述句1", "画像中的完整旧描述句2"], "new_sentence": "归并后的新描述句"}}
       ],
       "modify": {{"画像中的完整旧描述句": "修改后的完整新描述句"}},
       "add": ["具体的用户特征描述句"],
       "remove": ["画像中要删除的完整描述句"]
     }}
   }}

【重要】add字段必须是具体的用户特征描述，而不是抽象的操作指令！
- 错误示例：add: ["新增关于用户转发行为的描述"] ← 这是操作指令，不是画像内容
- 正确示例：add: ["用户关注南京相关的文化新闻和乡愁话题"] ← 这是具体的用户特征
- 正确示例：add: ["用户对00后年轻群体持积极态度"] ← 这是具体的用户特征
- 正确示例：add: ["用户喜欢用简短的表情或感叹词回应热点新闻"] ← 这是具体的用户特征
   
   【预测成功时的返回格式】当预测转发与真实转发高度一致时，返回：
   {{
     "reflection": "预测成功的原因分析",
     "improvement_suggestion": "预测成功，无需更新画像",
     "profile_patch": {{}}
   }}

【关键约束】画像操作必须针对完整描述句：
- 每条画像描述都是独立完整的句子，代表用户某一维度的特质
- modify: 必须提供要修改的【完整旧描述句】（与画像中的某一行完全匹配），替换为新描述句
- merge: old_sentences必须是【完整的多条旧描述句】（每条与画像中的某一行完全匹配），合并为一条新描述
- remove: 必须提供要删除的【完整描述句】（与画像中的某一行完全匹配）
- 禁止对描述句进行拆分或部分匹配

【反思要点】
1) 回顾预测过程：预测时使用了哪些画像特征和记忆信息？预测理由是否合理？
2) 分析专家评估：评分专家和一致性专家分别指出了哪些维度的差异？
3) 定位差异根源：差异是因为画像缺失、画像错误、还是画像未被正确使用？
4) 提出改进方案：如何修改画像能让下次遇到类似情况时预测更准确？

【操作优先级与强制约束】
**基础优先级**：merge > modify > add > remove

**画像长度约束**：
- 画像≤15行：正常操作，但优先考虑merge
- 画像16-25行：**强制要求**每次更新必须包含≥1个merge操作
- 画像26-30行：**禁止add操作**，只允许merge/modify/remove
- 画像>30行：**强制整理**，必须通过merge将画像压缩到25行以内

**操作数量限制**：
- add：最多1条（画像≤25行时），画像>25行时禁止
- merge：画像>8行时每次至少1个，画像>15行时每次至少2个
- remove：每次最多1条，需明确证伪

【重要】关于remove操作的谨慎原则：
- 只有当画像描述被明确证伪（与用户真实行为直接矛盾）时，才应该remove
- 如果只是当前话题未涉及某特征，应保留该描述

================================================================================
【输入信息】
================================================================================

【当前用户画像】（每行一个描述句，共{profile_line_count}行）：
{profile_str}

【画像相似性分析】（请重点关注以下可能需要合并的相似描述组）：
{similarity_analysis}

【原博文内容】：
{json.dumps(root_post, ensure_ascii=False, indent=2, default=str)}

【用户历史记忆】（长期记忆，按相关性检索）：
{mem_str}

【近期转发样例】（短期记忆，最近的转发行为）：
{few_shot_str}

【预测转发文本】：
{predicted_forward_text}

【预测理由】：
{pred_rationale}

【真实转发文本】：
{true_forward_text}

================================================================================
【专家评估反馈】
================================================================================

【评分专家的五维度评估理由】（语义、情感、立场、风格、焦点相似度分析）：
{llm_score_rationale}
"""


def prompt_merge_profile(current_profile: str, merge_threshold: int = 30) -> str:
    """生成画像归并的提示词
    
    当画像描述句数量超过阈值时，调用此函数让LLM归并相似描述
    """
    profile_lines = [line.strip() for line in current_profile.split('\n') if line.strip()]
    current_count = len(profile_lines)
    
    return f"""你是一个用户画像整理专家。

任务：对用户画像进行智能压缩，将语义相似或相关的描述合并为更精炼的表述。

当前画像状态：{current_count}行（超过{merge_threshold}行阈值，需要压缩）

压缩要求：
1) **保留核心信息**：不丢失任何重要的用户特征和行为模式
2) **合并相似内容**：将语义相近、主题相关的多条描述合并为一条
3) **精炼表达**：合并后的描述应简洁但信息完整，避免冗余
4) **保持结构**：每条描述保持独立性，一行一个特征
5) **目标长度**：将画像压缩到20行左右

输出格式：
直接输出压缩后的用户画像，每行一个描述句，不需要JSON格式。

原始用户画像：
{current_profile}

请输出压缩后的用户画像："""


def prompt_predict_forward_with_focus(
    root_post: Dict[str, Any],
    current_profile: str,
    memories: List[Tuple[str, float]],
    few_shot_examples: List[Tuple[str, str]] = None,
    forward_time: str = None,
) -> str:
    """生成转发预测提示词，让LLM重点关注与当前任务相关的画像描述
    
    Args:
        root_post: 原微博数据
        current_profile: 当前用户画像
        memories: 检索到的相关记忆
        few_shot_examples: 用户历史转发样例列表，每个元素为(原微博摘要, 用户转发语)
        forward_time: 转发时间（用户发表转发的时间点）
    """
    mem_str = "\n".join([f"- (score={score:.4f}) {text}" for text, score in memories]) or "- 无"
    profile_str = current_profile if current_profile else "（当前画像为空）"
    
    # 构建few-shot样例（当前模拟数据之前最近的三条转发样例，作为短期记忆）
    examples_str = ""
    if few_shot_examples and len(few_shot_examples) > 0:
        examples_str = "\n【短期记忆】以下是该用户在当前时刻之前最近发表的转发，用于理解预测任务和参考用户近期状态：\n"
        for i, (post_summary, forward_text) in enumerate(few_shot_examples[:3], 1):
            examples_str += f"近期转发{i}:\n  原微博: {post_summary[:100]}...\n  用户转发: {forward_text}\n"
        examples_str += "（注意：短期记忆仅供参考用户近期表达风格和状态，预测内容应主要基于用户画像和长期记忆）\n"
    
    # 构建时间信息
    time_str = ""
    if forward_time:
        time_str = f"\n当前转发时刻：{forward_time}\n（注：这是用户发表转发的时间点，请考虑时间因素对用户表达的影响）\n"

    return f"""你是一个社交媒体用户模拟智能体。

任务：给定用户画像、原微博内容（被转发的微博）、以及检索到的相关记忆，预测该用户在指定时刻的转发文本内容。
{time_str}
**核心依据优先级**：
1. **用户画像**（最重要）：用户的长期稳定特征，是预测的主要依据
2. **长期记忆**（重要）：与当前原微博主题相关的历史行为记录
3. **短期记忆**（参考）：用户最近发表的转发内容，反映用户近期状态和表达风格

**重要**：在预测时，请重点关注用户画像中与当前原微博主题相关的描述，忽略无关的特征描述。
{examples_str}
要求：
1) 首先分析原微博的主题和关键词
2) 从用户画像中筛选出与该主题相关的特征描述
3) 结合长期记忆和短期记忆中的相关信息进行预测
4) 输出JSON，必须可被解析
5) 返回格式：{{"relevant_traits": ["相关的画像特征1", ...], "predicted_forward_text": "...", "rationale": "..."}}
6) predicted_forward_text 要基于用户画像和长期记忆预测，可参考短期记忆中的近期表达风格，但不要凭空杜撰事实

用户画像（每行一个特征描述，请筛选与当前微博相关的）：
{profile_str}

长期记忆（与当前原微博主题相关的历史行为）：
{mem_str}

原微博数据：
{json.dumps(root_post, ensure_ascii=False, indent=2, default=str)}
"""


def prompt_llm_similarity_score(text1: str, text2: str) -> str:
    """生成LLM多维度相似度打分的提示词"""
    return f"""你是一个文本相似度评估专家。

任务：对比两段文本，从多个维度评估它们的相似程度，每个维度打分范围为0-10分（0表示完全不相似，10表示完全相同）。

评估维度：
1) **语义相似度**：两段文本表达的核心意思是否相近
2) **情感倾向**：两段文本的情感色彩（积极/消极/中性）是否一致
3) **立场观点**：两段文本对事件/话题的态度立场是否一致
4) **表达风格**：两段文本的语言风格（正式/口语/讽刺/幽默等）是否相似
5) **关注焦点**：两段文本关注的重点内容是否相同

要求：
1) 仅输出JSON，必须可被解析
2) 返回格式：
   {{
     "semantic_similarity": 0-10,
     "semantic_rationale": "语义相似度的详细分析：文本1表达了...，文本2表达了...，两者在核心意思上...",
     "emotion_similarity": 0-10,
     "emotion_rationale": "情感倾向的详细分析：文本1的情感色彩是...，文本2的情感色彩是...，两者...",
     "stance_similarity": 0-10,
     "stance_rationale": "立场观点的详细分析：文本1对事件的态度是...，文本2对事件的态度是...，两者...",
     "style_similarity": 0-10,
     "style_rationale": "表达风格的详细分析：文本1的语言风格是...，文本2的语言风格是...，两者...",
     "focus_similarity": 0-10,
     "focus_rationale": "关注焦点的详细分析：文本1关注的重点是...，文本2关注的重点是...，两者..."
   }}
3) 每个维度的rationale必须详细说明两段文本在该维度上的具体表现和差异
4) 打分要客观公正，有理有据

文本1：
{text1}

文本2：
{text2}
"""


def _analyze_profile_similarity(profile_lines: List[str]) -> str:
    """分析画像中的相似描述，返回相似性分析结果
    
    Args:
        profile_lines: 画像描述句列表
    
    Returns:
        相似性分析结果字符串
    """
    if len(profile_lines) <= 5:
        return "画像描述较少，暂无明显相似内容需要合并。"
    
    # 定义关键词组，用于识别相似描述
    keyword_groups = {
        "批评态度": ["批评", "批判", "质疑", "反对", "不满"],
        "表达风格": ["讽刺", "幽默", "直接", "委婉", "正式", "口语", "风格"],
        "关注领域": ["政治", "社会", "科技", "医学", "教育", "经济", "文化"],
        "情感倾向": ["愤怒", "支持", "赞同", "反感", "喜爱", "厌恶", "中性"],
        "行为模式": ["转发", "评论", "分享", "引用", "陈述", "分析"],
        "话题类型": ["争议", "热点", "新闻", "事件", "现象", "问题"]
    }
    
    # 按关键词分组
    groups = {}
    for category, keywords in keyword_groups.items():
        matching_lines = []
        for i, line in enumerate(profile_lines):
            if any(keyword in line for keyword in keywords):
                matching_lines.append((i+1, line))
        if len(matching_lines) >= 2:
            groups[category] = matching_lines
    
    # 生成分析结果
    if not groups:
        return "未发现明显的相似描述组，建议重点关注语义相近的表达。"
    
    analysis_parts = []
    for category, lines in groups.items():
        analysis_parts.append(f"**{category}相关描述**（{len(lines)}条，建议合并）：")
        for line_num, line_text in lines:
            analysis_parts.append(f"  {line_num}. {line_text}")
        analysis_parts.append("")
    
    return "\n".join(analysis_parts)

