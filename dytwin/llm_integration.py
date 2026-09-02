"""LLM集成模块

统一管理对LLM的调用，处理重试、解析等。
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Type, Callable

import llm_call  # 用户提供的LLM调用接口

logger = logging.getLogger(__name__)
T = TypeVar('T')


def extract_json_from_response(text: str) -> str:
    """从LLM响应中提取JSON内容，处理markdown代码块包裹的情况。"""
    # 尝试匹配 ```json ... ``` 或 ``` ... ```
    pattern = r'```(?:json)?\s*([\s\S]*?)```'
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    # 如果没有代码块，直接返回原文本
    return text.strip()


def smart_parse_json(text: str) -> dict:
    """智能解析JSON，处理各种LLM返回的非标准格式。
    
    支持处理的问题：
    1. markdown代码块包裹
    2. 字段值缺少引号
    3. 中文冒号替换
    4. 尾部逗号
    5. 单引号替换为双引号
    6. 缺少逗号（如 ]\n"key" 应为 ],\n"key"）
    7. JSON不完整（尝试补全括号）
    """
    # 第一步：提取JSON内容（去除markdown代码块）
    json_str = extract_json_from_response(text)
    
    # 第二步：尝试直接解析
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    # 第三步：尝试修复常见问题
    fixed_str = json_str
    
    # 替换中文冒号为英文冒号
    fixed_str = fixed_str.replace('：', ':')
    
    # 替换单引号为双引号
    fixed_str = fixed_str.replace("'", '"')
    
    # 修复缺少逗号的情况：] 后面直接跟 " 或 {
    # 例如: ]\n    "remove" 应该是 ],\n    "remove"
    fixed_str = re.sub(r'(\])\s*\n(\s*")', r'\1,\n\2', fixed_str)
    fixed_str = re.sub(r'(\})\s*\n(\s*")', r'\1,\n\2', fixed_str)
    fixed_str = re.sub(r'(")\s*\n(\s*")', r'\1,\n\2', fixed_str)
    
    # 移除尾部逗号（在 } 或 ] 之前的逗号）
    fixed_str = re.sub(r',\s*([}\]])', r'\1', fixed_str)
    
    # 尝试解析修复后的JSON
    try:
        return json.loads(fixed_str)
    except json.JSONDecodeError:
        pass
    
    # 第四步：尝试修复未加引号的字符串值
    # 匹配 "key": 后面跟着非引号开头的值（直到逗号或}）
    def fix_unquoted_values(match):
        key = match.group(1)
        value = match.group(2).strip()
        # 如果值是 true/false/null 或数字，保持原样
        if value.lower() in ('true', 'false', 'null') or re.match(r'^-?\d+\.?\d*$', value):
            return f'"{key}": {value}'
        # 否则给值加上引号，并转义内部的引号
        value = value.replace('"', '\\"')
        return f'"{key}": "{value}"'
    
    # 匹配 "key": value 模式，其中value不是以引号、{、[开头
    fixed_str = re.sub(
        r'"(\w+)":\s*([^"\[\{][^,}\]]*?)(?=[,}\]])',
        fix_unquoted_values,
        fixed_str
    )
    
    try:
        return json.loads(fixed_str)
    except json.JSONDecodeError:
        pass
    
    # 第五步：尝试提取JSON对象（从第一个{到最后一个}）
    brace_match = re.search(r'\{[\s\S]*\}', text)
    if brace_match:
        try:
            return json.loads(brace_match.group())
        except json.JSONDecodeError:
            pass
    
    # 第六步：尝试补全不完整的JSON（缺少闭合括号）
    incomplete_str = fixed_str
    # 统计未闭合的括号
    open_braces = incomplete_str.count('{') - incomplete_str.count('}')
    open_brackets = incomplete_str.count('[') - incomplete_str.count(']')
    
    if open_braces > 0 or open_brackets > 0:
        # 补全缺少的闭合括号
        incomplete_str = incomplete_str.rstrip()
        # 移除可能的尾部逗号
        incomplete_str = re.sub(r',\s*$', '', incomplete_str)
        # 添加缺少的闭合括号
        incomplete_str += ']' * open_brackets + '}' * open_braces
        try:
            return json.loads(incomplete_str)
        except json.JSONDecodeError:
            pass
    
    # 所有尝试都失败，抛出异常
    raise json.JSONDecodeError(f"无法解析JSON", text, 0)


def call_llm_with_retry(
    prompt: str,
    max_retries: int = 5,
    output_type: Type[T] = dict,
    parser: Optional[Callable[[str], T]] = None,
    temperature: float = 0.2,
) -> T:
    """调用LLM并解析输出，支持重试。
    
    Args:
        prompt: 提示词
        max_retries: 最大重试次数
        output_type: 期望的输出类型
        parser: 自定义解析函数，优先级高于output_type
        temperature: 温度参数，控制随机性（默认0.5，较低以提高一致性）
    
    Returns:
        解析后的输出
    """
    last_error = None
    
    for attempt in range(max_retries):
        try:
            # 调用LLM（使用较低temperature提高一致性）
            response = llm_call.generate(prompt, temperature=temperature)
            
            # 使用自定义解析器或默认JSON解析
            if parser is not None:
                return parser(response)
            
            # 尝试解析JSON
            if output_type is dict or output_type is None:
                # 使用智能JSON解析（处理各种非标准格式）
                return smart_parse_json(response)
            else:
                return output_type(response)
                
        except json.JSONDecodeError as e:
            last_error = f"JSON解析失败: {e}"
            logger.warning(f"LLM返回非JSON格式 (尝试 {attempt + 1}/{max_retries}), 长度={len(response)}字符")
        except Exception as e:
            last_error = str(e)
            logger.warning(f"LLM调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
    
    # 所有重试都失败
    error_msg = f"LLM调用失败，已达到最大重试次数 {max_retries}: {last_error}"
    logger.error(error_msg)
    raise RuntimeError(error_msg)


def reflect_and_improve_profile(
    simulation_time: str,
    root_post: Dict[str, Any],
    current_profile: str,
    memories: List[Tuple[str, float]],
    few_shot_examples: List[Tuple[str, str]],
    predicted_forward_text: str,
    pred_rationale: str,
    true_forward_text: str,
    llm_score_rationale: str,
) -> Dict[str, Any]:
    """反思预测结果，返回画像改进建议。
    
    Args:
        simulation_time: 当前模拟的时间
        root_post: 原微博数据
        current_profile: 当前用户画像（多行文本格式）
        memories: 检索到的相关记忆列表（长期记忆）
        few_shot_examples: few-shot样例（短期记忆）
        predicted_forward_text: 预测转发文本
        pred_rationale: 预测的理由
        true_forward_text: 真实转发文本
        llm_score_rationale: LLM五维度评分的详细理由
    
    Returns:
        包含reflection、improvement_suggestion和profile_patch的字典
    """
    from .llm_prompts import prompt_reflect_and_update_profile
    
    # 检查画像长度，如果超过30行则先进行压缩
    profile_lines = [line.strip() for line in current_profile.split('\n') if line.strip()]
    if len(profile_lines) > 30:
        print(f"  ⚠️ 画像长度超过阈值（{len(profile_lines)}行 > 30行），正在进行自动压缩...")
        try:
            compressed_profile = merge_profile_sentences(current_profile)
            compressed_lines = [line.strip() for line in compressed_profile.split('\n') if line.strip()]
            print(f"  ✅ 画像压缩完成：{len(profile_lines)}行 -> {len(compressed_lines)}行")
            current_profile = compressed_profile
        except Exception as e:
            print(f"  ❌ 画像压缩失败：{str(e)}，使用原始画像")
    
    prompt = prompt_reflect_and_update_profile(
        simulation_time, root_post, current_profile, memories,
        few_shot_examples, predicted_forward_text, pred_rationale,
        true_forward_text, llm_score_rationale
    )
    result = call_llm_with_retry(prompt)
    return result


def merge_profile_sentences(current_profile: str) -> str:
    """归并画像中相似的描述句。
    
    Args:
        current_profile: 当前用户画像（多行文本格式）
    
    Returns:
        归并后的用户画像（多行文本格式）
    """
    from .llm_prompts import prompt_merge_profile
    
    prompt = prompt_merge_profile(current_profile)
    # LLM直接返回压缩后的画像文本，不是JSON格式
    compressed_profile = call_llm_with_retry(prompt, output_type=str)
    
    # 清理输出，确保格式正确
    if isinstance(compressed_profile, str):
        # 移除可能的前后缀文本，保留核心画像内容
        lines = [line.strip() for line in compressed_profile.split('\n') if line.strip()]
        # 过滤掉可能的提示性文本
        filtered_lines = []
        for line in lines:
            if not any(keyword in line for keyword in ['请输出', '压缩后', '用户画像', '以下是', '结果']):
                filtered_lines.append(line)
        return '\n'.join(filtered_lines)
    else:
        # 如果返回的不是字符串，尝试从字典中提取
        if isinstance(compressed_profile, dict) and 'merged_sentences' in compressed_profile:
            return '\n'.join(compressed_profile['merged_sentences'])
        else:
            # 如果无法解析，返回原始画像
            return current_profile


def predict_forward_text_with_focus(
    root_post: Dict[str, Any],
    current_profile: str,
    memories: List[Tuple[str, float]],
    few_shot_examples: List[Tuple[str, str]] = None,
    forward_time: str = None,
) -> Dict[str, Any]:
    """预测用户转发内容，重点关注与当前任务相关的画像描述。
    
    Args:
        root_post: 原微博数据
        current_profile: 当前用户画像（多行文本格式）
        memories: 检索到的相关记忆列表
        few_shot_examples: 用户历史转发样例列表，每个元素为(原微博摘要, 用户转发语)
        forward_time: 转发时间（用户发表转发的时间点）
    
    Returns:
        包含relevant_traits, predicted_forward_text, rationale的字典
    """
    from .llm_prompts import prompt_predict_forward_with_focus
    
    prompt = prompt_predict_forward_with_focus(root_post, current_profile, memories, few_shot_examples, forward_time)
    result = call_llm_with_retry(prompt)
    return result


def llm_similarity_score(text1: str, text2: str) -> Dict[str, Any]:
    """使用LLM对两段文本进行多维度相似度打分。
    
    Args:
        text1: 第一段文本
        text2: 第二段文本
    
    Returns:
        包含各维度分数和平均分的字典：
        {
            "semantic_similarity": float,
            "emotion_similarity": float,
            "stance_similarity": float,
            "style_similarity": float,
            "focus_similarity": float,
            "average_score": float,
            "rationale": str
        }
    """
    from .llm_prompts import prompt_llm_similarity_score
    
    prompt = prompt_llm_similarity_score(text1, text2)
    result = call_llm_with_retry(prompt)
    
    # 计算平均分
    score_keys = ["semantic_similarity", "emotion_similarity", "stance_similarity", 
                  "style_similarity", "focus_similarity"]
    scores = []
    for key in score_keys:
        val = result.get(key)
        if isinstance(val, (int, float)):
            scores.append(float(val))
    
    if scores:
        result["average_score"] = sum(scores) / len(scores)
    else:
        result["average_score"] = 0.0
    
    # 合并五个维度的rationale为一个统一字段
    rationale_parts = []
    rationale_keys = [
        ("semantic_rationale", "【语义相似度】"),
        ("emotion_rationale", "【情感倾向】"),
        ("stance_rationale", "【立场观点】"),
        ("style_rationale", "【表达风格】"),
        ("focus_rationale", "【关注焦点】"),
    ]
    for key, prefix in rationale_keys:
        if key in result and result[key]:
            rationale_parts.append(f"{prefix}{result[key]}")
    result["rationale"] = "；".join(rationale_parts) if rationale_parts else ""
    
    return result

