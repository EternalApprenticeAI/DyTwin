import asyncio
import aiohttp
import itertools
import json
import sys
import threading
import warnings
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

# 抑制 Windows 上 asyncio 的 ProactorEventLoop 警告
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    warnings.filterwarnings("ignore", message=".*Event loop is closed.*")

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from async_stream_manager import SafeClientSession, create_safe_connector, create_safe_timeout, safe_gather
except ImportError:
    # 如果导入失败，使用原生aiohttp
    SafeClientSession = None
    def create_safe_connector(**kwargs):
        return aiohttp.TCPConnector(**kwargs)
    def create_safe_timeout(**kwargs):
        return aiohttp.ClientTimeout(**kwargs)
    async def safe_gather(*tasks, **kwargs):
        return await asyncio.gather(*tasks, **kwargs)

# 全局模型配置
_global_model = "Qwen/Qwen2.5-7B-Instruct"
_global_url = "https://api.siliconflow.cn/v1/chat/completions"
_global_api_key = os.getenv("DYTWIN_API_KEY", "")
_global_seed = None  # 全局随机数种子，用于实验可复现
_global_temperature = 0.2
_global_max_tokens = 1024

# API Key 轮询池（多key并行提升TPM配额）
_api_key_pool: List[str] = []
_api_key_counter = itertools.count()
_api_key_lock = threading.Lock()
_api_key_cooldown: Dict[str, float] = {}  # key -> 解冻时间戳(time.time)

def set_global_model(model: str):
    """设置全局使用的模型"""
    global _global_model
    _global_model = model

def get_global_model() -> str:
    """获取全局使用的模型"""
    return _global_model

def set_global_url(url: str):
    """设置全局使用的 API URL"""
    global _global_url
    _global_url = url

def get_global_url() -> str:
    """获取全局使用的 API URL"""
    return _global_url

def set_global_api_key(api_key: str):
    """设置全局使用的 API Key"""
    global _global_api_key
    _global_api_key = api_key

def get_global_api_key() -> str:
    """获取全局使用的 API Key"""
    return _global_api_key

def set_global_api_keys(api_keys: List[str]):
    """设置API Key轮询池（多个key轮流使用，成倍提升TPM配额）"""
    global _api_key_pool
    _api_key_pool = [k.strip() for k in api_keys if k.strip()]
    if _api_key_pool:
        print(f"[API Key池] 已加载 {len(_api_key_pool)} 个key，轮询使用")

def get_next_api_key() -> str:
    """从key池中轮询获取下一个可用key，自动跳过冷却中的key"""
    if not _api_key_pool:
        return _global_api_key
    import time
    with _api_key_lock:
        now = time.time()
        # 尝试找一个未冷却的key
        for _ in range(len(_api_key_pool)):
            idx = next(_api_key_counter) % len(_api_key_pool)
            key = _api_key_pool[idx]
            if _api_key_cooldown.get(key, 0) <= now:
                return key
        # 全部冷却中，返回最早解冻的key
        earliest_key = min(_api_key_pool, key=lambda k: _api_key_cooldown.get(k, 0))
        return earliest_key


def mark_key_cooldown(api_key: str):
    """标记某个key冷却到下一分钟（TPM按分钟重置）"""
    import time, datetime
    now = datetime.datetime.now()
    seconds_to_next_min = 60 - now.second + 1
    with _api_key_lock:
        _api_key_cooldown[api_key] = time.time() + seconds_to_next_min

def set_global_seed(seed: int):
    """设置全局随机数种子，用于实验可复现"""
    global _global_seed
    _global_seed = seed

def get_global_seed() -> int:
    """获取全局随机数种子"""
    return _global_seed

def set_global_temperature(temperature: float):
    """设置全局温度参数"""
    global _global_temperature
    _global_temperature = temperature

def get_global_temperature() -> float:
    """获取全局温度参数"""
    return _global_temperature

def set_global_max_tokens(max_tokens: int):
    """设置全局最大生成token数"""
    global _global_max_tokens
    _global_max_tokens = max_tokens

def get_global_max_tokens() -> int:
    """获取全局最大生成token数"""
    return _global_max_tokens


def extract_final_answer(text: str, verbose: bool = True) -> str:
    """从推理模型输出中提取最终答案，去除思维链内容。
    
    支持的思维链格式：
    - <think>...</think> 标签包裹的内容
    - 以 </think> 结尾的思维内容（DeepSeek-R1格式：思维内容在前，</think>后是最终答案）
    
    Args:
        text: LLM返回的原始文本
        verbose: 是否输出检测提示信息
        
    Returns:
        提取后的最终答案文本
    """
    import re
    
    if not text:
        return text
    
    # 检测是否包含思维链标签
    has_think_tag = bool(re.search(r'<think>|</think>', text))
    
    # if has_think_tag and verbose:
    #     print("  [思维链] 检测到返回结果包含思维链，自动剔除思维标签")
    
    result = text
    
    # 格式1: <think>...</think> 完整标签包裹
    result = re.sub(r'<think>.*?</think>\s*', '', result, flags=re.DOTALL)
    
    # 格式2: DeepSeek-R1格式 - 思维内容在 </think> 之前，最终答案在 </think> 之后
    # 如果文本中有 </think>，只保留 </think> 之后的内容
    if '</think>' in result:
        parts = result.split('</think>')
        # 取最后一个 </think> 之后的内容作为最终答案
        result = parts[-1]
    
    # 移除可能残留的单独 <think> 标签
    result = re.sub(r'<think>\s*', '', result)
    
    # 清理多余的空白行
    result = re.sub(r'\n{3,}', '\n\n', result)
    
    return result.strip()


def generate(prompt: str, max_tokens: int = None, temperature: float = None, seed: int = None) -> str:
    """
    简化的LLM调用接口，接收prompt字符串，返回生成的文本内容。
    
    Args:
        prompt: 提示词字符串
        max_tokens: 最大生成token数
        temperature: 温度参数
        seed: 随机数种子，用于实验可复现（None则使用全局种子）
        
    Returns:
        生成的文本内容字符串
        
    Raises:
        RuntimeError: 当API调用失败时
    """
    # 使用传入的seed或全局seed
    actual_seed = seed if seed is not None else get_global_seed()
    actual_max_tokens = max_tokens if max_tokens is not None else get_global_max_tokens()
    actual_temperature = temperature if temperature is not None else get_global_temperature()
    
    messages = [{"role": "user", "content": prompt}]
    response = call_silicon_model(
        messages=messages,
        max_tokens=actual_max_tokens,
        temperature=actual_temperature,
        seed=actual_seed
    )
    
    if "error" in response:
        raise RuntimeError(f"LLM调用失败: {response['error']}")
    
    if "choices" in response and len(response["choices"]) > 0:
        content = response["choices"][0]["message"]["content"]
        # 自动提取最终答案，去除思维链内容
        return extract_final_answer(content)
    
    raise RuntimeError("LLM返回格式异常，未找到有效响应内容")



async def call_silicon_model_async(
    messages: List[Dict[str, str]], 
    model: str = None, 
    api_key: Optional[str] = None,
    max_tokens: int = 1024, 
    thinking_budget: int = 4096,
    temperature: float = 0.7, 
    stream: bool = False,
    min_p: float = 0.05,
    top_p: float = 0.7,
    top_k: int = 50,
    frequency_penalty: float = 0.5,
    stop: Optional[List[str]] = None,
    session: Optional[aiohttp.ClientSession] = None,
    max_retries: int = 10,
    retry_delay: float = 5.0,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    异步调用Silicon Flow API发送请求并获取模型响应（带智能重试）
    
    Args:
        messages: 消息列表，格式为[{"role": "user", "content": "内容"}]
        model: 模型名称，默认为"THUDM/GLM-4-9B-0414"
        api_key: API密钥，如未提供则使用默认值
        max_tokens: 最大生成token数
        thinking_budget: 思考预算
        temperature: 温度参数，控制随机性
        stream: 是否使用流式输出
        min_p: 最小概率阈值
        top_p: 概率截断阈值
        top_k: 保留的最高概率token数量
        frequency_penalty: 频率惩罚系数
        stop: 停止生成的标记列表
        session: 可选的aiohttp会话，用于连接复用
        max_retries: 最大重试次数，默认10次
        retry_delay: 重试延迟（秒），默认5秒
        seed: 随机数种子，用于实验可复现
        
    Returns:
        dict: 解析后的模型响应
    """
    # 如果没有指定 URL，使用全局配置的 URL
    url = get_global_url()
    
    # 如果没有指定模型，使用全局配置的模型
    if model is None:
        model = get_global_model()
    
    # 使用提供的API密钥或从key池轮询
    if api_key is None:
        api_key = get_next_api_key()
    
    # 准备请求参数
    payload = {
        "model": model,
        "stream": stream,
        "max_tokens": max_tokens,
        "thinking_budget": thinking_budget,
        "min_p": min_p,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "frequency_penalty": frequency_penalty,
        "n": 1,
        "stop": stop or [],
        "messages": messages
    }
    
    # 添加随机数种子（用于实验可复现）
    if seed is not None:
        payload["seed"] = seed
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # 判断是否需要创建新的session，使用连接池限制
    close_session = False
    if session is None:
        # 创建带连接池限制的session，防止内存泄漏
        connector = aiohttp.TCPConnector(
            limit=10,  # 总连接数限制
            limit_per_host=5,  # 每个主机连接数限制
            ttl_dns_cache=300,  # DNS缓存5分钟
            use_dns_cache=True,
            enable_cleanup_closed=True  # 启用清理已关闭连接
        )
        session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=180)  # 全局超时（适配推理模型）
        )
        close_session = True
    
    # 重试逻辑
    last_error = None
    try:
        for attempt in range(max_retries):
            try:
                # 使用更宽松的超时设置（适配推理模型）
                request_timeout = aiohttp.ClientTimeout(total=120, connect=10, sock_read=110)
                async with session.post(url, json=payload, headers=headers, timeout=request_timeout) as response:
                    # 检查HTTP状态码
                    if response.status == 429:
                        try:
                            error_text = await response.text()
                            # 尝试解析JSON错误信息
                            try:
                                import json
                                error_json = json.loads(error_text)
                                error_detail = error_json.get('error', {}).get('message', error_text)
                            except:
                                error_detail = error_text if error_text else 'Too Many Requests'
                        except:
                            error_detail = 'Too Many Requests'
                        
                        if attempt < max_retries - 1:
                            if len(_api_key_pool) > 1:
                                # 标记当前key冷却，获取下一个可用key
                                mark_key_cooldown(api_key)
                                new_key = get_next_api_key()
                                # 检查新key是否也在冷却中
                                import time as _time
                                cd = _api_key_cooldown.get(new_key, 0)
                                wait = cd - _time.time()
                                if wait > 0:
                                    print(f"429 限流 ({attempt + 1}/{max_retries}), 全部key冷却中，等待{wait:.0f}s...")
                                    await asyncio.sleep(wait)
                                else:
                                    print(f"429 限流 ({attempt + 1}/{max_retries}), 切换key重试...")
                                    await asyncio.sleep(0.3)
                                api_key = new_key
                                headers["Authorization"] = f"Bearer {api_key}"
                            else:
                                import datetime
                                now = datetime.datetime.now()
                                seconds_to_next_min = 60 - now.second + 1
                                print(f"429 限流 ({attempt + 1}/{max_retries}), 等待{seconds_to_next_min}s到下一分钟...")
                                await asyncio.sleep(seconds_to_next_min)
                            continue
                        else:
                            return {"error": f"HTTP 429: 请求过于频繁，已重试{max_retries}次\n详细错误: {error_detail}\n请求URL: {url}"}
                    
                    # 检查其他HTTP错误状态码
                    elif response.status == 401:
                        try:
                            error_text = await response.text()
                            # 尝试解析JSON错误信息
                            try:
                                import json
                                error_json = json.loads(error_text)
                                error_detail = error_json.get('error', {}).get('message', error_text[:100])
                            except:
                                error_detail = error_text[:100] if error_text else '未授权'
                        except:
                            error_detail = '未授权'
                        return {"error": f"HTTP 401: API Key无效或未授权，请检查API Key配置"}
                    elif response.status == 404:
                        try:
                            error_text = await response.text()
                        except:
                            error_text = ''
                        return {"error": f"HTTP 404: API地址不存在，请检查API URL配置"}
                    elif response.status == 400:
                        try:
                            error_text = await response.text()
                            # 尝试解析JSON错误信息
                            try:
                                import json
                                error_json = json.loads(error_text)
                                error_detail = error_json.get('error', {}).get('message', error_text[:200])
                            except:
                                error_detail = error_text[:200] if error_text else '请求参数错误'
                        except:
                            error_detail = '请求参数错误'
                        return {"error": f"HTTP 400: 请求参数错误 - {error_detail}"}
                    elif response.status >= 500:
                        try:
                            error_text = await response.text()
                            error_detail = error_text[:200] if error_text else '服务器错误'
                        except:
                            error_detail = '服务器错误'
                        return {"error": f"HTTP {response.status}: 服务器错误，请稍后重试"}
                    
                    # 检查其他HTTP错误
                    response.raise_for_status()
                    result = await response.json()
                    
                    # 成功返回
                    if attempt > 0:
                        print(f"  ✓ 重试成功（第{attempt + 1}次尝试）")
                    return result
                
            except asyncio.TimeoutError as e:
                last_error = e
                print(f"API调用超时: 120秒超时 (尝试 {attempt + 1}/{max_retries})")
                
                if attempt < max_retries - 1:
                    # 线性递增：第一次5秒，之后每次递增5秒
                    wait_time = retry_delay * (attempt + 1)
                    print(f"  等待 {wait_time:.1f} 秒后重试...")
                    await asyncio.sleep(wait_time)
                else:
                    return {"error": f"API调用超时，已重试{max_retries}次"}
                    
            except aiohttp.ClientError as e:
                last_error = e
                import traceback
                error_details = traceback.format_exc()
                print(f"API调用失败: {type(e).__name__}: {e} (尝试 {attempt + 1}/{max_retries})")
                print(f"  详细错误信息: {error_details}")
                
                # 如果是DNS错误，增加更长的等待时间
                if "getaddrinfo failed" in str(e) or "ClientConnectorDNSError" in str(type(e).__name__):
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (attempt + 2) * 2  # DNS错误时等待更久
                        print(f"  DNS解析失败，等待 {wait_time:.1f} 秒后重试...")
                        await asyncio.sleep(wait_time)
                    else:
                        return {"error": f"DNS解析失败: {str(e)}, 已重试{max_retries}次，请检查网络连接"}
                else:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (attempt + 1)
                        print(f"  等待 {wait_time:.1f} 秒后重试...")
                        await asyncio.sleep(wait_time)
                    else:
                        return {"error": f"{type(e).__name__}: {str(e)}, 已重试{max_retries}次\n详细错误: {error_details}"}
                    
            except Exception as json_error:
                # 检查是否是JSON解码错误
                import json as json_module
                if isinstance(json_error, json_module.JSONDecodeError):
                    import traceback
                    error_details = traceback.format_exc()
                    print(f"无法解析响应: {type(json_error).__name__}: {json_error}")
                    print(f"  详细错误信息: {error_details}")
                    return {"error": f"响应解析失败: {type(json_error).__name__}: {str(json_error)}\n详细错误: {error_details}"}
                else:
                    # 其他异常继续抛出
                    raise json_error
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                print(f"未知错误: {type(e).__name__}: {e}")
                print(f"  详细错误信息: {error_details}")
                return {"error": f"未知错误: {type(e).__name__}: {str(e)}\n详细错误: {error_details}"}
        
        # 如果所有重试都失败
        import traceback
        if last_error:
            error_details = ''.join(traceback.format_exception(type(last_error), last_error, last_error.__traceback__))
            return {"error": f"所有重试失败: {type(last_error).__name__}: {str(last_error)}\n详细错误: {error_details}"}
        else:
            return {"error": "所有重试失败: 未知错误"}
    finally:
        # 确保session在任何情况下都能正确关闭
        if close_session and session:
            try:
                # 检查session是否已经关闭
                if not session.closed:
                    await session.close()
                # 等待连接完全关闭，但不要太久
                await asyncio.sleep(0.05)
            except Exception as e:
                # 静默处理关闭错误，避免影响主流程
                pass


async def call_silicon_model_batch(
    messages_list: List[List[Dict[str, str]]],
    model: str = None,
    api_key: Optional[str] = None,
    max_tokens: int = 1024,
    thinking_budget: int = 4096,
    temperature: float = 0.7,
    stream: bool = False,
    min_p: float = 0.05,
    top_p: float = 0.7,
    top_k: int = 50,
    frequency_penalty: float = 0.5,
    stop: Optional[List[str]] = None,
    max_concurrent: int = 10,
    max_retries: int = 10,
    retry_delay: float = 5.0
) -> List[Dict[str, Any]]:
    """
    并发批量调用Silicon Flow API（带智能重试）
    
    Args:
        messages_list: 多个消息列表的列表
        model: 模型名称
        api_key: API密钥
        max_tokens: 最大生成token数
        thinking_budget: 思考预算
        temperature: 温度参数
        stream: 是否使用流式输出
        min_p: 最小概率阈值
        top_p: 概率截断阈值
        top_k: 保留的最高概率token数量
        frequency_penalty: 频率惩罚系数
        stop: 停止生成的标记列表
        max_concurrent: 最大并发数，默认10
        max_retries: 最大重试次数，默认5次
        retry_delay: 重试延迟（秒），默认1秒
        
    Returns:
        list: 所有请求的响应列表，顺序与输入一致
    """
    # 使用安全的连接器和超时配置
    connector = create_safe_connector(
        limit=max_concurrent * 2,
        limit_per_host=max_concurrent,
        ttl_dns_cache=300,
        force_close=True,
        keepalive_timeout=30,
        enable_cleanup_closed=True
    )
    
    timeout = create_safe_timeout(
        total=60,
        connect=10,
        sock_read=30
    )
    
    # 使用安全的ClientSession包装器
    if SafeClientSession:
        async with SafeClientSession(connector=connector, timeout=timeout) as session:
            # 创建信号量来限制并发数
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def limited_call(messages):
                async with semaphore:
                    return await call_silicon_model_async(
                        messages=messages,
                        model=model,
                        api_key=api_key,
                        max_tokens=max_tokens,
                        thinking_budget=thinking_budget,
                        temperature=temperature,
                        stream=stream,
                        min_p=min_p,
                        top_p=top_p,
                        top_k=top_k,
                        frequency_penalty=frequency_penalty,
                        stop=stop,
                        session=session,
                        max_retries=max_retries,
                        retry_delay=retry_delay
                    )
            
            # 并发执行所有请求，使用更安全的异步处理
            tasks = [limited_call(messages) for messages in messages_list]
            try:
                results = await safe_gather(*tasks, return_exceptions=True)
            except Exception as e:
                # 如果gather本身失败，返回错误结果
                print(f"批量请求执行失败: {e}")
                results = [{"error": f"批量请求失败: {str(e)}"} for _ in messages_list]
            
            # 处理异常情况
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    import traceback
                    error_details = ''.join(traceback.format_exception(type(result), result, result.__traceback__))
                    print(f"请求 {i} 失败: {type(result).__name__}: {result}")
                    print(f"  详细错误信息: {error_details}")
                    processed_results.append({"error": f"{type(result).__name__}: {str(result)}\n详细错误: {error_details}"})
                else:
                    processed_results.append(result)
            
            return processed_results
    else:
        # 如果SafeClientSession不可用，使用原生aiohttp
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def limited_call(messages):
                async with semaphore:
                    return await call_silicon_model_async(
                        messages=messages,
                        model=model,
                        api_key=api_key,
                        max_tokens=max_tokens,
                        thinking_budget=thinking_budget,
                        temperature=temperature,
                        stream=stream,
                        min_p=min_p,
                        top_p=top_p,
                        top_k=top_k,
                        frequency_penalty=frequency_penalty,
                        stop=stop,
                        session=session,
                        max_retries=max_retries,
                        retry_delay=retry_delay
                    )
            
            tasks = [limited_call(messages) for messages in messages_list]
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as e:
                print(f"批量请求执行失败: {e}")
                results = [{"error": f"批量请求失败: {str(e)}"} for _ in messages_list]
            
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    import traceback
                    error_details = ''.join(traceback.format_exception(type(result), result, result.__traceback__))
                    print(f"请求 {i} 失败: {type(result).__name__}: {result}")
                    processed_results.append({"error": f"{type(result).__name__}: {str(result)}\n详细错误: {error_details}"})
                else:
                    processed_results.append(result)
            
            return processed_results


def call_silicon_model(
    messages: List[Dict[str, str]], 
    model: str = None, 
    api_key: Optional[str] = None,
    max_tokens: int = 1024, 
    thinking_budget: int = 4096,
    temperature: float = 0.7, 
    stream: bool = False,
    min_p: float = 0.05,
    top_p: float = 0.7,
    top_k: int = 50,
    frequency_penalty: float = 0.5,
    stop: Optional[List[str]] = None,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    同步调用Silicon Flow API（兼容旧代码）
    内部使用异步实现
    
    Args:
        messages: 消息列表，格式为[{"role": "user", "content": "内容"}]
        model: 模型名称，默认为"THUDM/GLM-4-9B-0414"
        api_key: API密钥，如未提供则使用默认值
        max_tokens: 最大生成token数
        thinking_budget: 思考预算
        temperature: 温度参数，控制随机性
        stream: 是否使用流式输出
        min_p: 最小概率阈值
        top_p: 概率截断阈值
        top_k: 保留的最高概率token数量
        frequency_penalty: 频率惩罚系数
        stop: 停止生成的标记列表
        seed: 随机数种子，用于实验可复现
        
    Returns:
        dict: 解析后的模型响应
    """
    return asyncio.run(call_silicon_model_async(
        messages=messages,
        model=model,
        api_key=api_key,
        max_tokens=max_tokens,
        thinking_budget=thinking_budget,
        temperature=temperature,
        stream=stream,
        min_p=min_p,
        top_p=top_p,
        top_k=top_k,
        frequency_penalty=frequency_penalty,
        stop=stop,
        seed=seed
    ))

        
# 使用示例
if __name__ == "__main__":
    import time
    
    # 测试1: 单个同步调用（兼容旧代码）
    print("=== 测试1: 单个同步调用 ===")
    test_messages = [{"role": "user", "content": "你好，请用一句话介绍你自己"}]
    
    # 显示请求信息
    print(f"使用模型: {get_global_model()}")
    print(f"API地址: {get_global_url()}")
    print(f"请求消息: {test_messages}")
    
    result = call_silicon_model(test_messages)
    
    # 显示完整响应信息
    print(f"\n完整响应:")
    print(f"  模型: {result.get('model', '未知')}")
    print(f"  对象类型: {result.get('object', '未知')}")
    print(f"  创建时间: {result.get('created', '未知')}")
    print(f"  ID: {result.get('id', '未知')}")
    
    if "usage" in result:
        usage = result["usage"]
        print(f"  Token使用情况:")
        print(f"    提示Token: {usage.get('prompt_tokens', '未知')}")
        print(f"    完成Token: {usage.get('completion_tokens', '未知')}")
        print(f"    总Token: {usage.get('total_tokens', '未知')}")
    
    if "error" not in result:
        if "choices" in result and len(result["choices"]) > 0:
            choice = result["choices"][0]
            print(f"  完成原因: {choice.get('finish_reason', '未知')}")
            print(f"  索引: {choice.get('index', '未知')}")
            raw_content = choice['message']['content']
            # 应用思维链剔除
            clean_content = extract_final_answer(raw_content)
            print(f"响应内容: {clean_content}")
        else:
            print("未找到有效响应内容")
    else:
        print(f"错误: {result['error']}")
    
    print("\n=== 测试2: 异步并发批量调用（实时输出） ===")
    # 测试2: 异步并发批量调用，每个请求完成时立即输出
    async def test_batch_realtime():
        # 准备多个不同的请求
        messages_list = [
            [{"role": "user", "content": "1+1等于几？"}],
            [{"role": "user", "content": "Python是什么？"}],
            [{"role": "user", "content": "什么是人工智能？"}],
            [{"role": "user", "content": "今天天气怎么样？"}],
            [{"role": "user", "content": "推荐一本好书"}],
        ]
        
        print(f"开始并发请求 {len(messages_list)} 个问题...\n")
        start_time = time.time()
        
        # 创建共享session
        async with aiohttp.ClientSession() as session:
            semaphore = asyncio.Semaphore(10)  # 限制并发数为10
            
            # 创建任务字典，保存问题索引
            async def call_with_index(idx, messages):
                async with semaphore:
                    result = await call_silicon_model_async(
                        messages=messages,
                        session=session
                    )
                    return idx, messages, result
            
            # 创建所有任务
            tasks = [call_with_index(i, msg) for i, msg in enumerate(messages_list)]
            
            # 使用as_completed实时获取完成的任务
            completed_count = 0
            for coro in asyncio.as_completed(tasks):
                idx, messages, result = await coro
                completed_count += 1
                elapsed = time.time() - start_time
                
                print(f"[{completed_count}/{len(messages_list)}] 完成 (耗时: {elapsed:.2f}秒)")
                print(f"  问题: {messages[0]['content']}")
                
                if "error" not in result:
                    if "choices" in result and len(result["choices"]) > 0:
                        raw_content = result["choices"][0]["message"]["content"]
                        # 应用思维链剔除
                        clean_content = extract_final_answer(raw_content)
                        # 显示前150个字符
                        display_content = clean_content[:150] + "..." if len(clean_content) > 150 else clean_content
                        print(f"  回答: {display_content}")
                    else:
                        print("  未找到有效响应内容")
                else:
                    print(f"  错误: {result['error']}")
                print()  # 空行分隔
        
        end_time = time.time()
        print(f"所有请求完成，总耗时: {end_time - start_time:.2f}秒")
    
    # 运行异步批量测试
    asyncio.run(test_batch_realtime())

