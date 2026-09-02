"""
异步流管理模块
专门处理aiohttp流操作，防止"Cannot call write after a stream was destroyed"错误
"""

import asyncio
import aiohttp
import weakref
import atexit
from typing import Optional, Dict, Any
import logging

class StreamManager:
    """异步流管理器，确保所有连接和流都能正确关闭"""
    
    def __init__(self):
        self._sessions = weakref.WeakSet()
        self._connectors = weakref.WeakSet()
        self._cleanup_registered = False
        
        # 配置日志
        self.logger = logging.getLogger(__name__)
        
    def register_session(self, session: aiohttp.ClientSession):
        """注册session以便统一管理"""
        self._sessions.add(session)
        if not self._cleanup_registered:
            atexit.register(self._cleanup_all)
            self._cleanup_registered = True
    
    def register_connector(self, connector: aiohttp.TCPConnector):
        """注册connector以便统一管理"""
        self._connectors.add(connector)
    
    async def safe_close_session(self, session: aiohttp.ClientSession):
        """安全关闭session"""
        if session and not session.closed:
            try:
                # 等待所有pending的请求完成
                await asyncio.sleep(0.01)
                
                # 关闭session
                await session.close()
                
                # 等待连接完全关闭
                await asyncio.sleep(0.02)
                
            except Exception as e:
                # 静默处理关闭错误，记录日志但不抛出异常
                self.logger.debug(f"关闭session时出现错误: {e}")
    
    async def safe_close_connector(self, connector: aiohttp.TCPConnector):
        """安全关闭connector"""
        if connector and not connector.closed:
            try:
                await connector.close()
                await asyncio.sleep(0.01)
            except Exception as e:
                self.logger.debug(f"关闭connector时出现错误: {e}")
    
    def _cleanup_all(self):
        """程序退出时清理所有资源"""
        try:
            # 创建新的事件循环来清理资源
            loop = None
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    loop = None
            except RuntimeError:
                loop = None
            
            if loop is None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # 清理所有session
            for session in list(self._sessions):
                if not session.closed:
                    loop.run_until_complete(self.safe_close_session(session))
            
            # 清理所有connector
            for connector in list(self._connectors):
                if not connector.closed:
                    loop.run_until_complete(self.safe_close_connector(connector))
                    
        except Exception as e:
            # 静默处理清理错误
            pass

# 全局流管理器实例
_global_stream_manager = StreamManager()

def get_stream_manager() -> StreamManager:
    """获取全局流管理器"""
    return _global_stream_manager

class SafeClientSession:
    """安全的ClientSession包装器"""
    
    def __init__(self, connector: Optional[aiohttp.TCPConnector] = None, 
                 timeout: Optional[aiohttp.ClientTimeout] = None, **kwargs):
        self._connector = connector
        self._timeout = timeout
        self._kwargs = kwargs
        self._session: Optional[aiohttp.ClientSession] = None
        self._manager = get_stream_manager()
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        if self._connector:
            self._manager.register_connector(self._connector)
        
        self._session = aiohttp.ClientSession(
            connector=self._connector,
            timeout=self._timeout,
            **self._kwargs
        )
        
        self._manager.register_session(self._session)
        return self._session
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self._session:
            await self._manager.safe_close_session(self._session)
            self._session = None

def create_safe_connector(limit: int = 100, 
                         limit_per_host: int = 30,
                         ttl_dns_cache: int = 300,
                         force_close: bool = True,
                         keepalive_timeout: int = 30,
                         enable_cleanup_closed: bool = True) -> aiohttp.TCPConnector:
    """创建安全的TCP连接器"""
    connector = aiohttp.TCPConnector(
        limit=limit,
        limit_per_host=limit_per_host,
        ttl_dns_cache=ttl_dns_cache,
        use_dns_cache=True,
        force_close=force_close,
        keepalive_timeout=keepalive_timeout,
        enable_cleanup_closed=enable_cleanup_closed
    )
    
    manager = get_stream_manager()
    manager.register_connector(connector)
    
    return connector

def create_safe_timeout(total: int = 60, 
                       connect: int = 10, 
                       sock_read: int = 30) -> aiohttp.ClientTimeout:
    """创建安全的超时配置"""
    return aiohttp.ClientTimeout(
        total=total,
        connect=connect,
        sock_read=sock_read
    )

async def safe_gather(*tasks, return_exceptions: bool = True):
    """安全的asyncio.gather，防止流错误"""
    try:
        # 添加短暂延迟，让所有任务有时间启动
        await asyncio.sleep(0.001)
        
        # 执行gather
        results = await asyncio.gather(*tasks, return_exceptions=return_exceptions)
        
        # 添加短暂延迟，让所有连接有时间清理
        await asyncio.sleep(0.001)
        
        return results
        
    except Exception as e:
        # 如果gather失败，确保所有任务都被取消
        for task in tasks:
            if hasattr(task, 'cancel') and not task.done():
                task.cancel()
        
        # 等待任务取消完成
        await asyncio.sleep(0.01)
        
        raise e

# 导出主要接口
__all__ = [
    'StreamManager',
    'SafeClientSession', 
    'create_safe_connector',
    'create_safe_timeout',
    'safe_gather',
    'get_stream_manager'
]

