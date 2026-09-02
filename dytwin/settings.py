"""dytwin 全局配置

统一管理所有配置项：
- 路径配置：项目根目录、数据目录、模型目录
- LLM配置：模型名称、API地址、API密钥、温度参数
- 实验配置：随机数种子、记忆检索参数
- 模拟配置：记忆top_k等

使用方式：
    from dytwin.settings import settings
    settings.seed = 42  # 设置随机种子
    settings.apply_llm_config()  # 应用LLM配置
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Optional


@dataclass
class Settings:
    """全局配置类，统一管理所有配置项"""
    
    # ==================== 路径配置 ====================
    # 项目根目录（自动推导为当前文件的上一级）
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    
    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"
    
    @property
    def user_data_dir(self) -> Path:
        if self._user_data_dir is not None:
            return self._user_data_dir
        return self.data_dir / "SocialTwin"

    @user_data_dir.setter
    def user_data_dir(self, value: Path) -> None:
        self._user_data_dir = Path(value)
    
    # 输出目录（可通过命令行覆盖）
    _output_dir: Optional[Path] = field(default=None, repr=False)
    
    @property
    def output_dir(self) -> Path:
        if self._output_dir is not None:
            return self._output_dir
        return self.project_root / "outputs"
    
    @output_dir.setter
    def output_dir(self, value: Path) -> None:
        self._output_dir = value
    
    @property
    def models_dir(self) -> Path:
        """模型根目录"""
        return self.project_root / "models"
    
    @property
    def embedding_model_dir(self) -> Path:
        """向量模型目录。

        默认遵循论文主实验设置，使用本地 BAAI/bge-small-zh-v1.5。
        如模型不放在仓库内，可通过 DYTWIN_EMBEDDING_MODEL_DIR 覆盖。
        """
        env_path = os.getenv("DYTWIN_EMBEDDING_MODEL_DIR", "").strip()
        if env_path:
            return Path(env_path)
        return self.models_dir / "BAAI" / "bge-small-zh-v1.5" / "BAAI" / "bge-small-zh-v1___5"
    
    # ==================== LLM配置 ====================
    llm_model: str = "THUDM/GLM-4-9B-0414"
    llm_api_url: str = "https://api.siliconflow.cn/v1/chat/completions"
    llm_api_key: str = field(default_factory=lambda: os.getenv("DYTWIN_API_KEY", ""))
    llm_temperature: float = 0.2  # 极低温度提高一致性和可复现性
    llm_max_tokens: int = 1024
    
    # ==================== 实验配置 ====================
    seed: Optional[int] = 42  # 随机数种子，用于实验可复现
    max_posts: Optional[int] = 50  # 最大模拟博文条数，None表示不限制
    
    # ==================== 模拟配置 ====================
    memory_top_k: int = 5  # 记忆检索top_k
    memory_similarity_threshold: float = 0.5  # 记忆检索相似度阈值
    
    # ==================== 评估配置 ====================
    use_llm_metrics: bool = True  # 是否使用LLM评估指标（默认开启）
    
    # ==================== 内部状态 ====================
    _llm_config_applied: bool = field(default=False, repr=False)
    _embedding_model: object = field(default=None, repr=False)
    _user_data_dir: Optional[Path] = field(default=None, repr=False)
    
    def apply_llm_config(self) -> None:
        """将配置应用到llm_call模块（只需调用一次）"""
        if self._llm_config_applied:
            return
            
        try:
            # 尝试不同的导入路径
            import sys
            from pathlib import Path
            
            # 添加项目根目录到Python路径
            project_root = Path(__file__).resolve().parents[1]
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            
            import llm_call
            llm_call.set_global_model(self.llm_model)
            llm_call.set_global_url(self.llm_api_url)
            llm_call.set_global_api_key(self.llm_api_key)
            llm_call.set_global_temperature(self.llm_temperature)
            llm_call.set_global_max_tokens(self.llm_max_tokens)
            if self.seed is not None:
                llm_call.set_global_seed(self.seed)
            
            self._llm_config_applied = True
            print(f"[配置] LLM配置已应用: model={self.llm_model}, url={self.llm_api_url}, seed={self.seed}")
            
        except ImportError as e:
            print(f"[警告] 无法导入llm_call模块: {e}")
            print(f"[警告] 请确保llm_call.py在项目根目录下")
            # 不设置_llm_config_applied为True，允许后续重试
    
    def get_embedding_model(self):
        """获取embedding模型（单例模式，避免重复加载）"""
        if self._embedding_model is None:
            from .embedding_model import EmbeddingModel
            self._embedding_model = EmbeddingModel(self.embedding_model_dir)
            print(f"[配置] 本地Embedding模型已加载: {self.embedding_model_dir.name}")
        return self._embedding_model
    
    def verify_llm_config(self) -> bool:
        """验证LLM配置是否正确应用"""
        try:
            import sys
            from pathlib import Path
            
            # 添加项目根目录到Python路径
            project_root = Path(__file__).resolve().parents[1]
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            
            import llm_call
            
            # 检查全局配置是否与设置一致
            current_model = llm_call.get_global_model()
            current_url = llm_call.get_global_url()
            current_key = llm_call.get_global_api_key()
            current_seed = llm_call.get_global_seed()
            
            config_matches = (
                current_model == self.llm_model and
                current_url == self.llm_api_url and
                current_key == self.llm_api_key and
                current_seed == self.seed
            )
            
            if not config_matches:
                print(f"[警告] LLM配置不一致:")
                print(f"  模型: 设置={self.llm_model}, 实际={current_model}")
                print(f"  URL: 设置={self.llm_api_url}, 实际={current_url}")
                print(f"  种子: 设置={self.seed}, 实际={current_seed}")
                return False
            
            print(f"[验证] LLM配置一致性检查通过")
            return True
            
        except ImportError as e:
            print(f"[错误] 无法验证LLM配置: {e}")
            return False
    
    
    def reset(self) -> None:
        """重置配置状态（用于测试）"""
        self._llm_config_applied = False
        self._embedding_model = None


# 全局单例配置对象
settings = Settings()

# 向后兼容：保留DEFAULT_SETTINGS别名
DEFAULT_SETTINGS = settings

