"""基于向量检索的记忆库

需求要点：
- 记忆机制基于向量数据库的构建和检索实现
- 记忆需要能序列化/持久化到用户结果目录，方便断点续跑

实现策略：
- 使用 faiss IndexFlatIP（内积） + 归一化向量 => cosine 相似度
- 存储结构：
  - memories.jsonl: 每行一条记忆（文本 + 元数据）
  - faiss.index: faiss索引文件

备注：如果你更希望使用 Chroma/FAISS-DB 等"向量数据库"，后续可以替换此实现；
当前实现是"轻量本地向量库"。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import numpy as np


@dataclass
class MemoryItem:
    text: str
    metadata: Dict[str, Any]


class VectorMemoryStore:
    def __init__(self, embedding_model, store_dir: Path):
        """ 
        Args:
            embedding_model: dytwin.embedding_model.EmbeddingModel
            store_dir: 存储目录（通常是 outputs/{user}/memory/ ）
        """
        self.embedding_model = embedding_model
        self.store_dir = store_dir
        self.store_dir.mkdir(parents=True, exist_ok=True)

        self._mem_path = self.store_dir / "memories.jsonl"
        self._faiss_path = self.store_dir / "faiss.index"

        self._memories: List[MemoryItem] = []
        self._index = None

        self._load_if_exists()

    @property
    def size(self) -> int:
        return len(self._memories)

    def clear(self) -> None:
        """清空记忆库（删除所有记忆和索引文件）"""
        self._memories = []
        self._index = None
        
        # 删除持久化文件
        if self._mem_path.exists():
            self._mem_path.unlink()
        if self._faiss_path.exists():
            self._faiss_path.unlink()

    def _init_index(self, dim: int):
        import faiss

        self._index = faiss.IndexFlatIP(dim)

    def _load_if_exists(self) -> None:
        # 加载 memories
        if self._mem_path.exists():
            with open(self._mem_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    self._memories.append(MemoryItem(text=obj["text"], metadata=obj.get("metadata", {})))

        # 加载索引
        if self._faiss_path.exists():
            import faiss
            
            try:
                self._index = faiss.read_index(str(self._faiss_path))
                # 检查维度是否匹配当前embedding模型
                current_dim = self.embedding_model.dim
                if self._index.d != current_dim:
                    print(f"[向量内存] 检测到维度不匹配: 索引维度={self._index.d}, 模型维度={current_dim}")
                    print(f"[向量内存] 删除旧索引，重新初始化...")
                    self._faiss_path.unlink()  # 删除旧索引文件
                    self._index = None
                    # 重新构建索引
                    if self._memories:
                        self._init_index(current_dim)
                        vecs = self.embedding_model.embed_texts([m.text for m in self._memories])
                        self._index.add(vecs)
                else:
                    print(f"[向量内存] 加载现有索引: 维度={self._index.d}")
            except Exception as e:
                print(f"[向量内存] 加载索引失败: {e}, 将重新创建")
                if self._faiss_path.exists():
                    self._faiss_path.unlink()
                self._index = None
                # 重新构建索引
                if self._memories:
                    dim = self.embedding_model.dim
                    self._init_index(dim)
                    vecs = self.embedding_model.embed_texts([m.text for m in self._memories])
                    self._index.add(vecs)
        else:
            # 没有索引则根据 memories 重建
            if self._memories:
                dim = self.embedding_model.dim
                self._init_index(dim)
                vecs = self.embedding_model.embed_texts([m.text for m in self._memories])
                self._index.add(vecs)

    def persist(self) -> None:
        """持久化 memories + faiss index"""
        # 写 memories.jsonl（仅保存text，因为text已包含时间和类型信息）
        with open(self._mem_path, "w", encoding="utf-8") as f:
            for m in self._memories:
                f.write(json.dumps({"text": m.text}, ensure_ascii=False) + "\n")

        # 写 faiss.index
        if self._index is not None:
            import faiss

            faiss.write_index(self._index, str(self._faiss_path))

    def add(self, text: str, metadata: Optional[Dict[str, Any]] = None, auto_persist: bool = True) -> None:
        """添加记忆（metadata参数保留兼容性但不再持久化）
        
        Args:
            text: 记忆文本
            metadata: 元数据（仅内存中保留）
            auto_persist: 是否自动持久化（默认True，防止程序中断丢失数据）
        """
        # 确保索引存在且维度匹配
        current_dim = self.embedding_model.dim
        if self._index is None:
            print(f"[向量内存] 初始化新索引: 维度={current_dim}")
            self._init_index(current_dim)
        elif self._index.d != current_dim:
            print(f"[向量内存] 维度不匹配，重新初始化索引: 旧维度={self._index.d}, 新维度={current_dim}")
            self._init_index(current_dim)
            # 如果有旧的memories，需要重新编码并添加
            if self._memories:
                print(f"[向量内存] 重新编码 {len(self._memories)} 个已有记忆")
                old_texts = [m.text for m in self._memories]
                vecs = self.embedding_model.embed_texts(old_texts)
                self._index.add(vecs)

        vec = self.embedding_model.embed_text(text)
        self._index.add(np.asarray([vec], dtype=np.float32))
        # metadata仅在内存中保留，不持久化（text已包含完整信息）
        self._memories.append(MemoryItem(text=text, metadata=metadata or {}))
        
        # 自动持久化，防止程序中断丢失数据
        if auto_persist:
            self.persist()

    def search(self, query: str, top_k: int = 3, similarity_threshold: float = 0.5) -> List[Tuple[MemoryItem, float]]:
        """检索相关记忆
        
        Args:
            query: 查询文本
            top_k: 最大返回数量（默认3，避免不相关记忆参与预测）
            similarity_threshold: 相似度阈值（默认0.5，只返回相似度大于此值的记忆）
        
        Returns:
            符合条件的记忆列表，按相似度降序排列
        """
        if self._index is None or not self._memories:
            return []

        qv = self.embedding_model.embed_text(query)
        qv = np.asarray([qv], dtype=np.float32)

        # faiss 返回 (scores, indices)
        scores, idxs = self._index.search(qv, min(top_k * 2, len(self._memories)))  # 多检索一些再过滤

        results: List[Tuple[MemoryItem, float]] = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if idx < 0:
                continue
            # 只保留相似度大于阈值的记忆
            if score >= similarity_threshold:
                results.append((self._memories[idx], float(score)))
        
        # 限制返回数量
        return results[:top_k]

