"""向量模型封装

根据用户要求：向量模型文件在 @models 中。
这里默认使用 SentenceTransformer 加载本地 bge-small-zh-v1.5。

注意：如果运行环境缺少 sentence-transformers，请先安装依赖。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np


@dataclass
class EmbeddingModel:
    model_dir: Path
    normalize: bool = True

    def __post_init__(self) -> None:
        # 延迟导入，避免在未安装依赖时影响其它模块
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(str(self.model_dir))

    @property
    def dim(self) -> int:
        # SentenceTransformer 通常没有直接暴露维度，这里通过一次空encode推断
        v = self.embed_texts(["维度探测"])
        return int(v.shape[1])

    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        texts_list: List[str] = list(texts)
        emb = self._model.encode(
            texts_list,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return emb.astype(np.float32)

