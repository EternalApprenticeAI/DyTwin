"""数据加载与预处理

从 data/SocialTwin/{user_id}.csv 读取用户数据。
注意：这里的 user_id 是MD5隐私代号（例如 f33fdf31）。

字段参考：
- 原创/转发
- 日期
- 原微博内容
- 全文内容
- 转发数/评论数/点赞数
- 话题
- 图文识别
- 涉及行业
- 微博情绪
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def load_user_data(user_data_dir: Path, user_id: str) -> pd.DataFrame:
    file_path = user_data_dir / f"{user_id}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"找不到用户数据文件: {file_path}")

    df = pd.read_csv(file_path, encoding="utf-8-sig")

    # 解析日期（兼容不同编码/异常列名显示）
    date_col_candidates = ["日期", "����"]
    date_col = None
    for c in date_col_candidates:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        raise ValueError(f"用户数据缺少日期字段，当前列名: {list(df.columns)}")

    df["datetime"] = pd.to_datetime(df[date_col], errors="coerce")

    # 按时间排序
    df = df.sort_values("datetime")
    return df


def filter_by_time(df: pd.DataFrame, start_time: Optional[str] = None, end_time: Optional[str] = None) -> pd.DataFrame:
    if start_time:
        start_dt = pd.to_datetime(start_time)
        df = df[df["datetime"] >= start_dt]
    if end_time:
        end_dt = pd.to_datetime(end_time)
        df = df[df["datetime"] <= end_dt]
    return df

