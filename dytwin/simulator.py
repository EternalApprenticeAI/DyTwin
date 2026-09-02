"""动态模拟主逻辑

核心流程(按时间推进):
- 用户画像初始为空(或使用预制画像)
- 对每条转发博文:
  1) 构建query(原微博/全文内容/话题等)
  2) 向量记忆检索(仅包含历史转发记录)
  3) LLM预测转发文本
  4) 评估: ROUGE-L, BERTScore, Embedding sim, LLM情绪/立场
  5) 反思: 对比真实与预测 -> 给出画像改进建议 -> 更新画像
  6) 将转发记录写入记忆库

输出:
- 保存到 outputs/{user_id}/simulation.csv
- 同时保存记忆库到对应目录
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .settings import Settings
from .embedding_model import EmbeddingModel
from .user_profile import UserProfile
from .vector_memory import VectorMemoryStore
from .evaluator import Evaluator
from .llm_integration import (
    predict_forward_text_with_focus,
    reflect_and_improve_profile,
)
from .visualization import visualize_simulation_results


@dataclass
class SimulatorConfig:
    memory_top_k: int = 5


class DynamicUserSimulator:
    def __init__(
        self, 
        settings: Settings, 
        user_id: str, 
        config: Optional[SimulatorConfig] = None, 
        overwrite: bool = True,
    ):
        """
        Args:
            settings: 配置对象
            user_id: 用户ID
            config: 模拟器配置
            overwrite: 是否覆盖已有结果,默认True(False时继续历史推演)
        """
        self.settings = settings
        self.user_id = user_id
        self.config = config or SimulatorConfig(memory_top_k=settings.memory_top_k)
        self._overwrite = overwrite

        # 应用LLM配置（包括随机种子）
        settings.apply_llm_config()
        
        # 使用单例embedding模型，避免重复加载
        self.embedding_model = settings.get_embedding_model()
        
        # 初始化画像(支持预制画像)
        self.profile = UserProfile()

        # 输出目录
        self.user_out_dir = settings.output_dir / user_id
        
        # 如果覆盖模式,清除旧数据
        if overwrite and self.user_out_dir.exists():
            self._clear_old_data()
        
        self.user_out_dir.mkdir(parents=True, exist_ok=True)

        # DP+DM 固定启用动态记忆
        self.memory_store = VectorMemoryStore(
            embedding_model=self.embedding_model,
            store_dir=self.user_out_dir / "memory",
        )

        # 评估器
        self.evaluator = Evaluator(
            self.embedding_model,
            skip_bert_score=False,
            bert_model_path=settings.embedding_model_dir,
            use_llm_metrics=settings.use_llm_metrics,
        )

        # 结果缓存
        self._rows: List[Dict[str, Any]] = []
        
        # 继续模式下恢复画像状态
        if not overwrite:
            self._restore_profile_from_history()
        
        print("[模拟模式] DP+DM: 动态画像 + 动态记忆")
    
    def _restore_profile_from_history(self) -> None:
        """从历史模拟结果恢复画像状态"""
        csv_path = self.user_out_dir / "simulation.csv"
        if not csv_path.exists():
            print("[继续模式] 未找到历史模拟文件,画像从空开始")
            return
        
        try:
            history_df = pd.read_csv(csv_path)
            if len(history_df) == 0:
                print("[继续模式] 历史模拟文件为空,画像从空开始")
                return
            
            # 获取最后一条记录的画像
            last_row = history_df.iloc[-1]
            profile_after = last_row.get('profile_after', '')
            
            if profile_after and pd.notna(profile_after) and profile_after.strip():
                # 恢复画像(profile_after 是字符串格式的画像)
                self.profile.restore_from_snapshot(profile_after)
                print(f"[继续模式] 已从历史恢复画像状态")
            else:
                print("[继续模式] 历史画像为空,画像从空开始")
                
        except Exception as e:
            print(f"[继续模式] 恢复画像失败: {e},画像从空开始")
    
    def _clear_old_data(self) -> None:
        """清除用户目录下的旧数据"""
        import shutil
        try:
            # 删除整个用户目录
            shutil.rmtree(self.user_out_dir)
            print(f"已清除用户 {self.user_id} 的旧数据")
        except Exception as e:
            print(f"清除旧数据时出错: {e}")

    def _serialize_profile(self) -> str:
        """序列化画像为字符串,用于CSV输出"""
        return self.profile.snapshot()

    def _serialize_memories(self, memories: List[Tuple[str, float]]) -> str:
        return json.dumps(
            [{"text": t, "score": s} for t, s in memories],
            ensure_ascii=False,
        )

    def _serialize_few_shot(self, few_shot_examples: List[Tuple[str, str]]) -> str:
        """序列化短期记忆（few-shot样例）为字符串"""
        if not few_shot_examples:
            return ""
        return json.dumps(
            [{"original_post": post, "user_forward": fwd} for post, fwd in few_shot_examples],
            ensure_ascii=False,
        )

    def _get_few_shot_examples(self, max_examples: int = 3) -> List[Tuple[str, str]]:
        """获取最近的转发样例作为few-shot（短期记忆）
        
        Args:
            max_examples: 最多返回的样例数量
        
        Returns:
            列表，每个元素为(原博文摘要, 用户转发语)
        """
        examples = []
        # 从已处理的行中获取最近的转发记录
        for row in reversed(self._rows[-max_examples:]):
            original_post = row.get("original_post", "")
            true_forward = row.get("true_forward", "")
            if original_post and true_forward:
                examples.append((original_post, true_forward))
        return list(reversed(examples))  # 保持时间顺序

    def _init_csv_file(self) -> Path:
        """初始化CSV文件,写入表头(继续模式下不覆盖已有文件)"""
        out_path = self.user_out_dir / "simulation.csv"
        # 定义CSV列顺序
        self._csv_columns = [
            "datetime", "type", "original_post", "true_forward", "pred_forward",
            "retrieved_memories", "few_shot_examples", "pred_rationale", 
            "rouge_l_f1", "bert_score_f1", "embedding_similarity",
            "llm_semantic", "llm_emotion", "llm_stance", "llm_style", "llm_focus",
            "llm_average", "llm_rationale",
            "profile_before", "profile_patch",
            "profile_after", "reflection", "improvement_suggestion", "memory_written", "error"
        ]
        # 继续模式下不覆盖已有文件
        if not self._overwrite and out_path.exists():
            print(f"[继续模式] 追加到已有CSV文件: {out_path}")
            return out_path
        # 写入表头
        pd.DataFrame(columns=self._csv_columns).to_csv(out_path, index=False, encoding="utf-8-sig")
        return out_path

    def _append_row_to_csv(self, row_data: Dict[str, Any]) -> None:
        """追加一行数据到CSV文件"""
        out_path = self.user_out_dir / "simulation.csv"
        # 确保所有列都存在
        row_with_all_cols = {col: row_data.get(col, "") for col in self._csv_columns}
        pd.DataFrame([row_with_all_cols]).to_csv(
            out_path, mode='a', header=False, index=False, encoding="utf-8-sig"
        )

    def run(self, df: pd.DataFrame, auto_visualize: bool = True) -> pd.DataFrame:
        """对已按时间排序且包含 datetime 列的df执行模拟
        
        Args:
            df: 转发博文数据
            auto_visualize: 是否自动生成可视化图表，默认True
        """
        total_count = len(df)
        
        print(f"\n{'='*60}")
        print(f"开始模拟: 共 {total_count} 条转发数据")
        print(f"{'='*60}\n")
        
        # 初始化CSV文件
        self._init_csv_file()
        
        for idx, (_, row) in enumerate(df.iterrows(), start=1):
            item = row.to_dict()
            post_date = str(item.get("日期") or "")
            
            print(f"[{idx}/{total_count}] {post_date}", end=" | ")
            self._handle_forward(item)
            print("处理完成 ✓")

        # 持久化记忆
        self.memory_store.persist()
        
        print(f"\n{'='*60}")
        print(f"模拟完成: 共处理 {total_count} 条转发数据")
        print(f"结果保存至: {self.user_out_dir / 'simulation.csv'}")
        print(f"{'='*60}\n")

        # 自动生成可视化图表
        if auto_visualize:
            csv_path = self.user_out_dir / "simulation.csv"
            viz_output_dir = self.user_out_dir / "visualization"
            try:
                visualize_simulation_results(csv_path, viz_output_dir, show_plots=False, train_size=0)
            except Exception as e:
                print(f"生成可视化图表时出错: {e}")

        return pd.DataFrame(self._rows)

    def _handle_forward(self, item: Dict[str, Any]) -> None:
        root_post_content = str(item.get("原微博内容") or "").strip()
        true_forward_text = str(item.get("全文内容") or "").strip()
        post_date = str(item.get("日期") or "")
        
        # 检索历史动态记忆
        query = root_post_content
        searched = self.memory_store.search(query, top_k=self.config.memory_top_k)
        memories = [(m.text, score) for m, score in searched]

        # 构建传给预测函数的数据,排除真实转发内容(避免数据泄露)
        root_post_for_prediction = {
            k: v for k, v in item.items() 
            if k not in ["全文内容", "标题／微博内容"]  # 排除可能包含真实转发的字段
        }

        # 预测(使用带重点关注的预测，传入转发时间)
        few_shot_for_pred = self._get_few_shot_examples()
        
        current_profile_text = self.profile.snapshot()
        pred_obj = predict_forward_text_with_focus(
            root_post=root_post_for_prediction,
            current_profile=current_profile_text,
            memories=memories,
            few_shot_examples=few_shot_for_pred,
            forward_time=post_date,
        )
        predicted_forward_text = (pred_obj.get("predicted_forward_text") or "").strip()
        prediction_rationale = (pred_obj.get("rationale") or "").strip()

        # 评估(传递原博文用于立场判断)
        metrics = self.evaluator.evaluate_all(predicted_forward_text, true_forward_text, root_post_content)

        # 反思并动态更新画像
        profile_before = self.profile.snapshot()
        reflection = ""
        improvement_suggestion = ""
        profile_patch = {}
        few_shot_examples = self._get_few_shot_examples()
        reflect = reflect_and_improve_profile(
            simulation_time=post_date,
            root_post=item,
            current_profile=profile_before,
            memories=memories,
            few_shot_examples=few_shot_examples,
            predicted_forward_text=predicted_forward_text,
            pred_rationale=prediction_rationale,
            true_forward_text=true_forward_text,
            llm_score_rationale=metrics.get("llm_rationale", ""),
        )
        reflection = reflect.get("reflection", "")
        improvement_suggestion = reflect.get("improvement_suggestion", "")
        profile_patch = reflect.get("profile_patch", {})
        self.profile.apply_patch(profile_patch)
        profile_after = self.profile.snapshot()

        # 保存转发记忆
        memory_written = ""
        if true_forward_text or root_post_content:
            try:
                memory_text = f"[转发] {post_date} | 原微博: {root_post_content} | 用户回应: {true_forward_text}"
                self.memory_store.add(
                    memory_text,
                    metadata={
                        "type": "forward",
                        "date": post_date,
                    },
                )
                memory_written = memory_text
            except Exception as e:
                print(f"添加转发记忆时出错: {str(e)}")

        # 拆分metrics各维度为独立字段
        row_data = {
            "datetime": post_date,
            "type": "转发",
            "original_post": root_post_content,
            "true_forward": true_forward_text,
            "pred_forward": predicted_forward_text,
            "retrieved_memories": self._serialize_memories(memories),
            "few_shot_examples": self._serialize_few_shot(few_shot_for_pred),
            "pred_rationale": prediction_rationale,
            "rouge_l_f1": metrics.get("rouge-l-f1", ""),
            "bert_score_f1": metrics.get("bert-score-f1", ""),
            "embedding_similarity": metrics.get("embedding-similarity", ""),
            "llm_semantic": metrics.get("llm_semantic", ""),
            "llm_emotion": metrics.get("llm_emotion", ""),
            "llm_stance": metrics.get("llm_stance", ""),
            "llm_style": metrics.get("llm_style", ""),
            "llm_focus": metrics.get("llm_focus", ""),
            "llm_average": metrics.get("llm_average", ""),
            "llm_rationale": metrics.get("llm_rationale", ""),
            "profile_before": profile_before,
            "profile_patch": json.dumps(profile_patch, ensure_ascii=False),
            "profile_after": profile_after,
            "reflection": reflection,
            "improvement_suggestion": improvement_suggestion,
            "memory_written": memory_written,
        }
        self._rows.append(row_data)
        self._append_row_to_csv(row_data)  # 实时保存

