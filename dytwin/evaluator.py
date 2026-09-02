"""评估模块

评估指标分为两类：
1. 确定性指标（无随机性，适合稳定复现和比较）：
   - ROUGE-L F1：词级重叠度
   - BERTScore F1：语义相似度
   - Embedding Similarity：向量余弦相似度

2. LLM指标（有随机性，可选）：
   - LLM多维度打分（语义、情感、立场、风格、焦点）

默认使用确定性指标，LLM指标可通过参数启用。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# 支持直接运行此文件
if __name__ == "__main__":
    _project_root = Path(__file__).resolve().parents[1]
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    from dytwin.embedding_model import EmbeddingModel
    from dytwin.llm_integration import llm_similarity_score
else:
    from .embedding_model import EmbeddingModel
    from .llm_integration import llm_similarity_score

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(
        self, 
        embedding_model: EmbeddingModel, 
        skip_bert_score: bool = False,
        bert_model_path: Optional[Path] = None,
        use_llm_metrics: bool = False,  # 默认不使用LLM指标
    ):
        """
        Args:
            embedding_model: 向量模型
            skip_bert_score: 是否跳过BERTScore计算
            bert_model_path: BERTScore使用的模型路径
            use_llm_metrics: 是否使用LLM指标（有随机性，默认False）
        """
        self.embedding_model = embedding_model
        self._rouge = None
        self._rouge_init_failed = False
        self._bert_scorer = None
        self._bert_scorer_init_failed = False
        self._skip_bert_score = skip_bert_score
        self._use_llm_metrics = use_llm_metrics
        # 使用本地bge模型做BERTScore
        self._bert_model_path = bert_model_path

    def _lazy_init_rouge(self):
        if self._rouge is None and not self._rouge_init_failed:
            try:
                from rouge import Rouge

                self._rouge = Rouge()
            except ImportError:
                self._rouge_init_failed = True
                logger.warning("rouge not installed. ROUGE metrics will be skipped.")
            except Exception as e:
                self._rouge_init_failed = True
                logger.warning(f"Rouge initialization failed: {e}")

    def _lazy_init_bert_score(self):
        if self._bert_scorer is None and not self._bert_scorer_init_failed:
            if self._skip_bert_score:
                self._bert_scorer_init_failed = True
                return
            try:
                from bert_score import BERTScorer

                if self._bert_model_path and self._bert_model_path.exists():
                    # 使用本地模型（bge-small-zh-v1.5 只有4层）
                    model_path = str(self._bert_model_path)
                    self._bert_scorer = BERTScorer(
                        model_type=model_path,
                        num_layers=4,  # bge-small 模型层数
                        rescale_with_baseline=False  # 本地模型无baseline
                    )
                    logger.info(f"BERTScorer initialized with local model: {model_path}")
                else:
                    # 本地模型不存在，跳过BERTScore（避免网络下载）
                    self._bert_scorer_init_failed = True
                    logger.warning("Local model not found. BERTScore metrics will be skipped.")
            except ImportError:
                self._bert_scorer_init_failed = True
                logger.warning("bert-score not installed. BERTScore metrics will be skipped.")
            except Exception as e:
                self._bert_scorer_init_failed = True
                logger.warning(f"BERTScorer initialization failed: {e}. BERTScore metrics will be skipped.")

    def _tokenize_chinese(self, text: str) -> str:
        """对中文文本进行词级分词，使用jieba分词"""
        try:
            import jieba
            # 使用jieba进行中文分词，用空格连接
            return ' '.join(jieba.cut(text))
        except ImportError:
            # 如果jieba未安装，回退到字符级分词
            result = []
            for char in text:
                if '\u4e00' <= char <= '\u9fff':  # 中文字符范围
                    result.append(' ' + char + ' ')
                else:
                    result.append(char)
            return ''.join(result).strip()

    def calculate_rouge(self, pred: str, true: str, debug: bool = False) -> Dict[str, float]:
        self._lazy_init_rouge()
        if not self._rouge or not pred or not true:
            return {}
        try:
            # 对中文文本进行分词处理
            pred_tokenized = self._tokenize_chinese(pred)
            true_tokenized = self._tokenize_chinese(true)
            
            if debug:
                print("\n[ROUGE-L 分词调试]")
                print(f"  原文本1: {pred[:100]}{'...' if len(pred) > 100 else ''}")
                print(f"  分词后1: {pred_tokenized[:150]}{'...' if len(pred_tokenized) > 150 else ''}")
                print(f"  原文本2: {true[:100]}{'...' if len(true) > 100 else ''}")
                print(f"  分词后2: {true_tokenized[:150]}{'...' if len(true_tokenized) > 150 else ''}")
            
            scores = self._rouge.get_scores(pred_tokenized, true_tokenized, avg=True)
            return {"rouge-l-f1": scores["rouge-l"]["f"]}
        except Exception:
            return {}

    def calculate_bert_score(self, pred: str, true: str) -> Dict[str, float]:
        self._lazy_init_bert_score()
        if not self._bert_scorer or not pred or not true:
            return {}
        try:
            _, _, f1 = self._bert_scorer.score([pred], [true])
            return {"bert-score-f1": f1.item()}
        except Exception:
            return {}

    def calculate_embedding_similarity(self, pred: str, true: str) -> Dict[str, float]:
        if not pred or not true:
            return {}
        try:
            v_pred = self.embedding_model.embed_text(pred)
            v_true = self.embedding_model.embed_text(true)
            # (v_pred @ v_true.T) / (||v_pred|| * ||v_true||)
            # 归一化向量的内积即为cosine similarity
            similarity = np.dot(v_pred, v_true)
            return {"embedding-similarity": float(similarity)}
        except Exception:
            return {}

    def calculate_llm_score(self, pred: str, true: str) -> Dict[str, Any]:
        """使用LLM对两段文本进行多维度相似度打分"""
        if not pred or not true:
            return {}
        try:
            result = llm_similarity_score(pred, true)
            # 提取各维度分数和平均分
            return {
                "llm_semantic": result.get("semantic_similarity", 0),
                "llm_emotion": result.get("emotion_similarity", 0),
                "llm_stance": result.get("stance_similarity", 0),
                "llm_style": result.get("style_similarity", 0),
                "llm_focus": result.get("focus_similarity", 0),
                "llm_average": result.get("average_score", 0),
                "llm_rationale": result.get("rationale", ""),
            }
        except Exception as e:
            logger.warning(f"LLM similarity scoring failed: {e}")
            return {}

    def evaluate_all(self, pred: str, true: str, original_post: str = "") -> Dict[str, Any]:
        """评估预测文本与真实文本的相似度
        
        默认使用确定性指标（无随机性），LLM指标需要在初始化时设置use_llm_metrics=True。
        
        确定性指标：
        - rouge-l-f1: ROUGE-L F1分数
        - bert-score-f1: BERTScore F1分数
        - embedding-similarity: 向量余弦相似度
        
        LLM指标（可选，有随机性）：
        - llm_semantic/emotion/stance/style/focus: LLM多维度打分
        
        Args:
            pred: 预测文本
            true: 真实文本
            original_post: 原博文（保留参数以兼容调用）
        """
        metrics = {}
        
        # ===== 确定性指标（始终计算） =====
        metrics.update(self.calculate_rouge(pred, true))
        metrics.update(self.calculate_bert_score(pred, true))
        metrics.update(self.calculate_embedding_similarity(pred, true))
        
        # ===== LLM指标（可选，有随机性） =====
        if self._use_llm_metrics:
            # LLM五维度打分（论文中定义的 e_llm ∈ R^5）
            try:
                llm_result = self.calculate_llm_score(pred, true)
                metrics.update(llm_result)
            except Exception as e:
                logger.warning(f"LLM scoring failed: {e}")

        return metrics


def interactive_evaluate():
    """交互式文本相似度评估入口"""
    from dytwin.settings import DEFAULT_SETTINGS
    from dytwin.embedding_model import EmbeddingModel
    
    print("=" * 60)
    print("文本相似度评估工具")
    print("=" * 60)
    print("\n请输入第一段文本（输入完成后按两次回车）：")
    
    lines1 = []
    while True:
        line = input()
        if line == "":
            if lines1:
                break
        else:
            lines1.append(line)
    text1 = "\n".join(lines1)
    
    print("\n请输入第二段文本（输入完成后按两次回车）：")
    lines2 = []
    while True:
        line = input()
        if line == "":
            if lines2:
                break
        else:
            lines2.append(line)
    text2 = "\n".join(lines2)
    
    print("\n" + "=" * 60)
    print("正在评估中，请稍候...")
    print("=" * 60)
    
    # 初始化评估器
    embedding_model = DEFAULT_SETTINGS.get_embedding_model()
    evaluator = Evaluator(
        embedding_model,
        skip_bert_score=False,
        bert_model_path=DEFAULT_SETTINGS.embedding_model_dir,
        use_llm_metrics=DEFAULT_SETTINGS.use_llm_metrics,
    )
    
    # 执行评估
    metrics = evaluator.evaluate_all(text1, text2)
    
    # 输出结果
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    
    print("\n【传统指标】")
    print(f"  ROUGE-L F1:          {metrics.get('rouge-l-f1', 'N/A')}")
    print(f"  BERTScore F1:        {metrics.get('bert-score-f1', 'N/A')}")
    print(f"  Embedding相似度:     {metrics.get('embedding-similarity', 'N/A')}")
    
    print("\n【LLM多维度打分】(0-10分)")
    print(f"  语义相似度:          {metrics.get('llm_semantic', 'N/A')}")
    print(f"  情感倾向:            {metrics.get('llm_emotion', 'N/A')}")
    print(f"  立场观点:            {metrics.get('llm_stance', 'N/A')}")
    print(f"  表达风格:            {metrics.get('llm_style', 'N/A')}")
    print(f"  关注焦点:            {metrics.get('llm_focus', 'N/A')}")
    print(f"  ----------------------")
    print(f"  平均分:              {metrics.get('llm_average', 'N/A')}")
    
    rationale = metrics.get('llm_rationale', '')
    if rationale:
        print(f"\n【打分理由】\n  {rationale}")
    
    print("\n" + "=" * 60)
    print("评估完成")
    print("=" * 60)


if __name__ == "__main__":
    interactive_evaluate()

