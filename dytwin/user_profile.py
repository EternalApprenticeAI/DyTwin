"""动态用户画像

需求：
- 初始为空
- 随时间推进，通过LLM对【原创】博文进行分析，更新画像
- 画像结构为多行自然语言文本，每行是一个独立的描述句

这里实现：
- UserProfile: 持有多行文本列表
- apply_patch: 支持添加、修改、删除描述句
- snapshot: 返回当前画像文本
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class UserProfile:
    """多行自然语言文本形式的用户画像
    
    每行是一个独立的描述句，例如：
    - "确定喜爱狗狗，推测喜爱宠物"
    - "表达风格直接犀利，善用讽刺"
    - "关注医疗教育领域，对学阀现象持批判态度"
    """
    sentences: List[str] = field(default_factory=list)

    def apply_patch(self, patch: Dict[str, Any]) -> None:
        """将LLM给出的更新应用到画像。

        patch格式：
        {
            "add": ["新增的描述句1", "新增的描述句2", ...],      # 添加新描述
            "modify": {"完整旧描述句": "完整新描述句", ...},      # 修改已有描述（完整匹配）
            "remove": ["要删除的完整描述句1", ...],               # 删除描述（完整匹配）
            "merge": [                                           # 归并多条描述为一条
                {
                    "old_sentences": ["完整旧描述句1", "完整旧描述句2", ...],
                    "new_sentence": "归并后的新描述句"
                },
                ...
            ]
        }
        
        注意：modify/remove/merge操作均要求完整匹配画像中的描述句，不支持部分匹配。
        """
        if not patch:
            return

        # 记录已处理的句子，确保操作隔离性（同一条句子最多被一个操作处理）
        processed_sentences = set()
        
        # 第1步：处理归并（merge）- 将多条旧描述合并为一条新描述
        merge_list = patch.get("merge", [])
        if merge_list:
            for merge_item in merge_list:
                if not isinstance(merge_item, dict):
                    continue
                # 兼容新旧格式：优先使用old_sentences，兼容old_fragments
                old_sentences = merge_item.get("old_sentences", merge_item.get("old_fragments", []))
                new_sentence = str(merge_item.get("new_sentence", "")).strip()
                if not old_sentences or not new_sentence:
                    continue
                
                # 检查是否有句子已被其他操作处理
                conflict_found = False
                for old_sent in old_sentences:
                    old_sent = str(old_sent).strip()
                    if old_sent in processed_sentences:
                        conflict_found = True
                        break
                
                if conflict_found:
                    continue  # 跳过有冲突的merge操作
                
                # 删除完整匹配的旧描述句
                for old_sent in old_sentences:
                    old_sent = str(old_sent).strip()
                    if old_sent:
                        self.sentences = [s for s in self.sentences if s != old_sent]
                        processed_sentences.add(old_sent)
                
                # 添加归并后的新句子
                if new_sentence not in self.sentences:
                    self.sentences.append(new_sentence)

        # 第2步：处理修改（modify）- 对单条描述进行精确修改
        modify_dict = patch.get("modify", {})
        if modify_dict:
            for old_sentence, new_sentence in modify_dict.items():
                old_sentence = str(old_sentence).strip()
                new_sentence = str(new_sentence).strip()
                if not old_sentence or not new_sentence:
                    continue
                
                # 检查操作隔离性
                if old_sentence in processed_sentences:
                    continue  # 跳过已被处理的句子
                
                # 完整匹配替换
                for i, s in enumerate(self.sentences):
                    if s == old_sentence:
                        self.sentences[i] = new_sentence
                        processed_sentences.add(old_sentence)
                        break

        # 第3步：处理添加（add）- 补充画像中缺失的新维度或特征
        add_list = patch.get("add", [])
        if add_list:
            for sentence in add_list:
                sentence = str(sentence).strip()
                if sentence and sentence not in self.sentences:
                    self.sentences.append(sentence)

        # 第4步：处理删除（remove）- 删除明确错误或过时的描述
        remove_list = patch.get("remove", [])
        if remove_list:
            for remove_sent in remove_list:
                remove_sent = str(remove_sent).strip()
                if not remove_sent:
                    continue
                
                # 检查操作隔离性
                if remove_sent in processed_sentences:
                    continue  # 跳过已被处理的句子
                
                # 完整匹配删除
                self.sentences = [s for s in self.sentences if s != remove_sent]
                processed_sentences.add(remove_sent)

    def snapshot(self) -> str:
        """返回当前画像的多行文本表示"""
        return "\n".join(self.sentences) if self.sentences else ""
    
    def restore_from_snapshot(self, snapshot: str) -> None:
        """从快照字符串恢复画像状态"""
        if not snapshot or not snapshot.strip():
            self.sentences = []
            return
        self.sentences = [s.strip() for s in snapshot.split("\n") if s.strip()]
    
    def to_list(self) -> List[str]:
        """返回画像句子列表"""
        return list(self.sentences)

