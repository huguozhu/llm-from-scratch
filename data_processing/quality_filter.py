# ==============================================================================
# 基于规则的文本质量过滤模块
# ==============================================================================
# 功能概述：
#   实现启发式规则过滤低质量文本（参考 C4/RefinedWeb 策略）：
#   1. 词数过滤：50 <= 词数 <= 100,000
#   2. 平均词长过滤：3 <= 平均词长 <= 10（过滤乱码/URL堆砌）
#   3. 字母比过滤：>= 80% token 含字母字符
#   4. 省略号过滤："..." 结尾行 < 30%（过滤截断内容）
#   使用 NLTK word_tokenize 分词。pass_all_filters() 全部通过才保留。
# ==============================================================================
import nltk
from nltk.tokenize import word_tokenize


class QualityFilter:
    def __init__(self):
        nltk.download("punkt")
        nltk.download("punkt_tab")

    def pass_wc_filter(self, tokens: list[str]) -> bool:
        wc = len(tokens)
        return 50 <= wc <= 100000

    def pass_word_len_filter(self, tokens: list[str]) -> bool:
        lens = [len(token) for token in tokens]
        avg_len = sum(lens) / len(lens)
        return 3 <= avg_len <= 10

    def pass_alphabetic_filter(self, tokens: list[str]) -> bool:
        has_alpha = [1 if any(char.isalpha() for char in token) else 0 for token in tokens]
        return 1.0 * sum(has_alpha) / len(has_alpha) >= 0.8

    def pass_ellipsis_filter(self, content: str) -> bool:
        line_cnt = 0
        ellipsis_cnt = 0
        for line in content.split("\n"):
            line_cnt += 1
            if line.endswith("..."):
                ellipsis_cnt += 1
        return ellipsis_cnt / line_cnt < 0.3

    def pass_all_filters(self, content: str) -> bool:
        tokens = word_tokenize(content)
        return (
            self.pass_wc_filter(tokens)
            and self.pass_word_len_filter(tokens)
            and self.pass_alphabetic_filter(tokens)
            and self.pass_ellipsis_filter(content)
        )
