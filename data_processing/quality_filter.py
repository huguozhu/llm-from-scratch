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
