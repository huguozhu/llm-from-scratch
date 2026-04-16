# ==============================================================================
# 语种识别模块
# ==============================================================================
# 功能概述：
#   使用 Facebook/Meta 的 fastText 预训练模型（lid.176.bin）识别文本语种。
#   支持 176 种语言，返回语种代码（如 "en", "zh"）和置信度分数。
#   包含 NumPy 2.0 兼容性补丁（修复 np.array(..., copy=False) 被移除的问题）。
# ==============================================================================
import fasttext
import numpy as np
import fasttext.FastText


# Monkey-patch fasttext to fix NumPy 2.0 incompatibility
def _patched_predict(self, text, k=1, threshold=0.0, on_unicode_error="strict"):
    """Patched version of FastText.predict to avoid np.array(..., copy=False)"""

    def check(entry):
        if entry.find("\n") != -1:
            raise ValueError("predict processes one line at a time (remove '\\n')")
        entry += "\n"
        return entry

    if type(text) is list:
        text = [check(entry) for entry in text]
        all_labels, all_probs = self.f.multilinePredict(text, k, threshold, on_unicode_error)

        return all_labels, all_probs
    else:
        text = check(text)
        predictions = self.f.predict(text, k, threshold, on_unicode_error)
        if predictions:
            probs, labels = zip(*predictions)
        else:
            probs, labels = ([], ())

        return labels, np.array(probs)


fasttext.FastText._FastText.predict = _patched_predict


class LanguageIdentifier:
    def __init__(self, model_path="pre_trained/lid.176.bin"):
        self.model = fasttext.load_model(model_path)
        self.label_prefix = "__label__"

    def identify(self, text: str, k=1):
        text = text.replace("\n", " ")
        text = text.replace("\r", " ")
        label, probs = self.model.predict(text, k)
        label = [l.replace(self.label_prefix, "") for l in label]
        if k == 1:
            return label[0], probs[0]
        else:
            return label, probs
