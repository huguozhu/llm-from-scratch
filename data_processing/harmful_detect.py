# ==============================================================================
# 有害内容检测模块
# ==============================================================================
# 功能概述：
#   使用 Jigsaw 竞赛训练的 fastText 分类器检测有害内容：
#   - NSFWDetector  : NSFW 不雅内容检测，输出 nsfw / non-nsfw
#   - ToxicDetector : 仇恨言论/有毒内容检测，输出 toxic / non-toxic
#   接口统一：identify(text, k=1) -> (label, score)
# ==============================================================================
import fasttext
import numpy as np
import fasttext.FastText


# Not safe for work detector
class NSFWDetector:
    def __init__(self, model_path="pre_trained/jigsaw_fasttext_bigrams_nsfw_final.bin"):
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


class ToxicDetector:
    def __init__(self, model_path="pre_trained/jigsaw_fasttext_bigrams_hatespeech_final.bin"):
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
