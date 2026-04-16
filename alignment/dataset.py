# ==============================================================================
# SFT 训练数据集模块
# ==============================================================================
# 功能概述：
#   实现 GSM8K 数学推理数据集的加载和格式化。
#   使用 ChatML 格式模板（<|im_start|>system/user/assistant<|im_end|>）。
#   SFTDataset：加载 GSM8K -> tokenize -> 构建 input_ids + labels
#   labels 中 prompt 部分 mask 为 -100，只对 answer 部分计算损失。
# ==============================================================================
from torch.utils.data import Dataset
import json
from .r1_prompt import R1PromptTemplate


class Gsm8kDataset(Dataset):
    def __init__(self, data_path: str, promt_template_path: str):
        template = R1PromptTemplate(promt_template_path)
        self.data = []
        self.label = []
        self.ground_truth = []
        with open(data_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            qa = json.loads(line)
            question = qa["question"]
            answer_think = qa["answer"]
            think, answer = answer_think.split("####")
            think, answer = think.strip(), answer.strip()
            self.data.append(template.gen_prompt(question))
            self.label.append(template.gen_response(think, answer))
            self.ground_truth.append(answer)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx], self.ground_truth[idx]
