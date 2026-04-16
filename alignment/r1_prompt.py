# ==============================================================================
# DeepSeek-R1 风格思维链提示模板
# ==============================================================================
# 功能概述：
#   定义数学推理的 system prompt 和 ChatML 格式化函数。
#   采用 "Think-then-Answer" 范式：先在 <think> 标签内推理，再在 \\boxed{} 中给出答案。
#   format_r1_prompt() 将 system + user 消息格式化为 ChatML token 列表。
# ==============================================================================
import os

# Deepseek-R1 Prompt Template
class R1PromptTemplate:
    def __init__(self, template_path: os.PathLike):
        with open(template_path, "r") as f:
            self.template = f.read().strip()

    def gen_all_corpus(self, question: str, think: str, answer: str) -> str:
        return (
            self.template.replace(r"{question}", question)
            + think
            + "</think>"
            + " <answer>"
            + answer
            + " </answer>"
        )

    def gen_prompt(self, question: str) -> str:
        return self.template.replace(r"{question}", question)

    def gen_response(self, think: str, answer: str) -> str:
        return think + "</think>" + " <answer>" + answer + " </answer>"
