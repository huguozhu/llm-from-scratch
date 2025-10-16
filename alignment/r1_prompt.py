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
