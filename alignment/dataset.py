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
