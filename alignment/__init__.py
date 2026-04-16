# ==============================================================================
# alignment 包初始化模块
# ==============================================================================
# 功能概述：
#   LLM 对齐（Alignment）训练工具包，实现从预训练模型到可用模型的后训练流程：
#   - sft      : 监督微调（Supervised Fine-Tuning），教会模型遵循指令
#   - grpo     : GRPO 强化学习（Group Relative Policy Optimization）
#   - train_rl : RL 训练主循环，基于 GRPO 优化数学推理能力
#   - evaluate : 模型评估（GSM8K 数学基准测试）
#   - r1_prompt: DeepSeek-R1 风格的思维链提示模板
#   - util     : 通用工具（文本生成、答案提取、奖励计算等）
# ==============================================================================
from .evaluate import *
from .r1_prompt import *
from .util import *
from .sft import *
from .grpo import *
from .train_rl import *
