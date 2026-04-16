# ==============================================================================
# BPE 分词器（Byte Pair Encoding Tokenizer）
# ==============================================================================
# 功能概述：
#   从零实现的 BPE（字节对编码）分词器，用于将原始文本转化为 token ID 序列。
#   BPE 是 GPT 系列模型使用的核心分词算法。
#
# 核心原理：
#   1. 初始词表包含 256 个单字节 token（覆盖所有 UTF-8 字节值）+ 特殊 token
#   2. 训练阶段：统计语料中相邻 token 对的出现频率，反复合并最高频的一对
#      形成新 token，直到词表达到目标大小
#   3. 编码阶段：对输入文本先做预分词（正则匹配），再在每个片段上
#      按合并优先级（rank）反复执行最优合并，最终得到 token ID 序列
#   4. 解码阶段：直接将 token ID 映射回字节序列再做 UTF-8 解码
#
# 预分词正则（GPT-2 风格）：
#   r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s"""
#   将文本切分为：缩写（'s, 'll 等）、单词、数字、标点、空白
#
# 包含功能：
#   - train()        : 在语料上训练 BPE 词表和合并规则
#   - encode()       : 将文本编码为 token ID 列表
#   - decode()       : 将 token ID 列表解码为文本
#   - save() / load(): 持久化/加载词表和合并规则（pickle 格式）
#   - encode_iterable(): 流式编码（生成器），适合处理大规模数据
#   - inspect_data() : 调试工具，检查训练数据的分词效果
#
# 数据结构：
#   - vcab2id     : dict[bytes, int]  词表正向映射（token 字节串 -> ID）
#   - id2vcab     : dict[int, bytes]  词表反向映射（ID -> token 字节串）
#   - merges      : list[tuple[bytes, bytes]]  有序合并规则列表
#   - merge_ranks : dict[tuple[bytes, bytes], int]  合并优先级（rank 越小优先级越高）
#   - special_tokens: 特殊 token 列表，如 <|endoftext|>，在预分词时单独处理

# 操作	        时间复杂度	空间复杂度	说明
# train()	    O(V * N)	O(N)	    V=词表大小, N=语料大小
# encode()	    O(L * M)	O(L)	    L=文本长度, M=平均合并次数
# decode()	    O(T)	    O(T)	    T=token数量
# _pre_token()	O(L)	    O(L)	    正则匹配开销较大


#       与标准BPE的对比
# 特性	        标准BPE	        本实现
# 基础单元	    字符	字节         （UTF-8）
# 预分词   	    空格分词	        GPT-2正则
# 未登录词处理	<UNK>	        字节级fallback
# 特殊token	    需特殊处理	    预分词阶段隔离
# 编码算法	    贪心最长匹配	    rank优先合并

# ==============================================================================
from collections.abc import Iterator
import regex as re
import os
import pickle

from args import get_parser


class BpeTokenizer:
    """
    BPE 分词器：支持训练、编码、解码的完整实现。
    """
    def __init__(self, special_tokens: list[str] | None = None, errors="replace"):
        """
        初始化分词器。

        参数：
            special_tokens : 特殊 token 列表（如 ["<|endoftext|>"]），会被排在词表最前面
            errors         : UTF-8 编解码错误处理策略（默认 "replace"）
        """
        # GPT-2 风格的预分词正则：将文本切分为缩写、单词、数字、标点、空白
        # 匹配：
        # - 's, 'd, 't, 'll, 've, 're (英文缩写)
        # - 单词（可选前导空格）
        # - 数字（可选前导空格）
        # - 标点符号（可选前导空格）
        # - 连续空白
        self.pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s"""
        self.errors = errors
        self.vcab2id = dict[bytes, int]()                    # token 字节串 -> ID
        self.id2vcab = dict[int, bytes]()                    # ID -> token 字节串
        self.merges = list[tuple[bytes, bytes]]()            # 有序合并规则列表
        self.merge_ranks = dict[tuple[bytes, bytes], int]()  # 合并对 -> 优先级 rank
        # 特殊 token 按长度降序排列，确保较长的特殊 token 优先匹配
        self.special_tokens = sorted(
            [s.encode("utf-8", errors) for s in special_tokens] if special_tokens else [], key=len, reverse=True
        )

    def from_pretrained(
        self,
        id2vcab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | list[bytes] | None = None,
    ):
        self.id2vcab = id2vcab
        self.merges = merges
        if special_tokens:

            def is_list_of_str(lst):
                return isinstance(lst, list) and all(isinstance(item, str) for item in lst)

            if is_list_of_str(special_tokens):
                self.special_tokens = [s.encode("utf-8", self.errors) for s in special_tokens]
            else:
                self.special_tokens = special_tokens
        self.vcab2id = {v: k for k, v in self.id2vcab.items()}
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

    # 预分词
    def _pre_token(self, corpus: bytes) -> list[bytes]:
        if not self.special_tokens:
            return [
                match.group(0).encode("utf-8", self.errors)
                for match in re.finditer(self.pattern, corpus.decode("utf-8", self.errors))
            ]

        # 1. 处理特殊token：用正则split隔离
        pattern = b"|".join(map(re.escape, self.special_tokens))
        parts = re.split(b"(" + pattern + b")", corpus)

        # 2. 对非特殊token部分应用GPT-2正则
        final_parts = []
        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                final_parts.append(part)
            else:
                # 使用GPT-2风格正则切分
                final_parts.extend(
                    [
                        match.group(0).encode("utf-8", self.errors)
                        for match in re.finditer(self.pattern, part.decode("utf-8", self.errors))
                    ]
                )
        return final_parts

    # 预分词后每个片段独立编码
    # 贪心算法：每次选择优先级最高的合并
    # 特殊token直接映射，不参与BPE合并
    def encode(self, text: str) -> list[int]:
        bs = text.encode("utf-8", self.errors)
        pre_tokens = self._pre_token(bs)

        token_ids = []
        for pre_token in pre_tokens:
            if pre_token in self.special_tokens:
                token_ids.append(self.vcab2id[pre_token])
                continue

            # 初始化为单个字节的token
            tokens = tuple(bytes([c]) for c in pre_token)
            # 贪心合并：每次合并rank最小的pair
            while len(tokens) > 1:
                pairs = list(zip(tokens[:-1], tokens[1:]))
                # Find the merge with the lowest rank
                # 找到最小rank的pair
                rank = float("inf")
                best_pair_idx = -1
                for i, pair in enumerate(pairs):
                    if pair in self.merge_ranks and self.merge_ranks[pair] < rank:
                        rank = self.merge_ranks[pair]
                        best_pair_idx = i

                if best_pair_idx == -1:
                    break

                # Merge the best pair
                new_tokens = []
                if best_pair_idx > 0:
                    new_tokens.extend(tokens[:best_pair_idx])
                new_tokens.append(tokens[best_pair_idx] + tokens[best_pair_idx + 1])
                if best_pair_idx + 2 < len(tokens):
                    new_tokens.extend(tokens[best_pair_idx + 2 :])
                tokens = tuple(new_tokens)

            for vcab in tokens:
                token_ids.append(self.vcab2id[vcab])
        return token_ids

    def encode_iterable(self, iterable: Iterator[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        # 1. token ID -> 字节串
        vcabs = [self.id2vcab.get(i, b"\xef\xbf\xbd") for i in token_ids]
        # 2. 拼接所有字节串
        bs = b"".join(vcabs)
        # 3. UTF-8解码
        return bs.decode("utf-8", self.errors)

    def save(self, out: str) -> None:
        obj = {"merge": self.merges, "id2vcab": self.id2vcab, "special_tokens": self.special_tokens}
        with open(out, "wb") as f:
            pickle.dump(obj, f)

    def load(self, ins: str):
        with open(ins, "rb") as f:
            obj = pickle.load(f)
            merges = obj["merge"]
            id2vcab = obj["id2vcab"]
            special_tokens = obj["special_tokens"]
        self.from_pretrained(id2vcab=id2vcab, merges=merges, special_tokens=special_tokens)

    def train(self, input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str], verbose=False):
        """
        Train the BPE tokenizer on the given corpus.

        :param corpus: A binary file-like object containing the training data.
        :param vocab_size: The desired vocabulary size.
        """
        self.special_tokens = sorted(
            [s.encode("utf-8", self.errors) for s in special_tokens] if special_tokens else [], key=len, reverse=True
        )

        corpus = list[bytes]()
        with open(input_path, encoding='utf-8') as f:
            corpus = f.read().encode("utf-8", self.errors)

        # 1. 初始化词表：特殊token + 256个单字节
        for token in self.special_tokens:
            self.vcab2id[token] = len(self.vcab2id)

        for i in range(256):
            self.vcab2id[bytes([i])] = len(self.vcab2id)

        pre_tokens = self._pre_token(corpus)
        # 2. 统计词频（重要优化）
        word_cnt: dict[tuple[bytes, ...], int] = {}
        for pre_token in pre_tokens:
            if pre_token in self.special_tokens:
                continue
            bs = tuple(bytes([b]) for b in pre_token)
            if not bs:
                continue
            word_cnt[bs] = word_cnt.get(bs, 0) + 1

        if verbose:
            print(f"cnt of distinct word: {len(word_cnt)}")

        # 3. 统计所有相邻token对的频率
        # pair_cnt 实时更新，避免重复扫描整个语料
        pair_cnt = dict[tuple[bytes, bytes], int]()
        for word, cnt in word_cnt.items():
            for pair in zip(word[:-1], word[1:]):
                pair_cnt[pair] = pair_cnt.get(pair, 0) + cnt

        # 使用 update_pair_counts 增量更新，效率高
        def update_pair_counts(word: tuple[bytes, ...], pair_cnt: dict[tuple[bytes, bytes], int], cnt: int, sign=1):
            for pair in zip(word[:-1], word[1:]):
                pair_cnt[pair] = pair_cnt.get(pair, 0) + cnt * sign
                if pair_cnt.get(pair, 0) <= 0:
                    del pair_cnt[pair]

        # 4. 迭代合并直到词表达到目标大小
        while len(self.vcab2id) < vocab_size:
            if verbose:
                print(f"vocab_size = {len(self.vcab2id)}, target {vocab_size}")
            # 选择频率最高的pair（频率相同时按字节值排序）
            best_pair = max(pair_cnt.keys(), key=lambda p: (pair_cnt.get(p, 0), p))
            # 创建新token并记录合并规则
            self.vcab2id[best_pair[0] + best_pair[1]] = len(self.vcab2id)
            self.merges.append(best_pair)

            new_word_cnt: dict[tuple[bytes, ...], int] = {}
            merged = False
            # 更新所有词的分词状态
            for word, cnt in word_cnt.items():                
                new_word_list = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                        new_word_list.append(best_pair[0] + best_pair[1])
                        i += 2
                    else:
                        new_word_list.append(word[i])
                        i += 1

                new_word = tuple(new_word_list)
                if word != new_word:
                    # 更新pair_cnt：减去旧的，加上新的
                    merged = True
                    update_pair_counts(word, pair_cnt, cnt, -1)
                    update_pair_counts(new_word, pair_cnt, cnt, 1)

                new_word_cnt[new_word] = new_word_cnt.get(new_word, 0) + cnt

            if not merged:
                break
            word_cnt = new_word_cnt

        self.id2vcab = {id: vcab for vcab, id in self.vcab2id.items()}
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
        return self.id2vcab, self.merges


def inspect_data():
    """
    Loads the tokenizer and a sample of the training data,
    then prints a detailed analysis of how the data is structured and tokenized.
    """
    tokenizer_path = "data/tokenizer"
    data_path = "data/training_data.npy"

    # Load the tokenizer
    tokenizer = BpeTokenizer()
    try:
        tokenizer.load(tokenizer_path)
    except FileNotFoundError:
        print(f"Error: Tokenizer file not found at '{tokenizer_path}'")
        return
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # Load a sample of the training data
    try:
        token_ids = np.load(data_path)
        sample_token_ids = token_ids[:200]  # Take the first 200 tokens
    except FileNotFoundError:
        print(f"Error: Training data file not found at '{data_path}'")
        return
    except Exception as e:
        print(f"Error reading data: {e}")
        return

    # Decode the sample
    sample_text = tokenizer.decode(sample_token_ids.tolist())
    decoded_tokens = [tokenizer.decode([token_id]) for token_id in sample_token_ids]

    # Print the analysis
    print("--- Data Inspection ---")
    print(f"Tokenizer: {tokenizer_path}")
    print(f"Data sample: {data_path}")
    print("-" * 20)
    print(f"Vocabulary size: {len(tokenizer.vcab2id)}")
    special_tokens_decoded = [st.decode("utf-8", errors="ignore") for st in tokenizer.special_tokens]
    print(f"Special tokens: {special_tokens_decoded}")
    if special_tokens_decoded:
        endoftext_token = special_tokens_decoded[0]
        endoftext_id = tokenizer.vcab2id[tokenizer.special_tokens[0]]
        print(f"The ID for '{endoftext_token}' is: {endoftext_id}")

    print("-" * 20)
    print("Sample Token IDs:")
    print(sample_token_ids)
    print("-" * 20)
    print("Decoded Text:")
    print(sample_text)
    print("-" * 20)
    print("Decoded Tokens (one by one):")
    print(decoded_tokens)
    print("--- End of Inspection ---")


if __name__ == "__main__":
    # train_data_file = os.path.join("data", "TinyStoriesV2-GPT4-train.txt")
    import numpy as np

    parser = get_parser()
    args = parser.parse_args()
    vocab_size = args.vocab_size
    tokenizer = BpeTokenizer(special_tokens=["<|endoftext|>"])
    print("start traning")
    tokenizer.train(args.train_source_file, vocab_size=vocab_size, special_tokens=["<|endoftext|>"], verbose=True)
    print("end traning")
    tokenizer.save(args.tokenizer_checkpoint)
    print(f"vocab size: {len(tokenizer.vcab2id)}")

    with open(args.train_source_file) as f:
        print("starting encoding train text to token ids")
        token_ids = tokenizer.encode(f.read())
        print("start persisting train tokens ids")
        np.save(args.train_data, np.array(token_ids))

    with open(args.valid_source_file) as f:
        print("starting encoding valid text to token ids")
        token_ids = tokenizer.encode(f.read())
        print("start persisting valid tokens ids")
        np.save(args.val_data, np.array(token_ids))

    inspect_data()
