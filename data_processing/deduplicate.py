# ==============================================================================
# 数据去重模块
# ==============================================================================
# 功能概述：
#   实现两种文本去重策略：
#   1. 精确行去重（exact_line_deduplicate）：哈希统计，只保留全局唯一行
#   2. MinHash 近似去重（MinHashDeduplicator）：
#      文本规范化 -> n-gram shingling -> MinHash 签名 -> LSH 分桶 -> Jaccard 验证
#      默认 Jaccard 阈值 0.8，超过即认为重复
#   MinHash 算法原理：num_hashes 个哈希函数取最小值作为签名，
#   分成 num_bands 个带做 LSH 桶键，任一带匹配则为候选对。
# ==============================================================================
import os
import hashlib
import unicodedata
from unidecode import unidecode
import nltk
from nltk.tokenize import word_tokenize
import re


def exact_line_deduplicate(files: list[os.PathLike], output_dir: os.PathLike):
    line_cnt = {}
    for file in files:
        with open(file) as f:
            for line in f.readlines():
                line_hash = hash(line)
                if line_hash in line_cnt:
                    line_cnt[line_hash] += 1
                else:
                    line_cnt[line_hash] = 1

    for file in files:
        with open(file) as f:
            with open(os.path.join(output_dir, os.path.basename(file)), "a") as f_out:
                for line in f.readlines():
                    line_hash = hash(line)
                    if line_cnt[line_hash] == 1:
                        f_out.write(line)


class Hasher:
    def __init__(self, seed: int):
        self.seed = seed

    def hash(self, content: str) -> int:
        hasher = hashlib.sha256()
        hasher.update(str(self.seed).encode("utf-8"))
        hasher.update(content.encode("utf-8"))
        return int.from_bytes(hasher.digest(), "big", signed=False)


class MinHashDeduplicator:
    def __init__(self, num_hashes: int, num_bands: int, n_gram: int, jaccard_threshold: float = 0.8):
        self.num_hashes = num_hashes
        self.num_bands = num_bands
        assert num_hashes % num_bands == 0
        self.n_gram = n_gram
        self.hasher = [Hasher(i) for i in range(num_hashes)]
        self.jaccard_threshold = jaccard_threshold

    @staticmethod
    def normalize(text: str) -> str:
        # 1. Apply Unicode NFD normalization
        text = unicodedata.normalize("NFD", text)

        # 2. Remove accents/diacritics (convert to closest ASCII)
        text = unidecode(text)

        # 3. Lowercase
        text = text.lower()

        # 4. Remove punctuation, leave：a-z, 0-9, space
        text = re.sub(r"[^a-z0-9\s]", " ", text)

        # 5. Normalize whitespace: 多个空白符 → 单空格，strip首尾
        text = re.sub(r"\s+", " ", text).strip()

        return text

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        nltk.download("punkt")
        nltk.download("punkt_tab")
        tokens = word_tokenize(text)
        return tokens

    @staticmethod
    def shingle(text: str, n_gram: int) -> set[str]:
        tokens = MinHashDeduplicator._tokenize(text)
        shingles = set()
        for i in range(len(tokens) - n_gram + 1):
            shingles.add(" ".join(tokens[i : i + n_gram]))
        return shingles

    def signatures(self, shingle: set[str]) -> list[int]:
        signature = []
        for hasher in self.hasher:
            min_hash = min([hasher.hash(shingle) for shingle in shingle])
            signature.append(min_hash)
        return signature

    def jaccard_similarity(self, f1: os.PathLike, f2: os.PathLike) -> float:
        with open(f1) as f:
            content1 = self.normalize(f.read())
        with open(f2) as f:
            content2 = self.normalize(f.read())
        shingles1 = self.shingle(content1, self.n_gram)
        shingles2 = self.shingle(content2, self.n_gram)
        return len(shingles1 & shingles2) / len(shingles1 | shingles2)

    def deduplicate(self, files: list[os.PathLike], output_dir: os.PathLike):
        signatures = {}
        buckets = {}
        for file in files:
            with open(file) as f:
                content = self.normalize(f.read())
            shingles = self.shingle(content, self.n_gram)
            signatures = self.signatures(shingles)
            for i in range(self.num_hashes - self.num_bands):
                bucket_id = (
                    i,
                    tuple(signatures[i : i + self.num_bands]),
                )
                if bucket_id not in buckets:
                    buckets[bucket_id] = []
                buckets[bucket_id].append(file)

        cadidates = set()
        for fs in buckets.values():
            for i in range(len(fs)):
                for j in range(i + 1, len(fs)):
                    cadidate = tuple(sorted((fs[i], fs[j])))
                    cadidates.add(cadidate)

        deduplicates = set()
        for f1, f2 in cadidates:
            if f1 in deduplicates or f2 in deduplicates:
                continue
            if self.jaccard_similarity(f1, f2) >= self.jaccard_threshold:
                deduplicates.add(f1)

        for file in files:
            if file in deduplicates:
                continue
            with open(file) as f:
                content = f.read()
            with open(os.path.join(output_dir, os.path.basename(file)), "a") as f_out:
                f_out.write(content)
