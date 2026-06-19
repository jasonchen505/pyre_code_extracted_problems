# Byte-Pair Encoding (BPE)

**中文标题:** 字节对编码（BPE）

**难度:** Hard

**函数名:** `SimpleBPE`

## 英文描述

Implement Byte-Pair Encoding (BPE) tokenization.

BPE iteratively merges the most frequent adjacent symbol pairs to build a subword vocabulary, widely used in modern language models.

**Signature:** `SimpleBPE()` (class)

**Methods:**
- `train(corpus: list[str], num_merges: int)` — learn merge rules from a word list
- `encode(text: str) -> list[str]` — tokenize text into subword tokens

**Constraints:**
- Append `</w>` to mark word boundaries
- `self.merges` stores learned pairs in order
- `encode` applies merges sequentially

## 中文描述

实现字节对编码（BPE）分词。

BPE 通过迭代合并最频繁的相邻符号对来构建子词词表，广泛用于现代语言模型。

**签名:** `SimpleBPE()`（类）

**方法:**
- `train(corpus: list[str], num_merges: int)` — 从词列表中学习合并规则
- `encode(text: str) -> list[str]` — 将文本分词为子词 token

**约束:**
- 使用 `</w>` 标记词边界
- `self.merges` 按顺序存储学习到的合并对
- `encode` 按顺序应用合并规则

## 提示

```
train: split each word → chars + `</w>`, count adjacent pairs, merge most frequent, repeat
encode: apply learned merges in order to split text into subword tokens
```

## 参考答案

```python
class SimpleBPE:
    def __init__(self):
        self.merges = []

    def train(self, corpus, num_merges):
        vocab = {}
        for word in corpus:
            symbols = tuple(word) + ('</w>',)
            vocab[symbols] = vocab.get(symbols, 0) + 1
        self.merges = []
        for _ in range(num_merges):
            pairs = {}
            for word, freq in vocab.items():
                for i in range(len(word) - 1):
                    pair = (word[i], word[i + 1])
                    pairs[pair] = pairs.get(pair, 0) + freq
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self.merges.append(best)
            new_vocab = {}
            for word, freq in vocab.items():
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i + 1]) == best:
                        new_word.append(word[i] + word[i + 1])
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_vocab[tuple(new_word)] = freq
            vocab = new_vocab

    def encode(self, text):
        all_tokens = []
        for word in text.split():
            symbols = list(word) + ['</w>']
            for a, b in self.merges:
                i = 0
                while i < len(symbols) - 1:
                    if symbols[i] == a and symbols[i + 1] == b:
                        symbols = symbols[:i] + [a + b] + symbols[i + 2:]
                    else:
                        i += 1
            all_tokens.extend(symbols)
        return all_tokens
```

## 示例代码

```python
bpe = SimpleBPE()
bpe.train(['low', 'low', 'low', 'lower', 'newest', 'widest'], num_merges=10)
print('Merges:', bpe.merges)
print('Encode:', bpe.encode('low lower newest'))
```

