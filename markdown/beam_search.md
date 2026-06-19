# Beam Search Decoding

**中文标题:** 束搜索解码

**难度:** Medium

**函数名:** `beam_search`

## 英文描述

Implement beam search decoding for sequence generation.

Beam search maintains multiple candidate sequences (beams) at each step, expanding and pruning to find the highest-scoring sequence.

**Signature:** `beam_search(log_prob_fn, start_token, max_len, beam_width, eos_token) -> list[int]`

**Parameters:**
- `log_prob_fn` — callable that takes a token sequence tensor and returns log-probabilities over vocabulary
- `start_token` — integer start token
- `beam_width` — number of beams to keep
- `eos_token` — end-of-sequence token

**Returns:** list of token IDs for the best sequence

**Constraints:**
- Stop when all beams end with eos or max_len is reached
- Return the highest-scoring complete sequence

## 中文描述

实现序列生成的束搜索解码。

束搜索在每一步维护多个候选序列（束），通过扩展和剪枝找到得分最高的序列。

**签名:** `beam_search(log_prob_fn, start_token, max_len, beam_width, eos_token) -> list[int]`

**参数:**
- `log_prob_fn` — 接受 token 序列张量并返回词表上对数概率的可调用对象
- `start_token` — 起始 token 整数
- `beam_width` — 保留的束数量
- `eos_token` — 序列结束 token

**返回:** 最佳序列的 token ID 列表

**约束:**
- 当所有束以 eos 结尾或达到 max_len 时停止
- 返回得分最高的完整序列

## 提示

```
1. beams = [(score=0.0, seq=[start_token])]
2. each step: expand each beam with all tokens → keep top beam_width by cumulative score
3. stop when all beams end with eos_token or max_len reached
4. return seq from highest-scoring beam
```

## 参考答案

```python
def beam_search(log_prob_fn, start_token, max_len, beam_width, eos_token):
    beams = [(0.0, [start_token])]
    completed = []
    for _ in range(max_len):
        candidates = []
        for score, seq in beams:
            if seq[-1] == eos_token:
                completed.append((score, seq))
                continue
            log_probs = log_prob_fn(torch.tensor(seq))
            topk_lp, topk_idx = log_probs.topk(beam_width)
            for j in range(beam_width):
                candidates.append((score + topk_lp[j].item(), seq + [topk_idx[j].item()]))
        if not candidates:
            break
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_width]
    all_seqs = completed + beams
    all_seqs.sort(key=lambda x: x[0], reverse=True)
    return all_seqs[0][1]
```

## 示例代码

```python
def simple_fn(tokens):
    lp = torch.full((5,), -10.0)
    lp[min(len(tokens), 4)] = 0.0
    return lp
seq = beam_search(simple_fn, start_token=0, max_len=5, beam_width=2, eos_token=4)
print('Sequence:', seq)
```

