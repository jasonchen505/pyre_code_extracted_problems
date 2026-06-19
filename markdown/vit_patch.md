# ViT Patch Embedding

**中文标题:** ViT Patch Embedding

**难度:** Medium

**函数名:** `PatchEmbedding`

## 英文描述

Implement ViT patch embedding as an nn.Module.

Patch embedding splits an image into fixed-size patches and projects each patch into an embedding vector, forming the input sequence for Vision Transformers.

**Signature:** `PatchEmbedding(img_size, patch_size, in_channels, embed_dim)` (nn.Module)

**Forward:** `forward(x) -> Tensor`
- `x` — image tensor (B, C, H, W)

**Returns:** patch embeddings (B, num_patches, embed_dim)

**Constraints:**
- `num_patches = (img_size / patch_size)^2`
- Store `self.num_patches` as an attribute
- Project with `nn.Linear(C * P * P, embed_dim)`

## 中文描述

实现 ViT 图像块嵌入（nn.Module）。

图像块嵌入将图像分割为固定大小的块，并将每个块投影为嵌入向量，形成 Vision Transformer 的输入序列。

**签名:** `PatchEmbedding(img_size, patch_size, in_channels, embed_dim)`（nn.Module）

**前向传播:** `forward(x) -> Tensor`
- `x` — 图像张量 (B, C, H, W)

**返回:** 图像块嵌入 (B, num_patches, embed_dim)

**约束:**
- `num_patches = (img_size / patch_size)^2`
- 将 `self.num_patches` 存储为属性
- 使用 `nn.Linear(C * P * P, embed_dim)` 投影

## 提示

```
`(B,C,H,W)` → reshape to `(B, n_h, P, n_w, P, C)` → permute → `(B, num_patches, C*P*P)` → `nn.Linear(C*P*P, embed_dim)`.
```

## 参考答案

```python
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        n_h, n_w = H // p, W // p
        x = x.reshape(B, C, n_h, p, n_w, p)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, n_h * n_w, C * p * p)
        return self.proj(x)
```

## 示例代码

```python
pe = PatchEmbedding(224, 16, 3, 768)
x = torch.randn(1, 3, 224, 224)
print('Output:', pe(x).shape)
print('Patches:', pe.num_patches)
```

