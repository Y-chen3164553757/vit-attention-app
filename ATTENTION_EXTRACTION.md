# DINOv3 ViT 注意力特征图提取原理

> 🌐 Language / 语言：**中文** | [English](ATTENTION_EXTRACTION_EN.md)

## 概述

本项目从 DINOv3 ViT-S/16 模型中提取 **CLS token 对图像各区域的注意力权重**，将其可视化为热力图。模型共 12 层、每层 6 个注意力头，最终生成 **72 张注意力特征图**。

![3D 可视化预览](images/preview-3d.png)

---

## 完整提取流程

### 第一步：图像预处理

```text
原始图片 (任意尺寸) → Resize(512×512) → ToFloat32 → Normalize
```

- 输入图片调整为 **512×512**（与 `patch_size=16` 整除对齐，512 / 16 = 32）
- 归一化参数采用 ImageNet 标准：
  - `mean = (0.485, 0.456, 0.406)`
  - `std  = (0.229, 0.224, 0.225)`
- 最终得到形状为 `[1, 3, 512, 512]` 的张量

<details>
<summary>📌 对应代码（model.py）</summary>

```python
transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
```

</details>

---

### 第二步：Patch 分割与嵌入

ViT 将图片切分为不重叠的 **16×16 像素 patch**：

```math
\text{grid\_h} = \frac{512}{16} = 32, \quad \text{grid\_w} = \frac{512}{16} = 32
```

共 **32 × 32 = 1024** 个 patch token。每个 patch 通过 `PatchEmbed` 线性层映射为 **384 维向量**（ViT-S 的隐藏维度 C = 384）。

加上模型的特殊 token：

| Token 类型 | 数量 | 说明 |
|:---|:---:|:---|
| CLS token | 1 | 全局类别聚合标记 |
| Register token | 4 | DINOv3 引入的辅助标记，缓解注意力伪影 |
| Patch token | 1024 | 图像区域表征 |

> 总序列长度：**N = 1 + 4 + 1024 = 1029**
>
> 输入 Transformer 的张量形状：`[1, 1029, 384]`

---

### 第三步：Transformer Block 中的 Self-Attention 计算

模型包含 **12 层 Transformer Block**，每层执行以下自注意力计算：

#### 3.1 QKV 线性变换

对输入 `[1, 1029, 384]`，通过一个联合线性层得到 Q、K、V：

```math
\text{QKV} = X \cdot W_{qkv} \in \mathbb{R}^{1 \times 1029 \times 1152}
```

其中权重维度为 `[384, 1152]`，1152 = 384 × 3。

#### 3.2 拆分为多头

重塑并拆分为 6 个注意力头：

```math
\text{QKV} \xrightarrow{\text{reshape}} [1, 1029, 3, 6, 64] \xrightarrow{\text{unbind}} Q, K, V \in \mathbb{R}^{1 \times 6 \times 1029 \times 64}
```

每个头的维度：`d_k = 384 / 6 = 64`。

<details>
<summary>📌 对应代码（model.py hook 内）</summary>

```python
qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, head_dim)
q, k, v = [t.transpose(1, 2) for t in torch.unbind(qkv, 2)]
```

</details>

#### 3.3 RoPE 旋转位置编码

对 Q 和 K 的 **patch token 部分**（跳过前 5 个 prefix token）应用旋转位置编码（Rotary Position Embedding）：

```math
q' = q \cdot \cos(\theta) + \text{rotate\_half}(q) \cdot \sin(\theta)
```

```math
k' = k \cdot \cos(\theta) + \text{rotate\_half}(k) \cdot \sin(\theta)
```

其中 `rotate_half` 将向量的前后两半交换并取负：

`[x0, x1, x2, x3]` → `[-x2, -x3, x0, x1]`

RoPE 为每个 patch 注入了二维空间位置信息，使模型能感知 token 间的相对位置。

<details>
<summary>📌 对应代码（attention.py）</summary>

```python
def rope_apply(x, sin, cos):
    return (x * cos) + (rope_rotate_half(x) * sin)
```

</details>

#### 3.4 计算注意力权重矩阵

```math
A = \text{softmax}\!\left(\frac{Q \cdot K^T}{\sqrt{d_k}}\right) = \text{softmax}\!\left(\frac{Q \cdot K^T}{\sqrt{64}}\right) = \text{softmax}\!\left(\frac{Q \cdot K^T}{8}\right)
```

注意力矩阵维度：`[1, 6, 1029, 1029]`

> **💡 为什么需要 Hook 手动计算？**
>
> DINOv3 原始代码中使用 PyTorch 的 `scaled_dot_product_attention` 融合算子（`attention.py` 第 119 行）。该算子将 QK 缩放、softmax、与 V 相乘全部融合为一步计算，**不暴露中间的注意力权重矩阵**。因此我们通过 `register_forward_hook` 拦截输入，手动分步计算以获得注意力矩阵 A。

<details>
<summary>📌 对应代码（model.py hook）</summary>

```python
def _attention_hook(self, module, input_args, kwargs, output):
    x = input_args[0]  # [B, N, C]
    rope = kwargs.get("rope", None)

    B, N, C = x.shape
    head_dim = C // module.num_heads
    qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, head_dim)
    q, k, v = [t.transpose(1, 2) for t in torch.unbind(qkv, 2)]

    if rope is not None:
        q, k = module.apply_rope(q, k, rope)

    attn_weights = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
    attn_weights = F.softmax(attn_weights, dim=-1)

    cls_attn = attn_weights[:, :, 0, N_PREFIX_TOKENS:]
    self.attention_maps.append(cls_attn.detach().cpu())
```

</details>

---

### 第四步：提取 CLS → Patch 注意力

从完整的 1029 × 1029 注意力矩阵中，提取第 **0 行**（CLS token）对第 **5~1028 列**（patch token）的注意力值：

```math
A_{\text{cls}} = A[:, :, 0, 5:] \in \mathbb{R}^{1 \times 6 \times 1024}
```

- 索引 `0` → CLS token（查询方）
- 跳过前 5 列（1 CLS + 4 registers），只保留 1024 个 patch 的注意力值
- **物理含义**：每个值表示 CLS token 对对应图像区域的"关注程度"

---

### 第五步：重塑为 2D 空间图

将 1024 个 patch 的一维注意力值还原为二维空间布局：

```math
A_{\text{cls}} \xrightarrow{\text{reshape}} [6, 32, 32]
```

每个位置 `(i, j)` 对应原图中第 `(i, j)` 个 16×16 patch 区域的注意力强度。

---

### 第六步：上采样

```math
[6, 32, 32] \xrightarrow{\text{nearest}} [6, 256, 256]
```

使用最近邻插值放大到 256×256 像素。选择最近邻（而非双线性）是为了保持 patch 边界的清晰性。

<details>
<summary>📌 对应代码</summary>

```python
attn_up = F.interpolate(
    attn.unsqueeze(0),
    size=(OUTPUT_SIZE, OUTPUT_SIZE),
    mode="nearest",
)[0]  # [heads, 256, 256]
```

</details>

---

### 第七步：逐头 Min-Max 归一化

对每个注意力头独立归一化到 [0, 1]：

```math
A_h^{\text{norm}} = \frac{A_h - \min(A_h)}{\max(A_h) - \min(A_h) + \epsilon}
```

目的：不同层/不同头的注意力值范围差异很大，归一化后使每张图都具有充分的对比度。

<details>
<summary>📌 对应代码</summary>

```python
for h in range(NUM_HEADS):
    a = attn_up[h]
    lo, hi = a.min(), a.max()
    attn_up[h] = (a - lo) / (hi - lo + 1e-8)
```

</details>

---

### 第八步：前端着色与混合

后端将归一化后的 [0, 1] 灰度值转为 **PNG 灰度图**，以 base64 编码传给前端。

前端（`app.js`）使用 **Viridis 色图**将灰度值映射为伪彩色，并与原图按用户设定的透明度叠加：

```math
\text{pixel}_{\text{out}} = \text{original} \times (1 - \alpha) + \text{viridis}(v) \times \alpha
```

**Viridis 色图映射示例：**

| 灰度值 | 颜色 | RGB |
|:---:|:---|:---|
| 0 | 🟣 深紫色 | `rgb(68, 1, 84)` |
| 0.5 | 🟢 青绿色 | `rgb(34, 167, 132)` |
| 1 | 🟡 亮黄色 | `rgb(253, 231, 37)` |

---

## 总结流程图

```text
原始图片 (任意尺寸)
    │
    ▼ Resize(512×512) + Normalize(ImageNet)
[1, 3, 512, 512]
    │
    ▼ PatchEmbed(16×16) + CLS + 4 Registers
[1, 1029, 384]
    │
    ▼ × 12 Transformer Blocks
    │   每层 hook 拦截 SelfAttention:
    │     QKV = Linear(X)              → [1, 1029, 1152]
    │     Q, K, V = split + reshape    → [1, 6, 1029, 64]
    │     Q, K = RoPE(Q, K)            → 旋转位置编码
    │     A = softmax(Q·Kᵀ / 8)       → [1, 6, 1029, 1029]
    │     cls_attn = A[:,:,0,5:]       → [1, 6, 1024]
    │
    ▼ 收集 12 层注意力
12 × [6, 1024]
    │
    ▼ reshape → [6, 32, 32]
    ▼ nearest ↑ 256×256
    ▼ min-max 归一化 → [0, 1]
    │
    ▼ → PNG base64 → 前端
    ▼ Viridis 着色 + 原图叠加
    │
    ▼
72 张注意力可视化热力图
(12 layers × 6 heads)
```

---

## 关键参数

| 参数 | 值 | 说明 |
|:---|:---:|:---|
| 模型 | ViT-S/16 | Small 变体，16×16 patch |
| 隐藏维度 C | 384 | ViT-S 标准配置 |
| 注意力头数 | 6 | d_k = 384/6 = 64 |
| Transformer 层数 | 12 | 每层产生 6 张注意力图 |
| 输入分辨率 | 512×512 | 与 patch 整除 |
| Patch 大小 | 16×16 | 产生 32×32 = 1024 个 patch |
| Prefix tokens | 5 | 1 CLS + 4 registers |
| 输出图尺寸 | 256×256 | 平衡质量与传输大小 |
| 位置编码 | RoPE | 旋转位置嵌入 |
| 预训练数据 | LVD-142M | Meta 大规模数据集 |

---

## 如何解读注意力图

- 🟡 **亮色区域（黄/绿）**：CLS token 高度关注的区域，通常对应图像中的显著目标或语义核心
- 🟣 **暗色区域（紫/蓝）**：CLS token 较少关注的区域，通常是背景
- 📊 **不同层的差异**：浅层倾向关注局部纹理/边缘，深层倾向关注全局语义结构
- 🔀 **不同头的差异**：同一层的不同头可能关注不同的语义模式（例如边缘 vs 颜色 vs 形状）

![2D 全局展示](images/preview-2d.png)
