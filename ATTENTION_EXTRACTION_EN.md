# DINOv3 ViT Attention Map Extraction — How It Works

## Overview

This project extracts the **attention weights of the CLS token over all image regions** from a DINOv3 ViT-S/16 model and visualizes them as heatmaps. The model has 12 layers and 6 attention heads per layer, producing **72 attention maps** in total.

![3D Visualization Preview](images/preview-3d.png)

---

## Full Extraction Pipeline

### Step 1: Image Pre-processing

```text
Input image (any size) → Resize(512×512) → ToFloat32 → Normalize
```

- Images are resized to **512×512** (evenly divisible by `patch_size=16`: 512 / 16 = 32)
- Normalization uses ImageNet statistics:
  - `mean = (0.485, 0.456, 0.406)`
  - `std  = (0.229, 0.224, 0.225)`
- Output tensor shape: `[1, 3, 512, 512]`

<details>
<summary>📌 Code reference (model.py)</summary>

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

### Step 2: Patch Splitting and Embedding

The ViT divides the image into non-overlapping **16×16 pixel patches**:

```math
\text{grid\_h} = \frac{512}{16} = 32, \quad \text{grid\_w} = \frac{512}{16} = 32
```

This yields **32 × 32 = 1024** patch tokens. Each patch is projected to a **384-dimensional vector** via the `PatchEmbed` linear layer (the hidden dimension C = 384 for ViT-S).

Together with the model's special tokens:

| Token type | Count | Description |
|:---|:---:|:---|
| CLS token | 1 | Global class aggregation token |
| Register token | 4 | Auxiliary tokens introduced in DINOv3 to suppress attention artifacts |
| Patch token | 1024 | Image region representations |

> Total sequence length: **N = 1 + 4 + 1024 = 1029**
>
> Tensor shape fed into the Transformer: `[1, 1029, 384]`

---

### Step 3: Self-Attention Computation in Transformer Blocks

The model contains **12 Transformer Blocks**, each performing the following self-attention computation:

#### 3.1 QKV Linear Projection

For input `[1, 1029, 384]`, a single fused linear layer produces Q, K, V jointly:

```math
\text{QKV} = X \cdot W_{qkv} \in \mathbb{R}^{1 \times 1029 \times 1152}
```

The weight matrix has shape `[384, 1152]`, where 1152 = 384 × 3.

#### 3.2 Splitting into Multiple Heads

Reshape and split into 6 attention heads:

```math
\text{QKV} \xrightarrow{\text{reshape}} [1, 1029, 3, 6, 64] \xrightarrow{\text{unbind}} Q, K, V \in \mathbb{R}^{1 \times 6 \times 1029 \times 64}
```

Per-head dimension: `d_k = 384 / 6 = 64`.

<details>
<summary>📌 Code reference (model.py hook)</summary>

```python
qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, head_dim)
q, k, v = [t.transpose(1, 2) for t in torch.unbind(qkv, 2)]
```

</details>

#### 3.3 RoPE Rotary Position Encoding

Rotary Position Embeddings (RoPE) are applied to the **patch token portion** of Q and K (the first 5 prefix tokens are skipped):

```math
q' = q \cdot \cos(\theta) + \text{rotate\_half}(q) \cdot \sin(\theta)
```

```math
k' = k \cdot \cos(\theta) + \text{rotate\_half}(k) \cdot \sin(\theta)
```

where `rotate_half` swaps and negates the two halves of each vector:

`[x0, x1, x2, x3]` → `[-x2, -x3, x0, x1]`

RoPE injects 2D spatial position information into each patch, enabling the model to perceive relative positions between tokens.

<details>
<summary>📌 Code reference (attention.py)</summary>

```python
def rope_apply(x, sin, cos):
    return (x * cos) + (rope_rotate_half(x) * sin)
```

</details>

#### 3.4 Computing the Attention Weight Matrix

```math
A = \text{softmax}\!\left(\frac{Q \cdot K^T}{\sqrt{d_k}}\right) = \text{softmax}\!\left(\frac{Q \cdot K^T}{\sqrt{64}}\right) = \text{softmax}\!\left(\frac{Q \cdot K^T}{8}\right)
```

Attention matrix shape: `[1, 6, 1029, 1029]`

> **💡 Why is a hook needed for manual computation?**
>
> The DINOv3 source code uses PyTorch's fused `scaled_dot_product_attention` operator (line 119 in `attention.py`). This operator fuses QK scaling, softmax, and the multiplication with V into a single step and **does not expose the intermediate attention weight matrix**. We therefore intercept the inputs via `register_forward_hook` and recompute the attention matrix step-by-step.

<details>
<summary>📌 Code reference (model.py hook)</summary>

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

### Step 4: Extracting CLS → Patch Attention

From the full 1029 × 1029 attention matrix, extract row **0** (the CLS token) over columns **5–1028** (patch tokens):

```math
A_{\text{cls}} = A[:, :, 0, 5:] \in \mathbb{R}^{1 \times 6 \times 1024}
```

- Index `0` → CLS token (the query)
- Skip the first 5 columns (1 CLS + 4 registers), keeping only the 1024 patch attention values
- **Physical meaning**: each value indicates how strongly the CLS token "attends to" the corresponding image region

---

### Step 5: Reshaping into a 2D Spatial Map

Restore the 1D attention values for 1024 patches back to a 2D spatial layout:

```math
A_{\text{cls}} \xrightarrow{\text{reshape}} [6, 32, 32]
```

Each position `(i, j)` represents the attention intensity for the `(i, j)`-th 16×16 patch in the original image.

---

### Step 6: Upsampling

```math
[6, 32, 32] \xrightarrow{\text{nearest}} [6, 256, 256]
```

Nearest-neighbor interpolation is used to upsample to 256×256 pixels. Nearest-neighbor (rather than bilinear) is chosen to preserve the sharp patch boundaries.

<details>
<summary>📌 Code reference</summary>

```python
attn_up = F.interpolate(
    attn.unsqueeze(0),
    size=(OUTPUT_SIZE, OUTPUT_SIZE),
    mode="nearest",
)[0]  # [heads, 256, 256]
```

</details>

---

### Step 7: Per-Head Min-Max Normalization

Each attention head is independently normalized to [0, 1]:

```math
A_h^{\text{norm}} = \frac{A_h - \min(A_h)}{\max(A_h) - \min(A_h) + \epsilon}
```

The value ranges across different layers and heads vary greatly; normalization ensures each map has full contrast.

<details>
<summary>📌 Code reference</summary>

```python
for h in range(NUM_HEADS):
    a = attn_up[h]
    lo, hi = a.min(), a.max()
    attn_up[h] = (a - lo) / (hi - lo + 1e-8)
```

</details>

---

### Step 8: Front-End Colorization and Blending

The back-end converts the normalized [0, 1] grayscale values to **PNG grayscale images**, which are base64-encoded and sent to the front-end.

The front-end (`app.js`) applies the **Viridis colormap** to map grayscale values to pseudo-colors, then blends the result with the original image at a user-defined opacity:

```math
\text{pixel}_{\text{out}} = \text{original} \times (1 - \alpha) + \text{viridis}(v) \times \alpha
```

**Viridis colormap examples:**

| Grayscale value | Color | RGB |
|:---:|:---|:---|
| 0 | 🟣 Deep purple | `rgb(68, 1, 84)` |
| 0.5 | 🟢 Cyan-green | `rgb(34, 167, 132)` |
| 1 | 🟡 Bright yellow | `rgb(253, 231, 37)` |

---

## Summary Flow Diagram

```text
Input image (any size)
    │
    ▼ Resize(512×512) + Normalize(ImageNet)
[1, 3, 512, 512]
    │
    ▼ PatchEmbed(16×16) + CLS + 4 Registers
[1, 1029, 384]
    │
    ▼ × 12 Transformer Blocks
    │   Each layer's hook intercepts SelfAttention:
    │     QKV = Linear(X)              → [1, 1029, 1152]
    │     Q, K, V = split + reshape    → [1, 6, 1029, 64]
    │     Q, K = RoPE(Q, K)            → rotary position encoding
    │     A = softmax(Q·Kᵀ / 8)       → [1, 6, 1029, 1029]
    │     cls_attn = A[:,:,0,5:]       → [1, 6, 1024]
    │
    ▼ Collect attention from 12 layers
12 × [6, 1024]
    │
    ▼ reshape → [6, 32, 32]
    ▼ nearest ↑ 256×256
    ▼ min-max normalization → [0, 1]
    │
    ▼ → PNG base64 → front-end
    ▼ Viridis colorization + original image blending
    │
    ▼
72 attention visualization heatmaps
(12 layers × 6 heads)
```

---

## Key Parameters

| Parameter | Value | Description |
|:---|:---:|:---|
| Model | ViT-S/16 | Small variant, 16×16 patch size |
| Hidden dimension C | 384 | Standard ViT-S configuration |
| Attention heads | 6 | d_k = 384/6 = 64 |
| Transformer layers | 12 | 6 attention maps per layer |
| Input resolution | 512×512 | Evenly divisible by patch size |
| Patch size | 16×16 | Produces 32×32 = 1024 patches |
| Prefix tokens | 5 | 1 CLS + 4 registers |
| Output map size | 256×256 | Balances quality and transfer size |
| Position encoding | RoPE | Rotary Position Embedding |
| Pre-training data | LVD-142M | Meta large-scale dataset |

---

## How to Interpret the Attention Maps

- 🟡 **Bright regions (yellow/green)**: areas the CLS token attends to strongly — typically salient objects or semantic centers of the image
- 🟣 **Dark regions (purple/blue)**: areas the CLS token largely ignores — typically background
- 📊 **Differences across layers**: shallow layers tend to focus on local textures and edges; deep layers tend to focus on global semantic structure
- 🔀 **Differences across heads**: different heads in the same layer may attend to different semantic patterns (e.g., edges vs. color vs. shape)

![2D Global View](images/preview-2d.png)
