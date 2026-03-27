"""
DINOv3 ViT-S/16 注意力提取模块

直接从本地 .pth 文件加载权重，通过 register_forward_hook 在每层
SelfAttention 上拦截 QKV 并手动计算 softmax(q @ k^T / sqrt(d))。
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.transforms import v2

# ── 路径 ──
_APP_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = str(_APP_DIR / "weights" / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth")

# dinov3 包现在直接在 vit-attention-app/dinov3/ 下，作为本地子包导入
# 确保 app 根目录在 sys.path 中以支持 `import dinov3`
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

# ── 模型常量 ──
NUM_LAYERS = 12
NUM_HEADS = 6
PATCH_SIZE = 16
IMAGE_SIZE = 512  # 16×32 = 512，与 patch_size=16 整除对齐
N_PREFIX_TOKENS = 5  # 1 CLS + 4 registers
OUTPUT_SIZE = 256    # 输出图片尺寸（API 传输用，平衡质量与大小）

# ── 官方推荐的图像预处理 (LVD-1689M 权重) ──
transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


class AttentionExtractor:
    """通过 forward hook 从 DINOv3 ViT 中提取每层 CLS→patch 注意力权重。"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.attention_maps: list[torch.Tensor] = []
        self._hooks: list[torch.utils.hooks.RemovableHook] = []

    def load_model(self):
        """加载 DINOv3 ViT-S/16，并显式使用本地权重文件。"""
        from dinov3.hub.backbones import dinov3_vits16

        weights_path = Path(WEIGHTS_PATH)
        if not weights_path.exists():
            raise FileNotFoundError(f"Local weights not found: {weights_path}")

        self.model = dinov3_vits16(pretrained=False)
        state_dict = torch.load(weights_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval().to(self.device)
        self._register_hooks()
        return self

    # ------------------------------------------------------------------ hooks
    def _register_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        for block in self.model.blocks:
            self._hooks.append(
                block.attn.register_forward_hook(self._attention_hook, with_kwargs=True)
            )

    def _attention_hook(self, module, input_args, kwargs, output):
        """拦截 SelfAttention.forward，复现 compute_attention 中的 QKV 变换，
        手动计算 softmax(q·k^T / √d) 以获取注意力权重。"""
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

        # CLS (idx 0) 对 patch tokens 的注意力
        cls_attn = attn_weights[:, :, 0, N_PREFIX_TOKENS:]  # [B, heads, hw]
        self.attention_maps.append(cls_attn.detach().cpu())

    # ------------------------------------------------------------------ extract
    @torch.inference_mode()
    def extract(self, image: Image.Image) -> dict:
        """提取每层每头的 CLS→patch 注意力图。

        Returns
        -------
        dict  attentions : list of [heads, H_up, W_up] numpy arrays
              model_info : dict
              image_size : [w, h]
              grid_size  : [gh, gw]
        """
        self.attention_maps.clear()

        img_tensor = transform(image).unsqueeze(0).to(self.device)
        _, _, H_img, W_img = img_tensor.shape
        grid_h, grid_w = H_img // PATCH_SIZE, W_img // PATCH_SIZE

        # 推理 — hooks 收集注意力
        self.model.forward_features(img_tensor)

        attentions = []
        for layer_attn in self.attention_maps:
            attn = layer_attn[0]  # [heads, hw]
            attn = attn.reshape(NUM_HEADS, grid_h, grid_w)

            # nearest 上采样至输出尺寸（与官方参考 worker.js 一致）
            attn_up = F.interpolate(
                attn.unsqueeze(0),
                size=(OUTPUT_SIZE, OUTPUT_SIZE),
                mode="nearest",
            )[0]  # [heads, OUTPUT_SIZE, OUTPUT_SIZE]

            # 逐 head min-max 归一化到 [0, 1]（官方参考做法）
            for h in range(NUM_HEADS):
                a = attn_up[h]
                lo, hi = a.min(), a.max()
                attn_up[h] = (a - lo) / (hi - lo + 1e-8)

            attentions.append(attn_up.numpy())

        return {
            "attentions": attentions,
            "model_info": {"layers": NUM_LAYERS, "heads": NUM_HEADS, "patch_size": PATCH_SIZE},
            "image_size": [image.width, image.height],
            "grid_size": [grid_h, grid_w],
        }


# ── 全局单例 ──
_extractor: AttentionExtractor | None = None


def get_extractor() -> AttentionExtractor:
    global _extractor
    if _extractor is None:
        _extractor = AttentionExtractor()
        _extractor.load_model()
    return _extractor
