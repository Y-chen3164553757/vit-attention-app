# ViT Attention Visualizer

> 🌐 Language / 语言：[中文](README.md) | **English**

A self-attention (Self-Attention) visualization web application built on FastAPI and DINOv3 (ViT-S/16). It allows you to intuitively observe how each layer and each Attention Head of a Vision Transformer focuses on different regions of an image.

## 🔄 Acknowledgments & Refactoring Notes

This project is a refactored and optimized version of the Hugging Face Space [webml-community/attention-visualization](https://huggingface.co/spaces/webml-community/attention-visualization).
The original project is open-sourced under the **MIT License**. The powerful visual feature extraction capability underlying this project is driven by [DINOv3](https://github.com/facebookresearch/dinov3), open-sourced by the Meta AI team.

We made the following significant modifications to the model and backend architecture compared to the original:

- **Model Upgrade**: Replaced the original visualization model with **DINOv3 (ViT-S/16)**, which offers stronger feature extraction capabilities.
- **Architecture Refactor**: Introduced **FastAPI** as the backend, improving the speed and stability of the API.
- **Loading Optimization**: Implemented a **fully local weight loading** mechanism, eliminating uncontrollable online downloads at runtime and Torch Hub cache conflicts.
- **Code Cleanup**: Removed a large amount of unused redundant code from the original model and re-encapsulated a clean Attention extraction Hook logic.

## 🌟 Features

![3D Preview](images/preview-3d.png)
*Layer-by-layer, head-by-head Attention heatmap stacking in 3D interactive view.*

![2D Preview](images/preview-2d.png)
*Global Attention Head display matrix in 2D flat view.*

- **Lightweight Frontend Interaction**: Upload images via an intuitive Web UI with support for click and drag-and-drop operations.
- **Per-Layer / Per-Head Attention Visualization**: Automatically extracts all Head attention heatmaps from the ViT and renders them in the frontend.
- **Fully Local**: Manually extracts model Attention via Forward Hooks, with support for **fully local model weight loading** — no network download required.
- **Powered by DINOv3**: Uses advanced DINOv3 ViT-S/16 pretrained weights for high-quality features and attention maps.

## 📂 Project Structure

```text
vit-attention-app/
├── app.py                     # FastAPI backend entry point and route definitions
├── model.py                   # Model loading and Attention extraction module (contains core hook logic)
├── requirements.txt           # Project dependencies
├── ATTENTION_EXTRACTION.md    # Detailed explanation of attention extraction principles
├── dinov3/                    # Bundled local DINOv3 library (modified from official, redundant code removed)
├── static/                    # Static assets (CSS, JS)
├── templates/                 # HTML templates
└── weights/                   # Local model weights directory
```

## ⚙️ Installation

Python >= 3.11 is required. A virtual environment such as `venv` or `conda` is recommended.

```bash
# Clone the repository
git clone https://github.com/Y-chen3164553757/vit-attention-app.git
cd vit-attention-app

# Install dependencies
pip install -r requirements.txt
```

## ⬇️ Model Weights

The project uses **local weight loading** by default. You need to place the pretrained model under the `weights/` directory.

The default file path is:
`weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth`

If you don't have the weight file after cloning, download the corresponding ViT-S/16 (LVD1689M) model weights from the official DINOv3 release page and save them to the `weights/` directory.

## 🚀 Start the Server

Once dependencies are installed and weights are in place, start the web application with uvicorn:

```bash
python app.py
```

Or run directly:

```bash
uvicorn app:app --host 127.0.0.1 --port 8080 --reload
```

After starting, open your browser and visit [http://127.0.0.1:8080](http://127.0.0.1:8080). Upload any image to start the self-attention visualization.

## 📄 License

The refactored frontend interaction and backend API architecture code of this project is **All Rights Reserved**. Unauthorized direct use for commercial purposes or redistribution is prohibited. For commercial cooperation or software copyright registration reference, please contact the author.

Parts of this project inherit or are modified from open-source community code and are used in compliance with their original licenses:
- Original inspiration and partial visualization reference from Hugging Face Space [webml-community/attention-visualization](https://huggingface.co/spaces/webml-community/attention-visualization) (under MIT License).
- The bundled DINOv3 code and weights follow the Meta [DINOv3 License Agreement](https://github.com/facebookresearch/dinov3/blob/main/LICENSE).

## 📝 Technical Details

The core logic for Attention extraction: by registering `register_forward_hook` on each layer's SelfAttention to intercept inputs, then manually computing using `Q` and `K`:

$$
\text{Attention} = \text{softmax}\left(\frac{Q \cdot K^T}{\sqrt{d}}\right)
$$

This extracts the attention weights of the CLS token over each Patch, which are then upsampled into visualization heatmaps.

For more detailed theory and mathematical derivations, refer to [ATTENTION_EXTRACTION.md](ATTENTION_EXTRACTION.md).
