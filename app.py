"""
FastAPI 应用: DINOv3 ViT Attention Visualizer
"""

import base64
import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from starlette.requests import Request
from fastapi.responses import FileResponse

from model import get_extractor, OUTPUT_SIZE

DEFAULT_IMAGE = Path(__file__).resolve().parent / "default.jpg"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时预加载模型"""
    logger.info("Loading DINOv3 ViT-S/16 model...")
    get_extractor()
    logger.info("Model loaded successfully.")
    yield


app = FastAPI(title="DINOv3 Attention Visualizer", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/api/default-image")
async def default_image():
    """返回默认预览图片"""
    if DEFAULT_IMAGE.exists():
        return FileResponse(str(DEFAULT_IMAGE), media_type="image/jpeg")
    return {"error": "default.jpg not found"}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    """接收上传图片，返回注意力图数据 (base64 PNG)"""
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    extractor = get_extractor()
    result = extractor.extract(image)

    img_size = OUTPUT_SIZE

    # 将注意力 numpy 数组转为 base64 PNG
    attention_data = []
    for layer_idx, layer_attn in enumerate(result["attentions"]):
        for head_idx in range(layer_attn.shape[0]):
            attn_map = layer_attn[head_idx]  # [H, W] float32, 0~1
            img_array = (attn_map * 255).astype(np.uint8)
            img = Image.fromarray(img_array, mode="L")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            attention_data.append({
                "layer": layer_idx,
                "head": head_idx,
                "image": b64,
            })

    # 返回与输出尺寸一致的缩放版本
    thumb = image.resize((img_size, img_size), Image.LANCZOS)
    buf = io.BytesIO()
    thumb.save(buf, format="PNG")
    original_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return {
        "attentions": attention_data,
        "original_image": original_b64,
        "model_info": result["model_info"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8080, reload=True)
