"""
FastAPI 应用: DINOv3 ViT Attention Visualizer
"""

import base64
import io
import logging
import mimetypes
from contextlib import asynccontextmanager
from pathlib import Path

# Fix Windows registry MIME type issues for JS/CSS module scripts
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
from starlette.requests import Request
from fastapi.responses import FileResponse

from model import get_extractor, OUTPUT_SIZE

DEFAULT_IMAGE = Path(__file__).resolve().parent / "default.jpg"

# 日志格式：时间 | 级别 | 模块 | 行号 | 消息
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_file_handler = logging.FileHandler("app.log", encoding="utf-8", mode="a")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))

_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))

logging.basicConfig(level=logging.DEBUG, handlers=[_file_handler, _console_handler])
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent

# Windows 上 .js 可能被系统注册为 text/plain，导致 module script 加载失败。
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("application/wasm", ".wasm")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时预加载模型"""
    logger.info("Loading DINOv3 ViT-S/16 model...")
    get_extractor()
    logger.info("Model loaded successfully.")
    yield


import time
from starlette.middleware.base import BaseHTTPMiddleware

STATIC_VERSION = str(int(time.time()))

app = FastAPI(title="DINOv3 Attention Visualizer", lifespan=lifespan)


class NoCacheMiddleware(BaseHTTPMiddleware):
    """对所有响应强制禁用浏览器缓存"""
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response


from starlette.staticfiles import StaticFiles as _StaticFiles

_NO_CACHE_HEADERS = [
    (b"cache-control", b"no-store, no-cache, must-revalidate"),
    (b"pragma", b"no-cache"),
    (b"expires", b"0"),
]
_NO_CACHE_KEYS = {b"cache-control", b"pragma", b"expires"}


class NoCacheStaticFiles(_StaticFiles):
    """覆盖 StaticFiles，对每个静态资源响应注入禁缓存头"""
    async def __call__(self, scope, receive, send):
        async def send_with_no_cache(message):
            if message["type"] == "http.response.start":
                # 过滤掉原有缓存相关头，再注入禁缓存头
                filtered = [
                    (k, v) for k, v in message.get("headers", [])
                    if k.lower() not in _NO_CACHE_KEYS
                ]
                message["headers"] = filtered + _NO_CACHE_HEADERS
            await send(message)
        await super().__call__(scope, receive, send_with_no_cache)


app.add_middleware(NoCacheMiddleware)
app.mount("/static", NoCacheStaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    response = templates.TemplateResponse(
        request=request, name="index.html",
        context={"v": STATIC_VERSION}
    )
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    return response


@app.get("/api/default-image")
async def default_image():
    """返回默认预览图片"""
    if DEFAULT_IMAGE.exists():
        return FileResponse(
            str(DEFAULT_IMAGE),
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
    return {"error": "default.jpg not found"}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    """接收上传图片，返回注意力图数据 (base64 PNG)"""
    logger.info("analyze request: filename=%s content_type=%s", file.filename, file.content_type)
    try:
        contents = await file.read()
        logger.debug("read %d bytes from upload", len(contents))
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        logger.debug("image size=%s", image.size)
    except Exception:
        logger.exception("Failed to read/decode uploaded image")
        raise

    try:
        extractor = get_extractor()
        result = extractor.extract(image)
    except Exception:
        logger.exception("Model inference failed")
        raise

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

    logger.info("analyze done: layers=%d", len(result["attentions"]))
    return {
        "attentions": attention_data,
        "original_image": original_b64,
        "model_info": result["model_info"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8080,
        reload=True,
    )
