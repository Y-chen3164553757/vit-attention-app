/**
 * 主交互模块 — 上传、API 通信、纹理生成、控件绑定
 */
import { AttentionScene } from './scene.js';

/* ═══════════ Viridis 色图 ═══════════ */
const VIRIDIS_STOPS = [
    [68, 1, 84], [72, 35, 116], [64, 67, 135], [52, 94, 141],
    [41, 120, 142], [32, 144, 140], [34, 167, 132], [68, 190, 112],
    [121, 209, 81], [189, 222, 38], [253, 231, 37],
];

const viridisLUT = new Uint8Array(256 * 3);
{
    const n = VIRIDIS_STOPS.length - 1;
    for (let i = 0; i < 256; i++) {
        const t = (i / 255) * n;
        const lo = Math.floor(t);
        const hi = Math.min(lo + 1, n);
        const f = t - lo;
        for (let c = 0; c < 3; c++) {
            viridisLUT[i * 3 + c] = Math.round(
                VIRIDIS_STOPS[lo][c] * (1 - f) + VIRIDIS_STOPS[hi][c] * f
            );
        }
    }
}

/* ═══════════ 全局状态 ═══════════ */
let scene = null;
let rawAttentions = [];     // { layer, head, img: Image }
let originalImg = null;     // Image
let colorMode = 'heatmap';
let overlayOpacity = 1.0;
let layerFilter = -1;
let headFilter = -1;

/* ═══════════ 工具函数 ═══════════ */

function loadImage(b64) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = 'data:image/png;base64,' + b64;
    });
}

/** 根据当前模式/透明度生成单张纹理 canvas */
function generateTexture(attnImg, origImg, mode, opacity) {
    const S = attnImg.width;
    const cv = document.createElement('canvas');
    cv.width = S; cv.height = S;
    const ctx = cv.getContext('2d');

    if (mode === 'heatmap' && origImg) {
        ctx.drawImage(origImg, 0, 0, S, S);
        const origData = ctx.getImageData(0, 0, S, S);
        const px = origData.data;

        const attnCv = document.createElement('canvas');
        attnCv.width = S; attnCv.height = S;
        const attnCtx = attnCv.getContext('2d');
        attnCtx.drawImage(attnImg, 0, 0, S, S);
        const attnPx = attnCtx.getImageData(0, 0, S, S).data;

        const a = opacity;
        for (let i = 0, len = S * S; i < len; i++) {
            const v = attnPx[i * 4];
            const ri = v * 3;
            px[i * 4]     = Math.round(px[i * 4]     * (1 - a) + viridisLUT[ri]     * a);
            px[i * 4 + 1] = Math.round(px[i * 4 + 1] * (1 - a) + viridisLUT[ri + 1] * a);
            px[i * 4 + 2] = Math.round(px[i * 4 + 2] * (1 - a) + viridisLUT[ri + 2] * a);
        }
        ctx.putImageData(origData, 0, 0);
    } else {
        ctx.drawImage(attnImg, 0, 0, S, S);
    }
    return cv;
}

/** 生成全部 72 张纹理并交给场景 */
function buildPlanes() {
    if (!rawAttentions.length || !scene) return;
    const items = rawAttentions.map(a => ({
        layer: a.layer,
        head: a.head,
        canvas: generateTexture(a.img, originalImg, colorMode, overlayOpacity),
    }));
    scene.setPlanes(items);
    scene.setFilter(layerFilter, headFilter);
}

/** 仅更新纹理 (模式/透明度切换时) */
function refreshTextures() {
    if (!rawAttentions.length || !scene || !scene.planes.length) return;
    const cvs = rawAttentions.map(a =>
        generateTexture(a.img, originalImg, colorMode, overlayOpacity)
    );
    scene.updateTextures(cvs);
}

/* ═══════════ 上传与 API ═══════════ */

async function handleFile(file) {
    if (!file || !file.type.startsWith('image/')) return;

    // 加载中
    const overlay = document.getElementById('loadingOverlay');
    const hint = document.getElementById('sceneHint');
    if (overlay) overlay.hidden = false;
    if (hint) hint.hidden = true;

    const form = new FormData();
    form.append('file', file);

    try {
        const res = await fetch('/api/analyze', { method: 'POST', body: form });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        // 加载所有图片
        const tasks = data.attentions.map(async (a) => ({
            layer: a.layer,
            head: a.head,
            img: await loadImage(a.image),
        }));
        rawAttentions = await Promise.all(tasks);
        originalImg = await loadImage(data.original_image);

        // 场景内放置原图
        scene.setImage(originalImg);

        buildPlanes();
    } catch (err) {
        console.error('Analysis failed:', err);
        alert('分析失败: ' + err.message);
    } finally {
        const ov = document.getElementById('loadingOverlay');
        if (ov) ov.hidden = true;
    }
}

/* ═══════════ 初始化 ═══════════ */

document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('canvas3d');
    scene = new AttentionScene(canvas);

    // ── 悬浮信息 ──
    const infoBadge = document.getElementById('infoBadge');
    const infoText  = document.getElementById('infoText');
    scene.onHover = (layer, head) => {
        if (layer >= 0) {
            infoBadge.hidden = false;
            infoText.textContent = `Layer ${layer + 1}  Head ${head + 1}`;
        } else {
            infoBadge.hidden = true;
        }
    };

    // ── 文件上传 ──
    const zone  = document.getElementById('uploadZone');
    const input = document.getElementById('fileInput');

    zone.addEventListener('click', () => input.click());
    input.addEventListener('change', (e) => {
        if (e.target.files?.[0]) handleFile(e.target.files[0]);
    });

    // 拖放
    zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('drag-over'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('drag-over');
        if (e.dataTransfer.files?.[0]) handleFile(e.dataTransfer.files[0]);
    });

    // ── 配色切换 ──
    document.querySelectorAll('.toggle-btn[data-mode]').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.toggle-btn[data-mode]').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            colorMode = btn.dataset.mode;
            refreshTextures();
        });
    });

    // ── 2D / 3D 切换 ──
    document.querySelectorAll('.toggle-btn[data-view]').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.toggle-btn[data-view]').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            scene.setMode(btn.dataset.view === '2d');
        });
    });

    // ── Layer 滑块 ──
    const layerSlider = document.getElementById('layerSlider');
    const layerVal   = document.getElementById('layerValue');
    layerSlider.addEventListener('input', () => {
        layerFilter = parseInt(layerSlider.value);
        layerVal.textContent = layerFilter < 0 ? 'All' : layerFilter + 1;
        scene.setFilter(layerFilter, headFilter);
    });

    // ── Head 滑块 ──
    const headSlider = document.getElementById('headSlider');
    const headVal   = document.getElementById('headValue');
    headSlider.addEventListener('input', () => {
        headFilter = parseInt(headSlider.value);
        headVal.textContent = headFilter < 0 ? 'All' : headFilter + 1;
        scene.setFilter(layerFilter, headFilter);
    });

    // ── 透明度滑块 ──
    const opSlider = document.getElementById('opacitySlider');
    const opVal    = document.getElementById('opacityValue');
    let opTimer = null;
    opSlider.addEventListener('input', () => {
        overlayOpacity = parseInt(opSlider.value) / 100;
        opVal.textContent = overlayOpacity.toFixed(1);
        clearTimeout(opTimer);
        opTimer = setTimeout(() => {
            if (colorMode === 'heatmap') refreshTextures();
        }, 80);
    });

    // ── 加载动画 ──
    const dots = document.querySelector('.dots');
    if (dots) {
        let c = 1;
        setInterval(() => { c = (c % 3) + 1; dots.textContent = '.'.repeat(c); }, 500);
    }

    // ── 自动加载默认图片 ──
    (async () => {
        try {
            const res = await fetch('/api/default-image');
            if (!res.ok) return;
            const blob = await res.blob();
            const file = new File([blob], 'default.jpg', { type: blob.type });
            await handleFile(file);
        } catch (e) {
            console.warn('Default image load skipped:', e);
        }
    })();
});
