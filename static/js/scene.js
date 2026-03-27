/**
 * Three.js 3D/2D 场景管理 — DINOv3 注意力可视化
 * 改编自 attention-visualization/src/App.jsx 的布局逻辑
 * 使用弹簧阻尼相机 + lerp 平滑，仿照原项目 @react-spring/three 行为
 */
import * as THREE from 'three';

// ── 布局常量 (与原 React 版本一致) ──
const HEAD_HEIGHT = 2.4;
const HEAD_WIDTH  = 2.4;
const X_SPACING   = 0.4;
const Z_SPACING   = 2.0;
const LAYER_SPACING = 0.25;
const Y_BASE   = 0.5 * HEAD_HEIGHT - 1;   // 0.2
const Y_HOVER  = HEAD_HEIGHT - 0.25;       // 2.15
const NUM_HEADS  = 6;
const NUM_LAYERS = 12;

// ── 场景图片 ──
const IMAGE_HEIGHT  = 4;
const MAX_IMAGE_WIDTH = 8;
const IMAGE_PADDING = 1.5;

// ── 相机 ──
const CAM_ANGLE = (Math.PI * 5) / 12;
const CAM_DIST  = 16;
const DEFAULT_CAM = new THREE.Vector3(
    -CAM_DIST * Math.cos(CAM_ANGLE), 3.5, CAM_DIST * Math.sin(CAM_ANGLE)
);
const ZOOM_DIST = 3.5;

// ── 弹簧参数 (模拟 @react-spring tension/friction) ──
const SPRING_TENSION  = 500;
const SPRING_FRICTION = 20;

// ── 悬浮弹簧 (hover Y) ──
const HOVER_TENSION  = 280;
const HOVER_FRICTION = 60;

// ── 边缘滚动 ──
const EDGE_ZONE  = 0.5;
const SCROLL_SPEED = 12;

// ── 射线防抖 ──
const HOVER_DEBOUNCE    = 0.04;  // 40ms — 基础防抖 (更灵敏)
const HOVER_LEAVE_MULT  = 2.0;   // 离开当前目标(→空)需要 2 倍防抖时间
const HOVER_SWITCH_MULT = 1.5;   // 从一个目标切换到另一个目标需要 1.5 倍防抖时间
const MOUSE_VEL_GATE    = 3.0;   // 鼠标速度超过此值时不更换悬浮目标 (NDC/s)

// ── 2D 悬浮上升量 ──
const Y_HOVER_2D = 0.5;

// ── 相机平滑速率 (帧率无关) ──
const CAM_SMOOTH_RATE = 6;     // 越大越快跟随

// ── 2D 网格布局 ──
const GRID_CELL     = 2.6;     // 格子尺寸
const GRID_GAP      = 0.4;    // 间距
const GROUP_Y       = 1;      // planeGroup.position.y 偏移

/* ========== 简易弹簧 (子步积分提升稳定性) ========== */
const SPRING_SUBSTEPS = 4;

class Spring3 {
    constructor(tension, friction) {
        this.tension  = tension;
        this.friction = friction;
        this.target = new THREE.Vector3();
        this.value  = new THREE.Vector3();
        this.vel    = new THREE.Vector3();
    }
    setTarget(x, y, z) { this.target.set(x, y, z); }
    snap(x, y, z)      { this.value.set(x, y, z); this.target.set(x, y, z); this.vel.set(0,0,0); }
    step(dt) {
        const sub = dt / SPRING_SUBSTEPS;
        const t = this.tension, f = this.friction;
        for (let i = 0; i < SPRING_SUBSTEPS; i++) {
            const dx = this.target.x - this.value.x;
            const dy = this.target.y - this.value.y;
            const dz = this.target.z - this.value.z;
            this.vel.x += (dx * t - this.vel.x * f) * sub;
            this.vel.y += (dy * t - this.vel.y * f) * sub;
            this.vel.z += (dz * t - this.vel.z * f) * sub;
            this.value.x += this.vel.x * sub;
            this.value.y += this.vel.y * sub;
            this.value.z += this.vel.z * sub;
        }
    }
}

class Spring1 {
    constructor(tension, friction) { this.tension = tension; this.friction = friction; this.target = 0; this.value = 0; this.vel = 0; }
    snap(v) { this.value = v; this.target = v; this.vel = 0; }
    step(dt) {
        const sub = dt / SPRING_SUBSTEPS;
        for (let i = 0; i < SPRING_SUBSTEPS; i++) {
            const d = this.target - this.value;
            this.vel += (d * this.tension - this.vel * this.friction) * sub;
            this.value += this.vel * sub;
        }
    }
}

/**
 * 计算某一 (layer, head) 对应的 3D 坐标
 */
function computePosition(layer, head) {
    const depthOff = (NUM_HEADS - 1) * X_SPACING;
    const xOff = HEAD_WIDTH / 2 + depthOff + layer * (HEAD_WIDTH + depthOff + LAYER_SPACING);
    return [
        xOff - head * X_SPACING,
        Y_BASE,
        ((NUM_HEADS + 1) / 2 - head - 1) * Z_SPACING,
    ];
}

/**
 * 计算 2D 网格中 (layer, head) 对应的坐标
 * 列=layer (X 轴，左→右)，行=head (Z 轴，head 0 在上方)
 */
function computePosition2D(layer, head) {
    const x = layer * (GRID_CELL + GRID_GAP);
    const z = -(NUM_HEADS - 1 - head) * (GRID_CELL + GRID_GAP);
    return [x, Y_BASE, z];
}

export class AttentionScene {
    constructor(canvas) {
        this.canvas = canvas;
        this.planes = [];         // { mesh, layer, head, basePos, basePos3D, basePos2D, texture, hoverSpring, posSpring }
        this.hoveredIdx = -1;
        this.activeIdx  = -1;
        this.sceneCenter = 0;
        this.sceneBoundsMin = 0;
        this.sceneBoundsMax = 0;
        this.windowMouseX = 0;
        this.mouseActive = true;
        this.clock = new THREE.Clock();

        // 射线防抖
        this._pendingHover = -1;
        this._hoverTimer   = 0;

        // 鼠标速度追踪
        this._prevMouseNDC  = new THREE.Vector2(-999, -999);
        this._mouseVelocity = 0;

        // 2D/3D 模式
        this.is2D = false;

        // 2D 轴标签
        this._2dLabels = [];

        // 2D 对齐网格
        this._2dGrid = null;

        // 回调
        this.onHover = null;
        this.onClick = null;

        // ── 场景 ──
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x040b1b);

        // ── 相机 ──
        const aspect = (canvas.clientWidth || 1) / (canvas.clientHeight || 1);
        this.camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 500);
        this.camera.position.copy(DEFAULT_CAM);

        // 弹簧驱动相机
        this.camSpring    = new Spring3(SPRING_TENSION, SPRING_FRICTION);
        this.lookAtSpring = new Spring3(SPRING_TENSION, SPRING_FRICTION);
        this.camSpring.snap(DEFAULT_CAM.x, DEFAULT_CAM.y, DEFAULT_CAM.z);
        this.lookAtSpring.snap(0, 0, 0);
        this.currentCamPos  = DEFAULT_CAM.clone();
        this.currentLookAt  = new THREE.Vector3();

        // ── 渲染器 ──
        this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.toneMapping = THREE.NoToneMapping;
        this.renderer.outputColorSpace = THREE.SRGBColorSpace;
        this._resize();

        // ── 网格 ──
        this.gridHelper = new THREE.GridHelper(200, 200, 0xffffff, 0x444444);
        this.gridHelper.position.y = -1;
        this.scene.add(this.gridHelper);

        // ── 灯光 ──
        this.scene.add(new THREE.AmbientLight(0xffffff, 1));

        // ── 平面容器 ──
        this.planeGroup = new THREE.Group();
        this.planeGroup.position.y = 1;
        this.scene.add(this.planeGroup);

        // ── 场景内图片 ──
        this.imageMesh = null;

        // ── 射线 ──
        this.raycaster = new THREE.Raycaster();
        this.mouseNDC  = new THREE.Vector2(-999, -999);

        this._bind();
        this._loop();
    }

    /* ========== 事件绑定 ========== */
    _bind() {
        this.canvas.addEventListener('pointermove', (e) => {
            const r = this.canvas.getBoundingClientRect();
            this.mouseNDC.x =  ((e.clientX - r.left) / r.width)  * 2 - 1;
            this.mouseNDC.y = -((e.clientY - r.top)  / r.height) * 2 + 1;
        });
        this.canvas.addEventListener('click', () => this._click());
        document.documentElement.addEventListener('mousemove', (e) => {
            this.windowMouseX = (e.clientX / window.innerWidth - 0.5) * 2;
        });
        document.documentElement.addEventListener('mouseleave', () => { this.mouseActive = false; });
        document.documentElement.addEventListener('mouseenter', () => { this.mouseActive = true; });
        window.addEventListener('resize', () => this._resize());
    }

    _resize() {
        const w = this.canvas.clientWidth || 1;
        const h = this.canvas.clientHeight || 1;
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(w, h, false);
    }

    /* ========== 场景内图片 ========== */

    /**
     * 在场景左侧放置原始图片 (仿照原项目 SceneImage)
     * @param {HTMLImageElement} imgElement
     */
    setImage(imgElement) {
        // 移除旧图片
        if (this.imageMesh) {
            this.imageMesh.geometry.dispose();
            this.imageMesh.material.map?.dispose();
            this.imageMesh.material.dispose();
            this.scene.remove(this.imageMesh);
            this.imageMesh = null;
        }
        if (!imgElement) return;

        const ar = imgElement.naturalWidth / imgElement.naturalHeight;
        let w = ar * IMAGE_HEIGHT;
        let h = IMAGE_HEIGHT;
        if (w > MAX_IMAGE_WIDTH) { w = MAX_IMAGE_WIDTH; h = w / ar; }

        const tex = new THREE.Texture(imgElement);
        tex.colorSpace = THREE.SRGBColorSpace;
        tex.needsUpdate = true;
        tex.minFilter = THREE.LinearFilter;
        tex.magFilter = THREE.LinearFilter;

        const geo = new THREE.PlaneGeometry(w, h);
        const mat = new THREE.MeshBasicMaterial({ map: tex });
        this.imageMesh = new THREE.Mesh(geo, mat);
        this.imageMesh.position.set(-w / 2 - IMAGE_PADDING, h / 2, 0);
        this.scene.add(this.imageMesh);

        // 更新滚动下限
        this.sceneBoundsMin = -w - IMAGE_PADDING;
    }

    /* ========== 平面管理 ========== */

    setPlanes(items) {
        this.clearPlanes();
        let maxX = 0;

        for (const item of items) {
            const pos3D = computePosition(item.layer, item.head);
            const pos2D = computePosition2D(item.layer, item.head);
            const pos = this.is2D ? pos2D : pos3D;

            const texture = new THREE.CanvasTexture(item.canvas);
            texture.colorSpace = THREE.SRGBColorSpace;
            texture.minFilter = THREE.LinearFilter;
            texture.magFilter = THREE.LinearFilter;

            const geo = new THREE.PlaneGeometry(HEAD_WIDTH, HEAD_HEIGHT);
            const mat = new THREE.MeshBasicMaterial({ map: texture, side: THREE.DoubleSide });
            const mesh = new THREE.Mesh(geo, mat);
            mesh.position.set(pos[0], pos[1], pos[2]);

            // ── 初始旋转: 2D 时平放 (-π/2)，3D 时竖直 (0) ──
            const initRot = this.is2D ? -Math.PI / 2 : 0;
            mesh.rotation.x = initRot;

            // 标签精灵
            const sprite = this._label(`L${item.layer + 1} H${item.head + 1}`);
            sprite.position.set(-HEAD_WIDTH / 2, HEAD_HEIGHT / 2 + 0.3, 0.01);
            mesh.add(sprite);

            // 悬浮弹簧
            const hoverSpring = new Spring1(HOVER_TENSION, HOVER_FRICTION);
            hoverSpring.value = pos[1];
            hoverSpring.target = pos[1];

            // 位置弹簧 — 用于 2D↔3D 过渡
            const posSpring = new Spring3(180, 22);
            posSpring.snap(pos[0], pos[1], pos[2]);

            // ── 旋转弹簧 — 用于 2D↔3D 平面翻转动画 ──
            const rotSpring = new Spring1(180, 22);
            rotSpring.snap(initRot);

            this.planeGroup.add(mesh);
            this.planes.push({
                mesh, layer: item.layer, head: item.head,
                basePos: [...pos], basePos3D: [...pos3D], basePos2D: [...pos2D],
                texture, hoverSpring, posSpring, rotSpring,
            });
            maxX = Math.max(maxX, pos[0] + HEAD_WIDTH / 2);
        }

        this.sceneBoundsMax = maxX;
        this.sceneCenter = maxX / 2;
        this.activeIdx  = -1;
        this.hoveredIdx = -1;

        this._build2DLabels();
        this._build2DGrid();
        this._update2DLabelVisibility();
        if (this._2dGrid) this._2dGrid.visible = this.is2D;
    }

    /* ========== 2D / 3D 切换 ========== */

    setMode(is2D) {
        if (this.is2D === is2D) return;
        this.is2D = is2D;
        this.activeIdx = -1;
        this.hoveredIdx = -1;

        for (const p of this.planes) {
            const target = is2D ? p.basePos2D : p.basePos3D;
            p.basePos = [...target];
            p.posSpring.setTarget(target[0], target[1], target[2]);
            p.hoverSpring.target = target[1];

            // ── 旋转目标: 2D 平放 / 3D 竖直 ──
            p.rotSpring.target = is2D ? -Math.PI / 2 : 0;
        }

        // 原始图片 & 3D 地面网格在 2D 中隐藏
        if (this.imageMesh) this.imageMesh.visible = !is2D;
        this.gridHelper.visible = !is2D;
        if (this._2dGrid) this._2dGrid.visible = is2D;

        // 重算场景边界
        let maxX = 0;
        for (const p of this.planes) {
            maxX = Math.max(maxX, p.basePos[0] + HEAD_WIDTH / 2);
        }
        this.sceneBoundsMax = maxX;
        this.sceneCenter = Math.min(this.sceneCenter, maxX);

        // 更新 2D 标签可见性
        this._update2DLabelVisibility();
    }

    /* ========== 2D 轴标签 ========== */

    _build2DLabels() {
        for (const l of this._2dLabels) {
            this.planeGroup.remove(l);
            l.material.map?.dispose();
            l.material.dispose();
        }
        this._2dLabels = [];

        // 列标签 (Layer 1–12) — 放在 head 0 上方整整一行的位置
        // head 0 中心 z = -(NUM_HEADS-1)*(GRID_CELL+GRID_GAP), 再往 -Z 移一整行
        const labelRowZ = -(NUM_HEADS) * (GRID_CELL + GRID_GAP);
        for (let layer = 0; layer < NUM_LAYERS; layer++) {
            const [x] = computePosition2D(layer, 0);
            const sprite = this._label(`Layer ${layer + 1}`, 24, '#74b9ff');
            sprite.position.set(x, 0.2, labelRowZ);
            sprite.scale.set(2.2, 0.55, 1);
            this.planeGroup.add(sprite);
            this._2dLabels.push(sprite);
        }

        // 行标签 (Head 1–6) — 在网格左侧
        const headLabelX = -GRID_CELL / 2 - 2.0;
        for (let head = 0; head < NUM_HEADS; head++) {
            const z = computePosition2D(0, head)[2];
            const sprite = this._label(`Head ${head + 1}`, 24, '#ffeaa7');
            sprite.position.set(headLabelX, 0.2, z);
            sprite.scale.set(2.0, 0.5, 1);
            this.planeGroup.add(sprite);
            this._2dLabels.push(sprite);
        }
    }

    _update2DLabelVisibility() {
        for (const l of this._2dLabels) l.visible = this.is2D;
    }

    /* ========== 2D 对齐网格 ========== */

    _build2DGrid() {
        if (this._2dGrid) {
            this.planeGroup.remove(this._2dGrid);
            this._2dGrid.geometry.dispose();
            this._2dGrid.material.dispose();
            this._2dGrid = null;
        }

        const step = GRID_CELL + GRID_GAP;   // 3.0 — 与 2D 布局步长一致
        const halfStep = step / 2;            // 1.5 — 线条穿过格间间隙中心
        const halfSize = 100;                 // 网格半径 (覆盖 200×200 区域)

        // 网格线位于 n*step + halfStep，正好穿过每个格间间隙的中心
        const count = Math.ceil(halfSize / step);
        const points = [];

        for (let i = -count; i <= count; i++) {
            const pos = i * step + halfStep;
            // 竖线 (固定 X)
            points.push(new THREE.Vector3(pos, 0, -halfSize));
            points.push(new THREE.Vector3(pos, 0,  halfSize));
            // 横线 (固定 Z)
            points.push(new THREE.Vector3(-halfSize, 0, pos));
            points.push(new THREE.Vector3( halfSize, 0, pos));
        }

        const geo = new THREE.BufferGeometry().setFromPoints(points);
        const mat = new THREE.LineBasicMaterial({ color: 0x444444 });
        this._2dGrid = new THREE.LineSegments(geo, mat);
        this._2dGrid.position.y = Y_BASE - 0.05;  // 略低于平面，确保可见
        this._2dGrid.visible = false;
        this.planeGroup.add(this._2dGrid);
    }

    updateTextures(canvases) {
        for (let i = 0; i < this.planes.length && i < canvases.length; i++) {
            this.planes[i].texture.image = canvases[i];
            this.planes[i].texture.needsUpdate = true;
        }
    }

    setFilter(layerF, headF) {
        for (const p of this.planes) {
            p.mesh.visible = (layerF < 0 || p.layer === layerF) && (headF < 0 || p.head === headF);
        }
    }

    clearPlanes() {
        for (const p of this.planes) {
            p.mesh.geometry.dispose();
            p.mesh.material.dispose();
            p.texture.dispose();
            this.planeGroup.remove(p.mesh);
        }
        this.planes = [];
        this.hoveredIdx = -1;
        this.activeIdx  = -1;
    }

    /* ========== 内部工具 ========== */

    _label(text, fontSize = 28, color = '#fff') {
        const c = document.createElement('canvas');
        c.width = 256; c.height = 64;
        const ctx = c.getContext('2d');
        ctx.font = `bold ${fontSize}px system-ui, -apple-system, sans-serif`;
        ctx.fillStyle = color;
        ctx.textBaseline = 'middle';
        ctx.fillText(text, 4, 32);
        const tex = new THREE.CanvasTexture(c);
        const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false });
        const s = new THREE.Sprite(mat);
        s.scale.set(1.5, 0.375, 1);
        s.raycast = () => {};
        return s;
    }

    /* ========== 交互 ========== */

    _click() {
        if (this.hoveredIdx >= 0) {
            this.activeIdx = this.activeIdx === this.hoveredIdx ? -1 : this.hoveredIdx;
        } else {
            this.activeIdx = -1;
        }
        if (this.onClick) {
            const p = this.activeIdx >= 0 ? this.planes[this.activeIdx] : null;
            this.onClick(p ? p.layer : -1, p ? p.head : -1);
        }
    }

    /* ========== 渲染循环 ========== */

    _loop() {
        requestAnimationFrame(() => this._loop());
        const dt = Math.min(this.clock.getDelta(), 0.033);  // 限制 dt ≤ 33ms (≥ 30fps)
        this._raycast(dt);
        this._hover(dt);
        this._cam(dt);
        this.renderer.render(this.scene, this.camera);
    }

    _raycast(dt) {
        if (!this.planes.length) return;

        // ── 鼠标速度计算 (NDC/s) ──
        if (this._prevMouseNDC.x > -900) {
            const dx = this.mouseNDC.x - this._prevMouseNDC.x;
            const dy = this.mouseNDC.y - this._prevMouseNDC.y;
            const speed = Math.sqrt(dx * dx + dy * dy) / Math.max(dt, 0.001);
            this._mouseVelocity = this._mouseVelocity * 0.7 + speed * 0.3;
        }
        this._prevMouseNDC.copy(this.mouseNDC);

        // ── 射线检测前: 暂时将所有 mesh.position.y 重置为基础值 ──
        //    避免已上浮的平面遮挡后方平面导致抽搐
        const savedY = [];
        for (const p of this.planes) {
            savedY.push(p.mesh.position.y);
            p.mesh.position.y = p.posSpring.value.y;
        }

        this.raycaster.setFromCamera(this.mouseNDC, this.camera);
        const visible = this.planes.filter(p => p.mesh.visible).map(p => p.mesh);
        const hits = this.raycaster.intersectObjects(visible, false);
        const rawIdx = hits.length ? this.planes.findIndex(p => p.mesh === hits[0].object) : -1;

        // ── 恢复实际 Y ──
        for (let i = 0; i < this.planes.length; i++) {
            this.planes[i].mesh.position.y = savedY[i];
        }

        // ── 鼠标移动过快时锁定当前悬浮目标 ──
        if (this._mouseVelocity > MOUSE_VEL_GATE && this.hoveredIdx >= 0 && rawIdx !== this.hoveredIdx) {
            return;
        }

        // ── 非对称防抖 ──
        // 1. 从有效目标切换到另一个有效目标 → 最长延迟 (防止边缘抽搐)
        // 2. 从有效目标到空 → 中等延迟
        // 3. 从空到有效目标 → 基础延迟
        let threshold = HOVER_DEBOUNCE;
        if (this.hoveredIdx >= 0 && rawIdx >= 0 && rawIdx !== this.hoveredIdx) {
            threshold = HOVER_DEBOUNCE * HOVER_SWITCH_MULT;
        } else if (rawIdx < 0 && this.hoveredIdx >= 0) {
            threshold = HOVER_DEBOUNCE * HOVER_LEAVE_MULT;
        }

        if (rawIdx === this.hoveredIdx) {
            this._pendingHover = rawIdx;
            this._hoverTimer = 0;
        } else if (rawIdx !== this._pendingHover) {
            this._pendingHover = rawIdx;
            this._hoverTimer = 0;
        } else {
            this._hoverTimer += dt;
            if (this._hoverTimer >= threshold) {
                this.hoveredIdx = rawIdx;
                this.canvas.style.cursor = rawIdx >= 0 ? 'pointer' : 'default';
                if (this.onHover) {
                    const p = rawIdx >= 0 ? this.planes[rawIdx] : null;
                    this.onHover(p ? p.layer : -1, p ? p.head : -1);
                }
            }
        }
    }

    _hover(dt) {
        const hoverDelta = this.is2D ? Y_HOVER_2D : Y_HOVER;
        const zoomed = this.activeIdx >= 0; // 单图预览模式

        for (let i = 0; i < this.planes.length; i++) {
            const p = this.planes[i];
            const active = i === this.hoveredIdx || i === this.activeIdx;

            // 位置弹簧 (2D↔3D 过渡)
            p.posSpring.setTarget(p.basePos[0], p.basePos[1], p.basePos[2]);
            p.posSpring.step(dt);
            p.mesh.position.x = p.posSpring.value.x;
            p.mesh.position.z = p.posSpring.value.z;

            // 悬浮弹簧 (Y 轴)
            p.hoverSpring.target = active ? p.posSpring.value.y + hoverDelta : p.posSpring.value.y;
            p.hoverSpring.step(dt);
            p.mesh.position.y = p.hoverSpring.value;

            // 旋转弹簧 (X 轴 — 2D 平放 / 3D 竖直)
            p.rotSpring.step(dt);
            p.mesh.rotation.x = p.rotSpring.value;

            // 单图预览时隐藏 mesh 上的标签精灵，避免遮挡
            const label = p.mesh.children[0];
            if (label) label.visible = !zoomed;
        }
    }

    _cam(dt) {
        // ── 边缘滚动 ──
        if (this.mouseActive && this.activeIdx < 0) {
            const x = this.windowMouseX;
            if (Math.abs(x) >= EDGE_ZONE) {
                const a = SCROLL_SPEED, b = EDGE_ZONE, c = 2;
                const spd = a * ((x * x - b * b) / (1 - b * b)) ** c;
                this.sceneCenter += spd * dt * Math.sign(x);
                this.sceneCenter = Math.max(this.sceneBoundsMin, Math.min(this.sceneCenter, this.sceneBoundsMax));
            }
        }

        // ── 目标计算 (仿照原项目 CameraAnimator) ──
        if (this.activeIdx >= 0) {
            const bp = this.planes[this.activeIdx].basePos;
            const worldY = bp[1] + GROUP_Y;
            const cx = bp[0], cz = bp[2];
            if (this.is2D) {
                // 2D 选中: 俯视放大
                this.camSpring.setTarget(cx, worldY + ZOOM_DIST + 2, cz + 0.01);
                this.lookAtSpring.setTarget(cx, worldY, cz);
            } else {
                // 3D 选中: 侧面放大
                const lookY = worldY + Y_HOVER;
                this.camSpring.setTarget(cx, lookY + 1, cz + ZOOM_DIST);
                this.lookAtSpring.setTarget(cx, lookY, cz);
            }
        } else if (this.is2D) {
            // 2D 俯视全局 — 包含 Layer 标签行的完整范围
            const firstX = computePosition2D(0, 0)[0];
            const lastX  = computePosition2D(NUM_LAYERS - 1, 0)[0];
            const labelRowZ = -(NUM_HEADS) * (GRID_CELL + GRID_GAP);  // Layer 标签行
            const botZ   = computePosition2D(0, NUM_HEADS - 1)[2];   // 最下方一行
            const cx = (firstX + lastX) / 2;
            const cz = (labelRowZ + botZ) / 2;       // 标签到最底行的中心
            const spanX = lastX - firstX + GRID_CELL + 6;   // +6 留白给 Head 标签
            const spanZ = Math.abs(botZ - labelRowZ) + GRID_CELL + 3;
            // 根据视口宽高比决定相机高度
            const aspect = this.camera.aspect || 1;
            const fov = this.camera.fov * Math.PI / 180;
            const hFromZ = (spanZ / 2) / Math.tan(fov / 2);
            const hFromX = (spanX / 2) / (aspect * Math.tan(fov / 2));
            const camHeight = Math.max(hFromZ, hFromX) * 1.1;
            this.camSpring.setTarget(cx, camHeight, cz + 0.01);
            this.lookAtSpring.setTarget(cx, 0, cz);
        } else {
            this.camSpring.setTarget(
                DEFAULT_CAM.x + this.sceneCenter,
                DEFAULT_CAM.y,
                DEFAULT_CAM.z
            );
            this.lookAtSpring.setTarget(this.sceneCenter, 0, 0);
        }

        // ── 弹簧步进 ──
        this.camSpring.step(dt);
        this.lookAtSpring.step(dt);

        // ── 帧率无关平滑（替代固定 lerp alpha，避免不同帧率下手感不同） ──
        const alpha = 1 - Math.exp(-CAM_SMOOTH_RATE * dt);
        this.currentCamPos.lerp(this.camSpring.value, alpha);
        this.currentLookAt.lerp(this.lookAtSpring.value, alpha);

        this.camera.position.copy(this.currentCamPos);
        this.camera.lookAt(this.currentLookAt);
    }
}
