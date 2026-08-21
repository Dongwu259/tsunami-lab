import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { TsunamiSolver } from '../simulation/TsunamiSolver';
import {
  GLOBE_TERRAIN_VERT,
  GLOBE_WATER_VERT,
  TERRAIN_FRAG,
  TERRAIN_VERT,
  WATER_FRAG,
  WATER_VERT,
} from '../simulation/shaders';

/** 视图模式:平面区域 / 全球球面 */
export type ViewMode = 'plane' | 'globe';

/** 全球模式球体半径(场景单位) */
const GLOBE_RADIUS = 120;

/**
 * 三维场景:海面网格、海底地形、相机与交互拾取。
 * 场景单位:水平 1 单位 = 1 km;垂直方向经夸张系数处理。
 */
export class SceneApp {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  private controls: OrbitControls;

  /** 当前视图模式 */
  private mode: ViewMode = 'plane';

  private waterMat: THREE.ShaderMaterial;
  private terrainMat: THREE.ShaderMaterial;
  private waterMesh: THREE.Mesh;
  private terrainMesh: THREE.Mesh;
  /** 海床纹理的 Linear 采样克隆(渲染用,随材质重建) */
  private smoothBathy: THREE.Texture | null = null;
  /** smoothBathy 是否为自建克隆(false = 直接引用 renderBathy,不可 dispose) */
  private smoothBathyOwned = false;
  /** 海面不透明度(跨模式保留用户设置) */
  private waterOpacity = 0.8;

  /** 渲染网格分辨率(可高于求解网格,仅影响地形/海面细节)
   *  null = 跟随求解网格 */
  private renderSize: { x: number; y: number } | null = null;
  /** 高分辨率渲染用海床纹理(可独立于求解器 bathyTexture) */
  private renderBathy: THREE.Texture | null = null;
  private renderBathySize: { x: number; y: number } | null = null;

  /** P/S 波圈叠加状态(地震体波可视化) */
  private seisOn = true;
  private seisT = 0;
  private readonly seisEpi = new THREE.Vector2(0.5, 0.5);

  /** 观测点(验潮站)标记 */
  private observerGroup = new THREE.Group();
  private observerGeo: THREE.ConeGeometry | null = null;
  private readonly observerMat = new THREE.MeshBasicMaterial({ color: 0xffd166 });

  /** 震源标记(地震学标准叉号) */
  private epicenterGroup = new THREE.Group();
  private epicenterGeos: THREE.BufferGeometry[] = [];
  private readonly epicenterMat = new THREE.MeshBasicMaterial({ color: 0xff5c5c });

  /** 相机飞行动画 */
  private camAnim: {
    cp: THREE.Vector3; ct: THREE.Vector3;
    pp: THREE.Vector3; pt: THREE.Vector3;
    t0: number; dur: number;
  } | null = null;

  /** 当前域尺度(km) */
  readonly domainKm = new THREE.Vector2(240, 240);

  private raycaster = new THREE.Raycaster();
  private pointerNdc = new THREE.Vector2();

  constructor(
    container: HTMLElement,
    renderer: THREE.WebGLRenderer,
    solver: TsunamiSolver
  ) {
    this.renderer = renderer;

    this.scene.background = new THREE.Color(0x05080f);

    // --- 相机与轨道控制 ---
    this.camera = new THREE.PerspectiveCamera(
      55,
      container.clientWidth / container.clientHeight,
      0.1,
      4000
    );
    this.camera.position.set(-70, 175, 285);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, -10, 0);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = 40;
    this.controls.maxDistance = 1400;
    this.controls.maxPolarAngle = Math.PI * 0.49;

    const texel = new THREE.Vector2(1 / solver.sizeX, 1 / solver.sizeY);
    this.domainKm.set(
      (solver.dxM * solver.sizeX) / 1000,
      (solver.dyM * solver.sizeY) / 1000
    );

    // --- 海底地形 ---
    this.terrainMat = new THREE.ShaderMaterial({
      vertexShader: TERRAIN_VERT,
      fragmentShader: TERRAIN_FRAG,
      uniforms: {
        uBathymetry: { value: solver.bathyTexture },
        uTexel: { value: texel },
        uTerrainScale: { value: 0.003 },
        uDomainKm: { value: this.domainKm },
      },
    });
    this.terrainMesh = new THREE.Mesh(this.buildPlane(solver), this.terrainMat);
    this.terrainMesh.rotation.x = -Math.PI / 2;
    this.scene.add(this.terrainMesh);

    // --- 海面 ---
    this.waterMat = new THREE.ShaderMaterial({
      vertexShader: WATER_VERT,
      fragmentShader: WATER_FRAG,
      uniforms: {
        uState: { value: solver.stateTexture },
        uTexel: { value: texel },
        uWaveScale: { value: 0.045 },
        uDomainKm: { value: this.domainKm },
        uColorScale: { value: 0.5 },
      },
      polygonOffset: true,
      polygonOffsetFactor: -1,
      polygonOffsetUnits: -1,
    });
    this.waterMesh = new THREE.Mesh(this.buildPlane(solver), this.waterMat);
    this.waterMesh.rotation.x = -Math.PI / 2;
    this.scene.add(this.waterMesh);

    // setDomain 会统一重建材质与网格(此处构造的材质随即被替换)
    this.setDomain(solver);

    this.scene.add(this.observerGroup);
    this.scene.add(this.epicenterGroup);

    this.initSize(container);
    window.addEventListener('resize', this.onResize);
  }

  /** 按求解器网格与域尺度构建平面网格 */
  private buildPlane(solver: TsunamiSolver): THREE.PlaneGeometry {
    return new THREE.PlaneGeometry(
      this.domainKm.x,
      this.domainKm.y,
      solver.sizeX - 1,
      solver.sizeY - 1
    );
  }

  /** 球面模式的网格:段数与渲染分辨率对齐(上限 1024×512),
   * 避免顶点采样与地形纹理栅格错位产生混叠碎斑。
   * 地形层半径略小于水层,避免海岸带 z-fighting。 */
  private buildSphere(terrain = false): THREE.SphereGeometry {
    const rx = this.renderSize?.x ?? 1024;
    const ry = this.renderSize?.y ?? 512;
    const segX = Math.min(rx, 1024);
    const segY = Math.min(ry, 512);
    return new THREE.SphereGeometry(
      terrain ? GLOBE_RADIUS * 0.998 : GLOBE_RADIUS, segX, segY
    );
  }

  /** 按当前模式创建海面/地形材质。
   * 水面采样求解状态纹理(求解分辨率);地形与网格段数可用更高分辨率
   * (renderSize),提升全球地形视觉细节。 */
  private buildMaterials(solver: TsunamiSolver): void {
    const texel = new THREE.Vector2(1 / solver.sizeX, 1 / solver.sizeY);
    // 地形法线偏移按渲染网格格距取;若渲染纹理比网格更细,
    // 按倍数放大偏移,使法线仍对应相邻顶点的实际高差
    const rX = this.renderSize?.x ?? solver.sizeX;
    const rY = this.renderSize?.y ?? solver.sizeY;
    const tX = this.renderBathySize?.x ?? rX;
    const tY = this.renderBathySize?.y ?? rY;
    const rTexel = new THREE.Vector2((tX / rX) / tX, (tY / rY) / tY);
    this.waterMat?.dispose();
    this.terrainMat?.dispose();

    // 渲染采样需双线性:优先用独立高分纹理;否则用海床计算纹理的
    // Linear 克隆,消除球面顶点采样的栅格混叠。
    // 浮点纹理线性过滤需 OES_texture_float_linear,不支持则保持 Nearest
    // (此时靠网格与纹理段数对齐避免混叠)
    if (this.smoothBathyOwned) this.smoothBathy?.dispose();
    this.smoothBathy = null;
    if (this.renderBathy) {
      this.smoothBathy = this.renderBathy;
      this.smoothBathyOwned = false;
    } else {
      this.smoothBathy = solver.bathyTexture.clone();
      if (this.renderer.extensions.get('OES_texture_float_linear')) {
        this.smoothBathy.minFilter = THREE.LinearFilter;
        this.smoothBathy.magFilter = THREE.LinearFilter;
      }
      this.smoothBathy.needsUpdate = true;
      this.smoothBathyOwned = true;
    }

    if (this.mode === 'globe') {
      this.waterMat = new THREE.ShaderMaterial({
        vertexShader: GLOBE_WATER_VERT,
        fragmentShader: WATER_FRAG,
        uniforms: {
          uState: { value: solver.stateTexture },
          uTexel: { value: texel },
          uWaveScale: { value: 0.5 },
          uColorScale: { value: 0.5 },
          uOpacity: { value: this.waterOpacity },
          ...this.seisUniforms(true),
        },
        transparent: true,
        polygonOffset: true,
        polygonOffsetFactor: -1,
        polygonOffsetUnits: -1,
      });
      this.terrainMat = new THREE.ShaderMaterial({
        vertexShader: GLOBE_TERRAIN_VERT,
        fragmentShader: TERRAIN_FRAG,
        uniforms: {
          uBathymetry: { value: this.smoothBathy },
          uTexel: { value: rTexel },
          uTerrainScale: { value: 0.0012 },
          // 略大于球壳半径差(0.002R),保证任意夸张系数下陆地都在水面球之上
          uLandLift: { value: GLOBE_RADIUS * 0.0025 },
          ...this.seisUniforms(true),
        },
      });
    } else {
      this.waterMat = new THREE.ShaderMaterial({
        vertexShader: WATER_VERT,
        fragmentShader: WATER_FRAG,
        uniforms: {
          uState: { value: solver.stateTexture },
          uTexel: { value: texel },
          uWaveScale: { value: 0.045 },
          uDomainKm: { value: this.domainKm },
          uColorScale: { value: 0.5 },
          uOpacity: { value: this.waterOpacity },
          ...this.seisUniforms(false),
        },
        transparent: true,
        polygonOffset: true,
        polygonOffsetFactor: -1,
        polygonOffsetUnits: -1,
      });
      this.terrainMat = new THREE.ShaderMaterial({
        vertexShader: TERRAIN_VERT,
        fragmentShader: TERRAIN_FRAG,
        uniforms: {
          uBathymetry: { value: this.smoothBathy },
          uTexel: { value: rTexel },
          uTerrainScale: { value: 0.003 },
          uDomainKm: { value: this.domainKm },
          ...this.seisUniforms(false),
        },
      });
    }
  }

  /** P/S 波圈叠加 uniforms(重建材质时保持引用,供每帧原地更新) */
  private seisUniforms(globe: boolean): Record<string, THREE.IUniform> {
    return {
      uSeisOn: { value: this.seisOn ? 1 : 0 },
      uSeisT: { value: this.seisT },
      uEpi: { value: this.seisEpi },
      uEpiGlobe: { value: globe ? 1 : 0 },
      uSeisDomKm: { value: this.domainKm },
    };
  }

  /** 设置 P/S 波圈显示开关 */
  setSeismicVisible(on: boolean): void {
    this.seisOn = on;
    this.waterMat.uniforms.uSeisOn.value = on ? 1 : 0;
    this.terrainMat.uniforms.uSeisOn.value = on ? 1 : 0;
  }

  /** 记录新震源:波圈时间归零并从该震中重新扩散 */
  setSeismicOrigin(u: number, v: number): void {
    this.seisEpi.set(u, v);
    this.seisT = 0;
    this.waterMat.uniforms.uSeisT.value = 0;
    this.terrainMat.uniforms.uSeisT.value = 0;
  }

  /** 每帧同步发震后经过的模拟时间(s) */
  setSeismicTime(t: number): void {
    this.seisT = t;
    this.waterMat.uniforms.uSeisT.value = t;
    this.terrainMat.uniforms.uSeisT.value = t;
  }

  /** 切换视图模式(平面区域 ↔ 全球球面)。材质与网格的构建统一由
   * setDomain 完成,此处仅记录模式。 */
  setMode(mode: ViewMode, solver: TsunamiSolver): void {
    this.mode = mode;
    this.setDomain(solver);
  }

  /** 设置高分辨率渲染用海床纹理(独立于求解网格,仅提升地形视觉细节);
   * 传 null 恢复为采样求解器海床纹理 */
  setRenderBathy(tex: THREE.Texture | null, size?: { x: number; y: number }): void {
    if (this.renderBathy && this.renderBathy !== tex) this.renderBathy.dispose();
    this.renderBathy = tex;
    this.renderBathySize = tex ? (size ?? null) : null;
  }

  /** 切换仿真域(导入新地形/切换模式后调用):重建材质、网格、uniform 并重新取景。
   * renderSize:可选的渲染网格分辨率(高于求解网格时提升地形细节) */
  setDomain(solver: TsunamiSolver, renderSize?: { x: number; y: number }): void {
    this.renderSize = renderSize ?? null;
    this.buildMaterials(solver);
    this.waterMesh.material = this.waterMat;
    this.terrainMesh.material = this.terrainMat;

    this.domainKm.set(
      (solver.dxM * solver.sizeX) / 1000,
      (solver.dyM * solver.sizeY) / 1000
    );

    const oldWaterGeo = this.waterMesh.geometry;
    const oldTerrainGeo = this.terrainMesh.geometry;
    if (this.mode === 'globe') {
      this.waterMesh.geometry = this.buildSphere();
      this.terrainMesh.geometry = this.buildSphere(true);
    } else {
      this.waterMesh.geometry = this.buildPlane(solver);
      this.terrainMesh.geometry = this.buildPlane(solver);
    }
    oldWaterGeo.dispose();
    oldTerrainGeo.dispose();

    // 平面网格绕 x 轴放平;球体保持原位
    const flat = this.mode === 'plane';
    this.waterMesh.rotation.x = flat ? -Math.PI / 2 : 0;
    this.terrainMesh.rotation.x = flat ? -Math.PI / 2 : 0;

    this.reframe();
  }

  /** 根据域尺度重新取景 */
  reframe(): void {
    if (this.mode === 'globe') {
      this.camera.position.set(0, 0.55 * GLOBE_RADIUS, 3.1 * GLOBE_RADIUS);
      this.controls.target.set(0, 0, 0);
      this.controls.minDistance = GLOBE_RADIUS * 1.25;
      this.controls.maxDistance = GLOBE_RADIUS * 8;
    } else {
      const d = Math.max(this.domainKm.x, this.domainKm.y);
      this.camera.position.set(-0.28 * d, 0.72 * d, 1.18 * d);
      this.controls.target.set(0, -0.04 * d, 0);
      this.controls.minDistance = d * 0.15;
      this.controls.maxDistance = d * 6;
    }
    this.controls.update();
  }

  private onResize = (): void => {
    const el = this.renderer.domElement.parentElement;
    if (!el) return;
    this.camera.aspect = el.clientWidth / el.clientHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(el.clientWidth, el.clientHeight);
  };

  /** 画布尺寸初始化为容器大小 */
  private initSize(container: HTMLElement): void {
    this.renderer.setSize(container.clientWidth, container.clientHeight);
  }

  /** 每帧同步最新状态纹理 */
  syncState(solver: TsunamiSolver): void {
    this.waterMat.uniforms.uState.value = solver.stateTexture;
  }

  /** 设置垂直夸张系数(同时作用于波形与地形,按模式取不同基准) */
  setExaggeration(ex: number): void {
    if (this.mode === 'globe') {
      this.waterMat.uniforms.uWaveScale.value = ex * 0.17;
      this.terrainMat.uniforms.uTerrainScale.value = ex * 0.0004;
    } else {
      this.waterMat.uniforms.uWaveScale.value = ex * 0.015;
      this.terrainMat.uniforms.uTerrainScale.value = ex * 0.001;
    }
  }

  /** 设置色标半量程(m) */
  setColorRange(meters: number): void {
    this.waterMat.uniforms.uColorScale.value = 1 / meters;
  }

  /** 设置海面不透明度(0–1),调低可透视海底地形 */
  setWaterOpacity(opacity: number): void {
    this.waterOpacity = opacity;
    this.waterMat.uniforms.uOpacity.value = opacity;
  }

  setTerrainVisible(visible: boolean): void {
    this.terrainMesh.visible = visible;
  }

  setWireframe(on: boolean): void {
    this.waterMat.wireframe = on;
    this.terrainMat.wireframe = on;
  }

  /** 更新观测点标记(uv 为求解域坐标;空列表清空) */
  setObserverMarkers(uvs: { u: number; v: number }[]): void {
    this.observerGroup.clear();
    this.observerGeo?.dispose();
    this.observerGeo = null;
    if (uvs.length === 0) return;

    if (this.mode === 'globe') {
      this.observerGeo = new THREE.ConeGeometry(1.2, 4, 8);
      const R = GLOBE_RADIUS * 1.008;
      const up = new THREE.Vector3(0, 1, 0);
      const dir = new THREE.Vector3();
      for (const { u, v } of uvs) {
        // 与 SphereGeometry 参数化一致:uv.y=1 ↔ 北极,uv.x=0 ↔ 180°W
        const phi = 2 * Math.PI * u;
        const theta = Math.PI * (1 - v);
        const sinT = Math.sin(theta);
        dir.set(-Math.cos(phi) * sinT, Math.cos(theta), Math.sin(phi) * sinT);
        const m = new THREE.Mesh(this.observerGeo, this.observerMat);
        m.position.copy(dir).multiplyScalar(R);
        m.quaternion.setFromUnitVectors(up, dir);
        this.observerGroup.add(m);
      }
    } else {
      // 平面域:标记尺寸随域尺度缩放,悬于海面上方
      const d = Math.max(this.domainKm.x, this.domainKm.y);
      const s = Math.max(d * 0.014, 1.5);
      this.observerGeo = new THREE.ConeGeometry(s * 0.35, s, 8);
      for (const { u, v } of uvs) {
        const m = new THREE.Mesh(this.observerGeo, this.observerMat);
        // 平面网格绕 x 轴旋平后:局部 (x, y) → 世界 (x, 0, -y)
        m.position.set((u - 0.5) * this.domainKm.x, s * 0.8, (0.5 - v) * this.domainKm.y);
        this.observerGroup.add(m);
      }
    }
  }

  /** uv → 球面单位方向向量(与 SphereGeometry 参数化一致) */
  private globeDir(u: number, v: number): THREE.Vector3 {
    const phi = 2 * Math.PI * u;
    const theta = Math.PI * (1 - v);
    const sinT = Math.sin(theta);
    return new THREE.Vector3(
      -Math.cos(phi) * sinT, Math.cos(theta), Math.sin(phi) * sinT
    );
  }

  /** 相机飞行到指定 uv 上方(约 0.9 s 平滑过渡) */
  focusUv(u: number, v: number): void {
    let camPos: THREE.Vector3;
    let target: THREE.Vector3;
    if (this.mode === 'globe') {
      const dir = this.globeDir(u, v);
      camPos = dir.multiplyScalar(GLOBE_RADIUS * 2.9);
      target = new THREE.Vector3(0, 0, 0);
    } else {
      const p = new THREE.Vector3(
        (u - 0.5) * this.domainKm.x, 0, (0.5 - v) * this.domainKm.y
      );
      const d = Math.max(this.domainKm.x, this.domainKm.y);
      camPos = p.clone().add(new THREE.Vector3(-0.15 * d, 0.38 * d, 0.6 * d));
      target = p;
    }
    this.camAnim = {
      cp: this.camera.position.clone(), ct: camPos,
      pp: this.controls.target.clone(), pt: target,
      t0: performance.now(), dur: 900,
    };
    this.controls.enabled = false; // 动画期间禁止手动操作,结束后恢复
  }

  /** 震源叉号标记(null 清除) */
  setEpicenterMarker(uv: { u: number; v: number } | null): void {
    this.epicenterGroup.clear();
    for (const g of this.epicenterGeos) g.dispose();
    this.epicenterGeos = [];
    if (!uv) return;

    if (this.mode === 'globe') {
      // 两条交叉薄片,贴在球面切平面上
      const g1 = new THREE.BoxGeometry(7, 1.0, 0.35);
      const g2 = new THREE.BoxGeometry(1.0, 7, 0.35);
      this.epicenterGeos.push(g1, g2);
      const dir = this.globeDir(uv.u, uv.v);
      const quat = new THREE.Quaternion().setFromUnitVectors(
        new THREE.Vector3(0, 0, 1), dir
      );
      for (const g of [g1, g2]) {
        const m = new THREE.Mesh(g, this.epicenterMat);
        m.position.copy(dir).multiplyScalar(GLOBE_RADIUS * 1.01);
        m.quaternion.copy(quat);
        this.epicenterGroup.add(m);
      }
    } else {
      // 平面域:两条绕 Y 轴 ±45° 的长条组成叉号,悬于海面上方
      const d = Math.max(this.domainKm.x, this.domainKm.y);
      const s = Math.max(d * 0.02, 2);
      const g1 = new THREE.BoxGeometry(s * 2.6, s * 0.3, s * 0.3);
      const g2 = new THREE.BoxGeometry(s * 2.6, s * 0.3, s * 0.3);
      this.epicenterGeos.push(g1, g2);
      const pos = new THREE.Vector3(
        (uv.u - 0.5) * this.domainKm.x, s * 0.6, (0.5 - uv.v) * this.domainKm.y
      );
      const m1 = new THREE.Mesh(g1, this.epicenterMat);
      m1.position.copy(pos);
      m1.rotation.y = Math.PI / 4;
      const m2 = new THREE.Mesh(g2, this.epicenterMat);
      m2.position.copy(pos);
      m2.rotation.y = -Math.PI / 4;
      this.epicenterGroup.add(m1, m2);
    }
  }

  /** 屏幕坐标 → 海面 uv;拾取失败返回 null */
  pickUv(clientX: number, clientY: number): { u: number; v: number } | null {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointerNdc.set(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -((clientY - rect.top) / rect.height) * 2 + 1
    );
    this.raycaster.setFromCamera(this.pointerNdc, this.camera);
    const hits = this.raycaster.intersectObject(this.waterMesh, false);
    if (hits.length > 0 && hits[0].uv) {
      return { u: hits[0].uv.x, v: hits[0].uv.y };
    }
    return null;
  }

  render(): void {
    if (this.camAnim) {
      const a = this.camAnim;
      let t = (performance.now() - a.t0) / a.dur;
      if (t >= 1) {
        t = 1;
        this.camAnim = null;
        this.controls.enabled = true;
      }
      const e = t * t * (3 - 2 * t); // smoothstep 缓动
      this.camera.position.lerpVectors(a.cp, a.ct, e);
      this.controls.target.lerpVectors(a.pp, a.pt, e);
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  dispose(): void {
    window.removeEventListener('resize', this.onResize);
    this.controls.dispose();
    this.renderBathy?.dispose();
    this.renderer.dispose();
  }
}
