import * as THREE from 'three';
import { computeStableDt, GRAVITY, H_MIN, MANNING_N, GLOBE_LAT_SPAN, reducedGridMinFactor } from '../config';
import { INJECT_FRAG, STEP_FRAG, STEP_FRAG_V2 } from './shaders';

/** 用于将纹理填充为常数的极简着色器 */
const FILL_FRAG = /* glsl */ `
uniform float uValue;
void main() { gl_FragColor = vec4(uValue); }
`;
const FILL_VERT = /* glsl */ `
void main() { gl_Position = vec4(position.xy, 0.0, 1.0); }
`;

/** 将外部 η 场拷贝/叠加进状态纹理(科研回放与真实震源共用) */
const COPY_FRAG = /* glsl */ `
uniform sampler2D uSrc;
uniform sampler2D uState;   // 当前状态(叠加模式读取;普通材质需手动绑定)
uniform vec2 uRes;
uniform float uAdd;   // 1 = 叠加到现有 η(动量保留);0 = 替换并清零动量
void main() {
  vec2 uv = gl_FragCoord.xy / uRes;
  float eta = texture2D(uSrc, uv).r;
  if (uAdd > 0.5) {
    vec4 cur = texture2D(uState, uv);
    gl_FragColor = vec4(cur.r + eta, cur.g, cur.b, 1.0);
  } else {
    gl_FragColor = vec4(eta, 0.0, 0.0, 1.0);
  }
}
`;

/** 数值格式:0 = LF(一阶,教学对照),1 = MUSCL-Rusanov-RK2(二阶) */
export type GpuScheme = 0 | 1;

/**
 * GPU 海啸求解器。
 * 手动 ping-pong 两个浮点渲染目标推进浅水方程,状态编码 RGBA:r=η, g=Hu, b=Hv。
 * RK2 格式每步两个 pass(阶段 A 写 other、阶段 B 写回 current),
 * 震源注入为独立 pass(INJECT_FRAG)。dt 由 CFL 条件自动计算。
 */
export class TsunamiSolver {
  /** 网格尺寸(x = 经向/宽,y = 纬向/高) */
  readonly sizeX: number;
  readonly sizeY: number;
  /** 网格物理边长(m) */
  readonly dxM: number;
  readonly dyM: number;
  /** 全球球面模式(经向周期环绕 + 纬度度量) */
  readonly globeMode: boolean;
  /** 时间步长(s,CFL 自动:0.5·min(dxEff,dy)/√(g·Hmax);globe 用缩减纬网最小经向格距) */
  readonly dtSeconds: number;

  private renderer: THREE.WebGLRenderer;
  /** ping-pong 状态渲染目标(cur 为当前状态) */
  private rts: [THREE.WebGLRenderTarget, THREE.WebGLRenderTarget];
  /** RK2 中间级 U* 暂存目标(不参与 ping-pong) */
  private rtU0: THREE.WebGLRenderTarget;
  private cur = 0;
  private lfMaterial: THREE.ShaderMaterial;
  private v2Material: THREE.ShaderMaterial;
  private injectMaterial: THREE.ShaderMaterial;
  private fillMaterial: THREE.ShaderMaterial;
  private copyMaterial: THREE.ShaderMaterial;
  private quadScene = new THREE.Scene();
  private fillScene = new THREE.Scene();
  private copyScene = new THREE.Scene();
  private injectScene = new THREE.Scene();
  private camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

  /** 求解器 uniform(供外部调整阻尼、格式开关等) */
  readonly uniforms: Record<string, THREE.IUniform>;

  /** 海床高程纹理(静态) */
  readonly bathyTexture: THREE.DataTexture;

  constructor(
    renderer: THREE.WebGLRenderer,
    bathymetry: Float32Array,
    sizeX: number,
    sizeY: number,
    dxM: number,
    dyM: number,
    globeMode = false,
    scheme: GpuScheme = 1
  ) {
    this.renderer = renderer;
    this.sizeX = sizeX;
    this.sizeY = sizeY;
    this.dxM = dxM;
    this.dyM = dyM;
    this.globeMode = globeMode;

    // CFL 自动时间步长:ν = dt·√(g·Hmax)/min(dxEff,dy) = CFL_SAFETY
    let maxDepth = 1;
    for (let i = 0; i < bathymetry.length; i++) {
      const d = -bathymetry[i];
      if (d > maxDepth) maxDepth = d;
    }
    // 缩减纬网:globe 经向有效格距按 min(k·cosφ) 缩减(与着色器/CPU dxEff 同构),
    // dt 取缩减后最小格距 → 去除旧 cflFloor 后极地仍稳定且波速正确
    const dxEffMin = globeMode ? dxM * reducedGridMinFactor(GLOBE_LAT_SPAN, sizeY) : dxM;
    this.dtSeconds = computeStableDt(dxEffMin, dyM, maxDepth);

    // --- 海床纹理 ---
    const bathyTex = new THREE.DataTexture(
      new Float32Array(sizeX * sizeY * 4), sizeX, sizeY,
      THREE.RGBAFormat, THREE.FloatType
    );
    const bData = bathyTex.image.data as unknown as Float32Array;
    for (let i = 0; i < bathymetry.length; i++) {
      bData[i * 4] = bathymetry[i];
      bData[i * 4 + 3] = 1;
    }
    bathyTex.minFilter = THREE.NearestFilter;
    bathyTex.magFilter = THREE.NearestFilter;
    bathyTex.needsUpdate = true;
    this.bathyTexture = bathyTex;

    // --- 状态渲染目标(ping-pong) ---
    const type = renderer.capabilities.isWebGL2
      ? THREE.FloatType
      : THREE.HalfFloatType;
    const mkRt = (): THREE.WebGLRenderTarget => {
      const rt = new THREE.WebGLRenderTarget(sizeX, sizeY, {
        minFilter: THREE.NearestFilter,
        magFilter: THREE.NearestFilter,
        format: THREE.RGBAFormat,
        type,
        depthBuffer: false,
        stencilBuffer: false,
      });
      // 球面模式:经向周期环绕,波可跨越 180° 经线
      rt.texture.wrapS = globeMode
        ? THREE.RepeatWrapping
        : THREE.ClampToEdgeWrapping;
      rt.texture.wrapT = THREE.ClampToEdgeWrapping;
      return rt;
    };
    this.rts = [mkRt(), mkRt()];
    this.rtU0 = mkRt();

    // --- 共享 uniforms(步进/注入材质共用同一批对象) ---
    this.uniforms = {
      uState: { value: null },
      u0: { value: null },
      uBathymetry: { value: bathyTex },
      uTexel: { value: new THREE.Vector2(1 / sizeX, 1 / sizeY) },
      uRes: { value: new THREE.Vector2(sizeX, sizeY) },
      uDx: { value: dxM },
      uDy: { value: dyM },
      uDt: { value: this.dtSeconds },
      uG: { value: GRAVITY },
      uDamping: { value: 0.9998 },   // 仅 LF(legacy)格式使用
      uManning: { value: MANNING_N }, // V2 隐式曼宁摩擦系数 n
      uHMin: { value: H_MIN },        // 干单元阈值(m)
      uGlobeMode: { value: globeMode ? 1 : 0 },
      uScheme: { value: scheme },
      uStage: { value: 0 },
      uDiag: { value: 0 },
      uSource: { value: new THREE.Vector3(0.3, 0.5, 0.05) },
      uSourceAmp: { value: 0 },
    };
    const share = (extra?: Record<string, THREE.IUniform>): Record<string, THREE.IUniform> =>
      Object.assign({}, this.uniforms, extra ?? {});

    this.lfMaterial = new THREE.ShaderMaterial({
      uniforms: share(),
      vertexShader: FILL_VERT,
      fragmentShader: STEP_FRAG,
    });
    this.v2Material = new THREE.ShaderMaterial({
      uniforms: share(),
      vertexShader: FILL_VERT,
      fragmentShader: STEP_FRAG_V2,
    });
    this.injectMaterial = new THREE.ShaderMaterial({
      uniforms: share(),
      vertexShader: FILL_VERT,
      fragmentShader: INJECT_FRAG,
    });

    // 常数填充材质(重置海面)
    this.fillMaterial = new THREE.ShaderMaterial({
      uniforms: { uValue: { value: 0 } },
      vertexShader: FILL_VERT,
      fragmentShader: FILL_FRAG,
    });
    this.fillScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.fillMaterial));

    // η 场拷贝材质(回放 GeoClaw 结果帧)
    this.copyMaterial = new THREE.ShaderMaterial({
      uniforms: {
        uSrc: { value: null },
        uState: { value: null },
        uRes: { value: new THREE.Vector2(sizeX, sizeY) },
        uAdd: { value: 0 },
      },
      vertexShader: FILL_VERT,
      fragmentShader: COPY_FRAG,
    });
    this.copyScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.copyMaterial));

    const quadLf = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.lfMaterial);
    quadLf.frustumCulled = false;
    this.quadScene.add(quadLf);
    const quadInject = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.injectMaterial);
    quadInject.frustumCulled = false;
    this.injectScene.add(quadInject);

    this.fill(0);
  }

  private get curRt(): THREE.WebGLRenderTarget { return this.rts[this.cur]; }
  private get otherRt(): THREE.WebGLRenderTarget { return this.rts[1 - this.cur]; }

  /** 全屏 pass:scene 渲染到目标 RT(restore 当前渲染目标) */
  private pass(scene: THREE.Scene, target: THREE.WebGLRenderTarget | null): void {
    const prev = this.renderer.getRenderTarget();
    this.renderer.setRenderTarget(target);
    this.renderer.render(scene, this.camera);
    this.renderer.setRenderTarget(prev);
  }

  /** 步进材质切换(uScheme 运行时可改)+ 绑定输入纹理 + quad 显隐 */
  private bindStepMaterial(): THREE.ShaderMaterial {
    const v2 = (this.uniforms.uScheme.value as number) > 0.5;
    const mat = v2 ? this.v2Material : this.lfMaterial;
    const quad = this.quadScene.children[0] as THREE.Mesh;
    quad.material = mat;
    this.uniforms.uState.value = this.curRt.texture;
    return mat;
  }

  /** 最新状态纹理(供渲染材质采样;ping-pong 交替,须每帧重绑) */
  get stateTexture(): THREE.Texture {
    return this.curRt.texture;
  }

  /** 推进 substeps 个子步 */
  step(substeps: number): void {
    const v2 = (this.uniforms.uScheme.value as number) > 0.5;
    for (let i = 0; i < substeps; i++) {
      if (!v2) {
        this.bindStepMaterial();
        this.uniforms.u0.value = null;   // 防残留绑定与目标 RT 形成反馈环
        this.pass(this.quadScene, this.otherRt);
        this.cur = 1 - this.cur;
      } else {
        // 阶段 A:U* = U + dt·L(U),读 cur(U0)写暂存 rtU0
        this.bindStepMaterial();
        this.uniforms.u0.value = null;   // 阶段 A 不用 u0,置空防反馈环
        this.uniforms.uStage.value = 0;
        this.pass(this.quadScene, this.rtU0);
        // 阶段 B:U' = ½U0 + ½(U* + dt·L(U*)),uState=U*(rtU0)、u0=U0(cur),
        // 写 other 后交换;读写目标互斥,无反馈环
        this.uniforms.uState.value = this.rtU0.texture;
        this.uniforms.u0.value = this.curRt.texture;
        this.uniforms.uStage.value = 1;
        this.pass(this.quadScene, this.otherRt);
        this.cur = 1 - this.cur;
      }
    }
  }

  /**
   * 注入一次海底地震(高斯型海底抬升,Okada 解的教学级近似)。
   * 独立 pass:读当前缓冲、写另一缓冲后交换。
   * @param u,v      震中 uv 坐标(0–1)
   * @param ampMeters 初始波幅(m)
   * @param radiusMeters 破裂半径(m)
   */
  inject(u: number, v: number, ampMeters: number, radiusMeters: number): void {
    (this.uniforms.uSource.value as THREE.Vector3).set(
      u,
      v,
      radiusMeters / (this.dxM * this.sizeX)
    );
    this.uniforms.uSourceAmp.value = ampMeters;
    this.uniforms.uState.value = this.curRt.texture;
    this.pass(this.injectScene, this.otherRt);
    this.cur = 1 - this.cur;
  }

  /** 重置为平静海面 */
  reset(): void {
    this.fill(0);
  }

  private fill(value: number): void {
    this.fillMaterial.uniforms.uValue.value = value;
    for (const rt of this.rts) this.pass(this.fillScene, rt);
  }

  /**
   * 载入一帧外部海面位移场(科研模式回放)。
   * 动量清零,η 场双线性拉伸到求解器网格。
   * @param eta 自南向北行主序的 η(m)
   */
  loadEta(eta: Float32Array, nx: number, ny: number): void {
    this.applyEta(eta, nx, ny, false);
  }

  /**
   * 叠加一个 η 场到现有海面(真实震源注入)。
   * 动量保留;场尺寸与求解网格不同时双线性拉伸。
   */
  injectField(eta: Float32Array, nx: number, ny: number): void {
    this.applyEta(eta, nx, ny, true);
  }

  /** 公共实现:把 η 场写入状态(替换或叠加) */
  private applyEta(eta: Float32Array, nx: number, ny: number, additive: boolean): void {
    const rgba = new Float32Array(nx * ny * 4);
    for (let i = 0; i < eta.length; i++) {
      rgba[i * 4] = eta[i];
      rgba[i * 4 + 3] = 1;
    }
    const tex = new THREE.DataTexture(
      rgba, nx, ny, THREE.RGBAFormat, THREE.FloatType
    );
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.needsUpdate = true;

    this.copyMaterial.uniforms.uSrc.value = tex;
    this.copyMaterial.uniforms.uAdd.value = additive ? 1 : 0;
    if (!additive) {
      this.copyMaterial.uniforms.uState.value = null;
      this.pass(this.copyScene, this.curRt);
      this.pass(this.copyScene, this.otherRt);
    } else {
      // 叠加模式:避免自读自写,先写另一缓冲再交换
      this.copyMaterial.uniforms.uState.value = this.curRt.texture;
      this.pass(this.copyScene, this.otherRt);
      this.cur = 1 - this.cur;
    }
    tex.dispose();
  }

  /** 回读全场,返回峰值 |η|(m)。仅在低频统计时调用。
   * @param out 可选复用缓冲区(尺寸需匹配),避免每秒重分配 */
  readPeak(out?: Float32Array): number {
    const need = this.sizeX * this.sizeY * 4;
    const buf = out && out.length === need ? out : new Float32Array(need);
    this.renderer.readRenderTargetPixels(
      this.curRt, 0, 0, this.sizeX, this.sizeY, buf
    );
    let peak = 0;
    for (let i = 0; i < buf.length; i += 4) {
      const a = Math.abs(buf[i]);
      if (a > peak) peak = a;
    }
    return peak;
  }

  dispose(): void {
    for (const rt of this.rts) rt.dispose();
    this.rtU0.dispose();
    this.bathyTexture.dispose();
    this.lfMaterial.dispose();
    this.v2Material.dispose();
    this.injectMaterial.dispose();
    this.fillMaterial.dispose();
    this.copyMaterial.dispose();
  }
}
