import * as THREE from 'three';
import { GPUComputationRenderer, Variable } from 'three/examples/jsm/misc/GPUComputationRenderer.js';
import { GRAVITY, SIM_DT } from '../config';
import { STEP_FRAG } from './shaders';

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


/**
 * GPU 海啸求解器。
 * 使用 GPUComputationRenderer 在浮点纹理上以 ping-pong 方式
 * 推进线性浅水方程,状态编码为 RGBA:r=η, g=Hu, b=Hv。
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
  /** 时间步长(s) */
  readonly dtSeconds: number;

  private gpgpu: GPUComputationRenderer;
  private stateVar: Variable;
  private renderer: THREE.WebGLRenderer;
  private fillMaterial: THREE.ShaderMaterial;
  private fillScene = new THREE.Scene();
  private copyScene = new THREE.Scene();
  private copyMaterial: THREE.ShaderMaterial;
  private fillCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

  /** 求解器 uniform(供外部调整阻尼等) */
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
    dtSeconds = SIM_DT
  ) {
    this.renderer = renderer;
    this.sizeX = sizeX;
    this.sizeY = sizeY;
    this.dxM = dxM;
    this.dyM = dyM;
    this.globeMode = globeMode;
    this.dtSeconds = dtSeconds;

    const gpgpu = new GPUComputationRenderer(sizeX, sizeY, renderer);
    if (!renderer.capabilities.isWebGL2) {
      gpgpu.setDataType(THREE.HalfFloatType);
    }
    this.gpgpu = gpgpu;

    // --- 海床纹理 ---
    this.bathyTexture = gpgpu.createTexture();
    const bData = this.bathyTexture.image.data as unknown as Float32Array;
    for (let i = 0; i < bathymetry.length; i++) {
      bData[i * 4] = bathymetry[i];
      bData[i * 4 + 3] = 1;
    }
    this.bathyTexture.minFilter = THREE.NearestFilter;
    this.bathyTexture.magFilter = THREE.NearestFilter;
    this.bathyTexture.needsUpdate = true;

    // --- 状态纹理(初始全零,平静海面) ---
    const stateTex = gpgpu.createTexture();
    stateTex.minFilter = THREE.NearestFilter;
    stateTex.magFilter = THREE.NearestFilter;

    // 常数填充用材质(重置海面时复用)
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

    // --- 计算变量 ---
    const v = gpgpu.addVariable('uState', STEP_FRAG, stateTex);
    v.wrapS = THREE.ClampToEdgeWrapping;
    v.wrapT = THREE.ClampToEdgeWrapping;

    const u = v.material.uniforms;
    u.uBathymetry = { value: this.bathyTexture };
    u.uTexel = { value: new THREE.Vector2(1 / sizeX, 1 / sizeY) };
    u.uDx = { value: dxM };
    u.uDy = { value: dyM };
    u.uDt = { value: dtSeconds };
    u.uG = { value: GRAVITY };
    u.uDamping = { value: 0.9998 };
    u.uInject = { value: 0 };
    u.uSource = { value: new THREE.Vector3(0.3, 0.5, 0.05) };
    u.uSourceAmp = { value: 0 };
    u.uGlobeMode = { value: globeMode ? 1 : 0 };
    this.uniforms = u;

    // 球面模式:经向周期环绕,波可跨越 180° 经线
    if (globeMode) {
      v.wrapS = THREE.RepeatWrapping;
    }

    gpgpu.setVariableDependencies(v, [v]);

    const error = gpgpu.init();
    if (error) {
      throw new Error('GPU 计算初始化失败:' + error);
    }
    this.stateVar = v;
    this.fillVariable(v, 0);
  }

  /** 用常数填充变量的双缓冲纹理(自行管理渲染目标,避免 varying 匹配问题) */
  private fillVariable(v: Variable, value: number): void {
    this.fillMaterial.uniforms.uValue.value = value;
    const prevTarget = this.renderer.getRenderTarget();
    this.renderer.setRenderTarget(v.renderTargets[0]);
    this.renderer.render(this.fillScene, this.fillCamera);
    this.renderer.setRenderTarget(v.renderTargets[1]);
    this.renderer.render(this.fillScene, this.fillCamera);
    this.renderer.setRenderTarget(prevTarget);
  }

  /** 最新状态纹理(供渲染材质采样) */
  get stateTexture(): THREE.Texture {
    return this.gpgpu.getCurrentRenderTarget(this.stateVar).texture;
  }

  /** 推进 substeps 个子步 */
  step(substeps: number): void {
    for (let i = 0; i < substeps; i++) {
      this.gpgpu.compute();
    }
  }

  /**
   * 注入一次海底地震(高斯型海底抬升)。
   * @param u,v      震中 uv 坐标(0–1)
   * @param ampMeters 初始波幅(m)
   * @param radiusMeters 破裂半径(m)
   */
  inject(u: number, v: number, ampMeters: number, radiusMeters: number): void {
    this.uniforms.uInject.value = 1;
    (this.uniforms.uSource.value as THREE.Vector3).set(
      u,
      v,
      radiusMeters / (this.dxM * this.sizeX)
    );
    this.uniforms.uSourceAmp.value = ampMeters;
    this.gpgpu.compute();
    this.uniforms.uInject.value = 0;
  }

  /** 重置为平静海面 */
  reset(): void {
    this.fillVariable(this.stateVar, 0);
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

  /** 公共实现:把 η 场写入双缓冲(替换或叠加) */
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
    const prevTarget = this.renderer.getRenderTarget();
    if (!additive) {
      for (const rt of this.stateVar.renderTargets) {
        this.renderer.setRenderTarget(rt);
        this.renderer.render(this.copyScene, this.fillCamera);
      }
    } else {
      // 叠加模式:避免自读自写,先写非当前缓冲,再反向回写
      const rts = this.stateVar.renderTargets;
      const cur = this.gpgpu.getCurrentRenderTarget(this.stateVar);
      const curIdx = rts[0] === cur ? 0 : 1;
      const otherIdx = 1 - curIdx;
      this.copyMaterial.uniforms.uState.value = cur.texture;
      this.renderer.setRenderTarget(rts[otherIdx]);
      this.renderer.render(this.copyScene, this.fillCamera);
      this.copyMaterial.uniforms.uState.value = rts[otherIdx].texture;
      this.renderer.setRenderTarget(rts[curIdx]);
      this.renderer.render(this.copyScene, this.fillCamera);
    }
    this.renderer.setRenderTarget(prevTarget);
    tex.dispose();
  }

  /** 回读全场,返回峰值 |η|(m)。仅在低频统计时调用。
   * @param out 可选复用缓冲区(尺寸需匹配),避免每秒重分配 */
  readPeak(out?: Float32Array): number {
    const rt = this.gpgpu.getCurrentRenderTarget(this.stateVar);
    const need = this.sizeX * this.sizeY * 4;
    const buf = out && out.length === need ? out : new Float32Array(need);
    this.renderer.readRenderTargetPixels(rt, 0, 0, this.sizeX, this.sizeY, buf);
    let peak = 0;
    for (let i = 0; i < buf.length; i += 4) {
      const a = Math.abs(buf[i]);
      if (a > peak) peak = a;
    }
    return peak;
  }

  dispose(): void {
    this.gpgpu.dispose();
    this.bathyTexture.dispose();
  }
}
