/**
 * DEV 调试工具:GPU↔CPU 小网格逐步对比(不进 CI)。
 * 访问 /debug.html 自动运行,结果打印在页面与 console。
 * 场景:1) 频散关(默认路径)  2) 频散开(uDispersion=1)。
 */
import * as THREE from 'three';
import { computeStableDt } from './config';
import { CpuSolver } from './simulation/cpuSolver';
import { TsunamiSolver } from './simulation/TsunamiSolver';
import {
  GLOBE_TERRAIN_VERT,
  GLOBE_WATER_VERT,
  TERRAIN_FRAG,
  TERRAIN_VERT,
  WATER_FRAG,
  WATER_VERT,
} from './simulation/shaders';

const NX = 64;
const NY = 64;
const DX = 2000;
const H0 = 4000;

function log(msg: string): void {
  console.log(msg);
  const pre = document.getElementById('out');
  if (pre) pre.textContent += msg + '\n';
}

/** 单场景:同构注入 → L 算子诊断 → 20 步 η/hu 剖面对比。
 * shoreline = true 时用斜坡海岸床面 + 左侧入射波(覆盖干湿前沿路径)。 */
function runCompare(
  renderer: THREE.WebGLRenderer,
  dispersion: boolean,
  shoreline = false
): void {
  log(`\n===== 场景:${shoreline ? '岸线 run-up' : `频散 ${dispersion ? '开' : '关'}`} =====`);
  const bed = new Float32Array(NX * NY);
  for (let j = 0; j < NY; j++)
    for (let i = 0; i < NX; i++) {
      // 斜坡:左深 4000 m → 右侧抬升为陆地(x > 0.78 为海岸高地)
      bed[j * NX + i] = shoreline
        ? Math.min(-H0 + (i / NX) * 5200, 800)
        : -H0;
    }
  const dt = computeStableDt(DX, DX, H0);

  const gpu = new TsunamiSolver(renderer, bed, NX, NY, DX, DX, false);
  gpu.uniforms.uDispersion.value = dispersion ? 1 : 0;
  const cpu = new CpuSolver({
    nx: NX, ny: NY, dx: DX, dy: DX,
    bed: Float64Array.from(bed),
    dt, scheme: 'v2', damping: 0.9998, dispersion,
  });

  // 相同高斯隆起(岸线场景放左侧深海,波向右传播爬坡)
  const cu = shoreline ? 0.3 : 0.5;
  gpu.inject(cu, 0.5, 1, 10e3);
  cpu.inject(cu, 0.5, 1, 10e3);
  log(`inject: gpuPeak=${gpu.readPeak().toFixed(6)} cpuPeak=${cpu.readPeak().toFixed(6)}`);

  const anyGpu = gpu as unknown as {
    curRt: THREE.WebGLRenderTarget;
    rtU0: THREE.WebGLRenderTarget;
    uniforms: Record<string, THREE.IUniform>;
    quadScene: THREE.Scene;
    pass: (s: THREE.Scene, t: THREE.WebGLRenderTarget | null) => void;
    bindStepMaterial: () => THREE.ShaderMaterial;
    camera: THREE.Camera;
  };

  const readRt = (rt: THREE.WebGLRenderTarget): Float32Array => {
    const b = new Float32Array(NX * NY * 4);
    renderer.readRenderTargetPixels(rt, 0, 0, NX, NY, b);
    return b;
  };

  // --- L 算子诊断:同一初始场上对比 GPU/CPU 的空间算子 ---
  const cpuAny = cpu as unknown as {
    computeL: (a: Float64Array, b: Float64Array, c: Float64Array) => void;
    lEta: Float64Array; lHu: Float64Array; lHv: Float64Array;
  };
  cpuAny.computeL(cpu.eta, cpu.hu, cpu.hv);
  anyGpu.bindStepMaterial();
  anyGpu.uniforms.uDiag.value = 1;
  anyGpu.uniforms.u0.value = null;
  anyGpu.uniforms.uStage.value = 0;
  anyGpu.pass(anyGpu.quadScene, anyGpu.rtU0);
  const diag = readRt(anyGpu.rtU0);
  anyGpu.uniforms.uDiag.value = 0;
  let maxDiff = 0;
  for (const [i, j] of [[32, 32], [33, 32], [30, 32], [32, 33], [20, 32], [44, 32]]) {
    const g = (j * NX + i) * 4;
    const c = j * NX + i;
    const dHu = Math.abs(diag[g + 1] - cpuAny.lHu[c]);
    if (dHu > maxDiff) maxDiff = dHu;
    log(`L(${i},${j}) gpu=[${diag[g].toFixed(6)}, ${diag[g + 1].toFixed(6)}, ${diag[g + 2].toFixed(6)}]`);
    log(`L(${i},${j}) cpu=[${(cpuAny.lEta[c] * 1000).toFixed(6)}, ${cpuAny.lHu[c].toFixed(6)}, ${cpuAny.lHv[c].toFixed(6)}]`);
  }
  log(`L 算子 hu 最大差 = ${maxDiff.toExponential(2)}(典型 ≤3e-6;频散开时含 S 项,容差相同)`);

  for (let n = 1; n <= 20; n++) {
    // 手动执行 GPU 一步(拆开两阶段以便检查中间级)
    anyGpu.bindStepMaterial();
    anyGpu.uniforms.u0.value = null;
    anyGpu.uniforms.uStage.value = 0;
    anyGpu.pass(anyGpu.quadScene, anyGpu.rtU0);
    anyGpu.uniforms.uState.value = anyGpu.rtU0.texture;
    anyGpu.uniforms.u0.value = anyGpu.curRt.texture;
    anyGpu.uniforms.uStage.value = 1;
    const other = (gpu as unknown as { otherRt: THREE.WebGLRenderTarget }).otherRt;
    anyGpu.pass(anyGpu.quadScene, other);
    (gpu as unknown as { cur: number }).cur =
      1 - (gpu as unknown as { cur: number }).cur;

    cpu.step();
    const gState = readRt(anyGpu.curRt);
    if (n % 5 === 0 || n === 1) {
      const row = Math.floor(NY / 2);
      const prof = (get: (i: number) => number): string => {
        const out: string[] = [];
        for (let i = 20; i <= 44; i += 3) out.push(get(i).toFixed(4));
        return out.join(' ');
      };
      const gEta = prof((i) => gState[(row * NX + i) * 4]);
      const gHu = prof((i) => gState[(row * NX + i) * 4 + 1]);
      const cEta = prof((i) => cpu.eta[row * NX + i]);
      const cHu = prof((i) => cpu.hu[row * NX + i]);
      log(`step ${n} gpuEta: ${gEta}`);
      log(`step ${n} cpuEta: ${cEta}`);
      log(`step ${n} gpuHu : ${gHu}`);
      log(`step ${n} cpuHu : ${cHu}`);
    }
  }
  gpu.dispose();
}

/** 渲染着色器编译冒烟测试:水/地形 × 平面/球面 4 材质编译 + 一次离屏渲染。
 * 捕获 GLSL 语法/链接错误(如 TERRAIN_FRAG 的 uRunup 图层改动)。 */
function smokeRenderShaders(renderer: THREE.WebGLRenderer): void {
  const mk = (vert: string, frag: string, extra: Record<string, THREE.IUniform>): THREE.Mesh =>
    new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.ShaderMaterial({
        vertexShader: vert,
        fragmentShader: frag,
        uniforms: {
          uState: { value: null },
          uBathymetry: { value: null },
          uTexel: { value: new THREE.Vector2(1 / 64, 1 / 64) },
          uWaveScale: { value: 0.045 },
          uDomainKm: { value: new THREE.Vector2(240, 240) },
          uTerrainScale: { value: 0.003 },
          uLandLift: { value: 0.3 },
          uColorScale: { value: 0.5 },
          uOpacity: { value: 0.8 },
          uRunup: { value: null },
          uRunupOn: { value: 1 },
          uSeisOn: { value: 0 },
          uSeisT: { value: 0 },
          uEpi: { value: new THREE.Vector2(0.5, 0.5) },
          uEpiGlobe: { value: 0 },
          uSeisDomKm: { value: new THREE.Vector2(240, 240) },
          ...extra,
        },
      })
    );
  const scene = new THREE.Scene();
  const cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  for (const [name, vert, frag] of [
    ['水面(平面)', WATER_VERT, WATER_FRAG],
    ['地形(平面,含淹没图层)', TERRAIN_VERT, TERRAIN_FRAG],
    ['水面(球面)', GLOBE_WATER_VERT, WATER_FRAG],
    ['地形(球面,含淹没图层)', GLOBE_TERRAIN_VERT, TERRAIN_FRAG],
  ] as [string, string, string][]) {
    scene.add(mk(vert, frag, {}));
    renderer.render(scene, cam);
    log(`着色器编译+离屏渲染 OK:${name}`);
    scene.clear();
  }
}

function main(): void {
  const renderer = new THREE.WebGLRenderer({ antialias: false });
  renderer.setSize(64, 64);
  document.body.appendChild(renderer.domElement);
  log(`WebGL2=${renderer.capabilities.isWebGL2}`);
  smokeRenderShaders(renderer);

  runCompare(renderer, false);
  runCompare(renderer, true);
  runCompare(renderer, false, true);
}

main();
