/**
 * DEV 调试工具:GPU↔CPU 小网格逐步对比(不进 CI)。
 * 访问 /debug.html 自动运行,结果打印在页面与 console。
 */
import * as THREE from 'three';
import { computeStableDt } from './config';
import { CpuSolver } from './simulation/cpuSolver';
import { TsunamiSolver } from './simulation/TsunamiSolver';

const NX = 64;
const NY = 64;
const DX = 2000;
const H0 = 4000;

function log(msg: string): void {
  console.log(msg);
  const pre = document.getElementById('out');
  if (pre) pre.textContent += msg + '\n';
}

function main(): void {
  const renderer = new THREE.WebGLRenderer({ antialias: false });
  renderer.setSize(64, 64);
  document.body.appendChild(renderer.domElement);
  log(`WebGL2=${renderer.capabilities.isWebGL2}`);

  const bed = new Float32Array(NX * NY).fill(-H0);
  const dt = computeStableDt(DX, DX, H0);
  log(`dt=${dt.toFixed(4)}`);

  const gpu = new TsunamiSolver(renderer, bed, NX, NY, DX, DX, false);
  log(`gpu.dt=${gpu.dtSeconds.toFixed(4)}`);
  const cpu = new CpuSolver({
    nx: NX, ny: NY, dx: DX, dy: DX,
    bed: new Float64Array(NX * NY).fill(-H0),
    dt, scheme: 'v2', damping: 0.9998,
  });

  // 相同高斯隆起
  gpu.inject(0.5, 0.5, 1, 20e3);
  cpu.inject(0.5, 0.5, 1, 20e3);
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
  for (const [i, j] of [[32, 32], [33, 32], [30, 32], [32, 33], [20, 32], [44, 32]]) {
    const g = (j * NX + i) * 4;
    const c = j * NX + i;
    log(`L(${i},${j}) gpu=[${diag[g].toFixed(6)}, ${diag[g + 1].toFixed(6)}, ${diag[g + 2].toFixed(6)}] dp=${diag[g + 3].toFixed(4)}`);
    log(`L(${i},${j}) cpu=[${(cpuAny.lEta[c] * 1000).toFixed(6)}, ${cpuAny.lHu[c].toFixed(6)}, ${cpuAny.lHv[c].toFixed(6)}]`);
  }

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
}

main();
