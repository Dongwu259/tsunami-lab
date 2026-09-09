/**
 * 阶段 1 基准:CPU 镜像求解器的耗散、收敛阶与长时稳定性。
 * GPU↔CPU 一致性为 dev 模式人工工具( CI 无 WebGL ),CI 仅跑本镜像基准。
 *
 * 行波诊断:η = cos(kx)、hu = c·η(c=√(gH))是线性浅水方程的精确行波解,
 * 单向传播、无分裂色散,峰值衰减只来自数值耗散;且解析解已知,可直接量收敛阶。
 */
import { describe, expect, it } from 'vitest';
import { computeStableDt, GRAVITY } from '../config';
import { CpuSolver, Scheme } from './cpuSolver';

const H0 = 4000;
const C0 = Math.sqrt(9.81 * H0);
const flat = (n: number): Float64Array => new Float64Array(n).fill(-H0);

/** 单向行波初值(沿 y 不变),波长 waveLen 需整除域长以兼容经向周期 */
function travelingWave(s: CpuSolver, waveLen: number, amp = 1): void {
  const k = (2 * Math.PI) / waveLen;
  for (let j = 0; j < s.ny; j++) {
    for (let i = 0; i < s.nx; i++) {
      const x = (i + 0.5) * s.dx;
      const e = amp * Math.cos(k * x);
      s.eta[j * s.nx + i] = e;
      s.hu[j * s.nx + i] = C0 * e;
    }
  }
}

function make(
  nx: number, ny: number, dx: number, dy: number,
  scheme: Scheme, globe = false, damping = 1,
  spongeWidth?: number, periodicX = false, manning = 0
): CpuSolver {
  return new CpuSolver({
    nx, ny, dx, dy, bed: flat(nx * ny),
    dt: computeStableDt(dx, dy, H0), scheme, globe, damping,
    spongeWidth, periodicX, manning,
  });
}

describe('阶段1:二阶格式耗散与收敛', () => {
  it('V2 行波 1000 步幅值衰减 <5%,且显著优于 LF', () => {
    const decayOf = (scheme: Scheme): number => {
      const L = 1000e3; // 5 个波长,1000 步行程约 99km 不触边界海绵
      const s = make(2048, 2, L / 2048, L / 2048, scheme);
      travelingWave(s, 200e3);
      for (let n = 0; n < 500; n++) s.step();
      const p500 = s.readPeak();
      for (let n = 0; n < 500; n++) s.step();
      return 1 - s.readPeak() / p500;
    };
    const dV2 = decayOf('v2');
    const dLf = decayOf('lf');
    expect(dV2).toBeLessThan(0.05);
    expect(dLf).toBeGreaterThan(dV2 * 3);
  }, 60000);

  it('V2 对解析行波二阶收敛(L2 阶 ≥1.5),LF 仅一阶', () => {
    const T = 200;
    const waveLen = 200e3;
    // 小振幅隔离非线性物理误差底(η/H≪量),量纯数值收敛阶;exact 同比例缩放
    const amp = 0.01;
    const errAt = (nx: number, scheme: Scheme): number => {
      const dx = waveLen / nx; // 域恰为一个波长
      // 周期 x + 关海绵:行波环绕不失真,误差只来自格式本身
      const s = make(nx, 2, dx, dx, scheme, false, 1, 0, true);
      travelingWave(s, waveLen, amp);
      const steps = Math.round(T / s.dt);
      for (let n = 0; n < steps; n++) s.step();
      const k = (2 * Math.PI) / waveLen;
      let sum = 0;
      for (let i = 0; i < nx; i++) {
        const exact = amp * Math.cos(k * ((i + 0.5) * dx - C0 * steps * s.dt));
        const d = s.eta[i] - exact;
        sum += d * d;
      }
      return Math.sqrt(sum / nx);
    };
    const orderOf = (scheme: Scheme): number => {
      const e256 = errAt(256, scheme);
      const e512 = errAt(512, scheme);
      return Math.log2(e256 / e512);
    };
    // 实测 v2≈1.65(色散主导二阶+耗散三阶混合),lf≈1.0;阈值取 1.5/1.3 区分两格式
    expect(orderOf('v2')).toBeGreaterThan(1.5);
    expect(orderOf('lf')).toBeLessThan(1.3);
  }, 60000);

  it('球面度量长时 1500 步无数值爆炸(极地保护守门)', () => {
    const s = new CpuSolver({
      nx: 96, ny: 48, dx: 40000, dy: 389000,
      bed: flat(96 * 48),
      dt: computeStableDt(40000, 389000, H0),
      scheme: 'v2', globe: true, damping: 0.9998,
    });
    s.inject(0.5, 0.5, 1, 600e3);
    const p0 = s.readPeak();
    for (let n = 0; n < 1500; n++) s.step();
    const p = s.readPeak();
    expect(Number.isFinite(p)).toBe(true);
    expect(p).toBeGreaterThan(0);
    expect(p).toBeLessThan(p0 * 1.5);
  }, 60000);
});

/**
 * 阶段2:非线性总水深 + 井平衡源项 + 隐式曼宁摩擦 + 干湿统一。
 * 三项物理保证:静水平衡(井平衡)、质量守恒(干床置零)、Green 浅水放大。
 */
describe('阶段2:非线性水深 + 曼宁摩擦 + 干湿统一', () => {
  /** 静水平衡(lake-at-rest):变地形上平坦海面必须保持静止。
   * 井平衡源项 g·η·∂h/∂x 精确抵消 g·h·η 通量多出的地形项,η=0 时严格不动。 */
  it('变地形静水平衡:平坦海面 100 步不产生伪波', () => {
    const nx = 200, ny = 2, dx = 2000;
    const bed = new Float64Array(nx * ny);
    for (let j = 0; j < ny; j++)
      for (let i = 0; i < nx; i++) bed[j * nx + i] = -(4000 - 3800 * (i / nx));
    const s = new CpuSolver({
      nx, ny, dx, dy: dx, bed,
      dt: computeStableDt(dx, dx, 4000), scheme: 'v2', spongeWidth: 0, manning: 0,
    });
    for (let n = 0; n < 100; n++) s.step();
    let maxEta = 0;
    for (let k = 0; k < s.eta.length; k++) maxEta = Math.max(maxEta, Math.abs(s.eta[k]));
    expect(maxEta).toBeLessThan(1e-6);   // float64 下实测恒为 0(精确井平衡)
  }, 30000);

  /** 质量守恒:非线性总水深通量 + 干床置零;曼宁摩擦只动量、不触碰质量。
   * 周期域 1 小时模拟,Σ(H+η) 漂移到机器精度。 */
  it('周期域 1 小时质量守恒 |Σh−Σh₀|/Σh₀ < 1e-4', () => {
    const nx = 512, ny = 2, L = 1000e3, dx = L / nx;
    const s = new CpuSolver({
      nx, ny, dx, dy: dx, bed: flat(nx * ny),
      dt: computeStableDt(dx, dx, H0), scheme: 'v2',
      spongeWidth: 0, manning: 0.025, periodicX: true,
    });
    const x0 = 500e3, w = 40e3;
    for (let j = 0; j < ny; j++)
      for (let i = 0; i < nx; i++) {
        const x = (i + 0.5) * dx;
        s.eta[j * nx + i] = Math.exp(-(((x - x0) / w) ** 2));
      }
    const v0 = s.totalVolume();
    const steps = Math.round(3600 / s.dt);
    for (let n = 0; n < steps; n++) s.step();
    expect(Math.abs(s.totalVolume() - v0) / Math.abs(v0)).toBeLessThan(1e-4);
  }, 30000);

  /** Green 定律浅水放大:绝热缓变坡(大 tanh 尺度 → 反射最小)下,
   * 深→浅峰值放大比 ≈ (H_deep/H_shallow)^(1/4)。残差为坡面反射 + 数值耗散(实测 −4.5%)。 */
  it('Green 定律浅水放大比符合 (H₁/H₂)^¼ ±5%', () => {
    const dx = 1000, nx = 1600, xcKm = 800, Hd = 2000, Hs = 250, LsKm = 650;
    const HAt = (i: number): number => {
      const x = (i * dx) / 1000;
      return Hs + (Hd - Hs) * 0.5 * (1 - Math.tanh((x - xcKm) / LsKm));
    };
    const bed = new Float64Array(nx * 2);
    for (let j = 0; j < 2; j++)
      for (let i = 0; i < nx; i++) bed[j * nx + i] = -HAt(i);
    const s = new CpuSolver({
      nx, ny: 2, dx, dy: dx, bed,
      dt: computeStableDt(dx, dx, Hd), scheme: 'v2', spongeWidth: 0.06, manning: 0,
    });
    const x0 = 180e3, w = 55e3, amp = 1;
    for (let j = 0; j < 2; j++)
      for (let i = 0; i < nx; i++) {
        const x = (i + 0.5) * dx;
        const e = amp * Math.exp(-(((x - x0) / w) ** 2));
        s.eta[j * nx + i] = e;
        s.hu[j * nx + i] = Math.sqrt(GRAVITY * HAt(i)) * e;
      }
    const iDeep = Math.round(250e3 / dx), iShal = Math.round(1350e3 / dx);
    let aDeep = 0, aShal = 0;
    const steps = Math.round(13000 / s.dt);
    for (let n = 0; n < steps; n++) {
      s.step();
      aDeep = Math.max(aDeep, Math.abs(s.eta[iDeep]));
      aShal = Math.max(aShal, Math.abs(s.eta[iShal]));
    }
    const ideal = Math.pow(HAt(iDeep) / HAt(iShal), 0.25);
    const meas = aShal / aDeep;
    expect(Math.abs(meas - ideal) / ideal).toBeLessThan(0.05);
  }, 60000);
});
