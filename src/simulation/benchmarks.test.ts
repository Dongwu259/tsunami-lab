/**
 * 阶段 1 基准:CPU 镜像求解器的耗散、收敛阶与长时稳定性。
 * GPU↔CPU 一致性为 dev 模式人工工具( CI 无 WebGL ),CI 仅跑本镜像基准。
 *
 * 行波诊断:η = cos(kx)、hu = c·η(c=√(gH))是线性浅水方程的精确行波解,
 * 单向传播、无分裂色散,峰值衰减只来自数值耗散;且解析解已知,可直接量收敛阶。
 */
import { describe, expect, it } from 'vitest';
import { computeStableDt } from '../config';
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
  spongeWidth?: number, periodicX = false
): CpuSolver {
  return new CpuSolver({
    nx, ny, dx, dy, bed: flat(nx * ny),
    dt: computeStableDt(dx, dy, H0), scheme, globe, damping,
    spongeWidth, periodicX,
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
    const errAt = (nx: number, scheme: Scheme): number => {
      const dx = waveLen / nx; // 域恰为一个波长
      // 周期 x + 关海绵:行波环绕不失真,误差只来自格式本身
      const s = make(nx, 2, dx, dx, scheme, false, 1, 0, true);
      travelingWave(s, waveLen);
      const steps = Math.round(T / s.dt);
      for (let n = 0; n < steps; n++) s.step();
      const k = (2 * Math.PI) / waveLen;
      let sum = 0;
      for (let i = 0; i < nx; i++) {
        const exact = Math.cos(k * ((i + 0.5) * dx - C0 * steps * s.dt));
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
