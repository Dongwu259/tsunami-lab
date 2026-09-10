/**
 * 阶段 1 基准:CPU 镜像求解器的耗散、收敛阶与长时稳定性。
 * GPU↔CPU 一致性为 dev 模式人工工具( CI 无 WebGL ),CI 仅跑本镜像基准。
 *
 * 行波诊断:η = cos(kx)、hu = c·η(c=√(gH))是线性浅水方程的精确行波解,
 * 单向传播、无分裂色散,峰值衰减只来自数值耗散;且解析解已知,可直接量收敛阶。
 */
import { describe, expect, it } from 'vitest';
import { computeStableDt, GRAVITY, GLOBE_LAT_SPAN, reducedGridMinFactor } from '../config';
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
      const L = 1000e3; // 5 个波长整除域长;周期 x + 关海绵 → 只量格式内蕴耗散(隔离边界效应)
      const s = make(2048, 2, L / 2048, L / 2048, scheme, false, 1, 0, true);
      travelingWave(s, 200e3);
      for (let n = 0; n < 500; n++) s.step();
      const p500 = s.readPeak();
      for (let n = 0; n < 500; n++) s.step();
      return 1 - s.readPeak() / p500;
    };
    const dV2 = decayOf('v2'); // 实测 ≈0.0023(二阶低耗散)
    const dLf = decayOf('lf'); // 实测 ≈0.0146(一阶,约 6× V2)
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
      dt: computeStableDt(40000 * reducedGridMinFactor(GLOBE_LAT_SPAN, 48), 389000, H0),
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

/**
 * 阶段3:辐射边界。plane 四边由动量海绵改为特征辐射 BC(边界单元投影到出射特征、
 * 令入射特征为零)+ 2 单元薄海绵兜底。右行脉冲撞右边界,探测器特征分解量反射能量。
 */
describe('阶段3:辐射边界反射', () => {
  /** 右行高斯脉冲撞 plane 右边界,返回探测器处能量反射比 R=(反射幅/入射幅)²。 */
  function reflectivity(radiation: boolean, spongeWidth?: number): number {
    const nx = 512, ny = 2, L = 1000e3, dx = L / nx;
    const s = new CpuSolver({
      nx, ny, dx, dy: dx, bed: flat(nx * ny),
      dt: computeStableDt(dx, dx, H0), scheme: 'v2', manning: 0,
      damping: 1, radiation, spongeWidth,
    });
    const x0 = 250e3, w = 40e3;
    for (let j = 0; j < ny; j++)
      for (let i = 0; i < nx; i++) {
        const e = Math.exp(-((((i + 0.5) * dx - x0) / w) ** 2));
        s.eta[j * nx + i] = e;
        s.hu[j * nx + i] = C0 * e; // 纯右行:hu = c·η
      }
    const iDet = Math.round(500e3 / dx);
    let inc = 0, refl = 0;
    const steps = Math.round(9000 / s.dt);
    for (let n = 0; n < steps; n++) {
      s.step();
      const e = s.eta[iDet], hu = s.hu[iDet];
      inc = Math.max(inc, Math.abs(0.5 * (e + hu / C0)));   // 右行(入射)特征
      refl = Math.max(refl, Math.abs(0.5 * (e - hu / C0))); // 左行(反射)特征
    }
    return (refl / inc) ** 2;
  }

  it('辐射边界反射能量 <2%,且远优于旧动量海绵', () => {
    const rRad = reflectivity(true);           // 辐射 BC + 2 单元薄海绵:实测 ≈0.010
    const rSponge = reflectivity(false, 0.06); // 旧宽动量海绵:实测 ≈0.40
    expect(rRad).toBeLessThan(0.02);
    expect(rRad).toBeLessThan(rSponge / 5);
  }, 60000);
});

/**
 * 阶段3:缩减纬网(reduced-latitude grid)。高纬经向格距 dx·cosφ 收缩会迫使 dt→0;
 * 旧 cflFloor hack 把 dxEff 抬到 ~3.7× 真实格距 → 74° 波速失真 ~73%(严重偏慢)。
 * 缩减纬网按 |φ| 每 k 列合并(stride-k 采样,>74°→2,>80°→4),dxEff=k·dx·cosφ 有限,
 * 波速恢复正确。测法:y-均匀东行脉冲(初始无 y 梯度)+ 早期窗口质心速度 → 物理波速,
 * 隔离球面剪切耦合(不同纬度角速度不同,随时间累积)对长时间测量的污染。
 */
describe('阶段3:缩减纬网极地波速', () => {
  const KM_LAT = 111.32;
  /** latTarget 行的东行波物理速度相对 c=√(gH0) 的相对误差 */
  function polarSpeedErr(latTarget: number, T = 1500): number {
    const nx = 720, ny = 180;
    const dx = (360 * KM_LAT * 1000) / nx;
    const dy = (GLOBE_LAT_SPAN * KM_LAT * 1000) / ny;
    const dt = computeStableDt(dx * reducedGridMinFactor(GLOBE_LAT_SPAN, ny), dy, H0);
    const s = new CpuSolver({
      nx, ny, dx, dy, bed: flat(nx * ny), dt, globe: true, scheme: 'v2',
      damping: 1, manning: 0, spongeWidth: 0,
    });
    let j0 = 0, best = 1e9;
    for (let j = 0; j < ny; j++) {
      const lat = ((j + 0.5) / ny - 0.5) * GLOBE_LAT_SPAN;
      if (Math.abs(lat - latTarget) < best) { best = Math.abs(lat - latTarget); j0 = j; }
    }
    const lat0 = ((j0 + 0.5) / ny - 0.5) * GLOBE_LAT_SPAN;
    const i0 = Math.round(nx * 0.2), wx = 12;
    // y-均匀注入(所有行相同 x 高斯 + 纯东行 hu=C0·η)→ 初始无 y 梯度,逐行独立 1D 传播
    for (let j = 0; j < ny; j++)
      for (let i = 0; i < nx; i++) {
        const e = Math.exp(-(((i - i0) / wx) ** 2));
        s.eta[j * nx + i] = e;
        s.hu[j * nx + i] = C0 * e;
      }
    const perCol = dx * Math.cos(Math.max(Math.abs(lat0), 5) * Math.PI / 180);
    const centroid = (): number => {
      let sw = 0, swi = 0;
      for (let i = 0; i < nx; i++) {
        const w = Math.max(s.eta[j0 * nx + i], 0);
        sw += w; swi += w * i;
      }
      return swi / sw;
    };
    const c0 = centroid();
    const steps = Math.round(T / dt);
    for (let n = 0; n < steps; n++) s.step();
    let dCol = centroid() - c0;
    if (dCol < -nx / 2) dCol += nx;
    const meas = dCol * perCol / (steps * dt);
    return Math.abs(meas - C0) / C0;
  }

  it('赤道基准:格式波速精确(误差 <1%)', () => {
    expect(polarSpeedErr(0)).toBeLessThan(0.01);   // 实测 ≈0.00%(无剪切,验证格式内禀波速=c)
  }, 60000);

  it('74° 缩减纬网波速误差 <3%(旧 cflFloor 约 73%)', () => {
    expect(polarSpeedErr(74)).toBeLessThan(0.03);  // 实测 ≈1%(k=2 stride 采样,早期窗口)
  }, 60000);
});
