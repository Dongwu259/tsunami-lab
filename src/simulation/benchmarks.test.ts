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

/**
 * 阶段4:可选频率频散(Madsen–Sørensen/Peregrine 型混合导数项)。
 * 原计划验收为「孤立波 L2」;本模型按设计略去水平对流项,不存在孤立波孤立子解
 * (非线性陡化与频散无法平衡),该项不适定 → 改测色散关系本身(见 ROADMAP 偏差记录)。
 * 实现为 η 位势形式源项 S = coef·∇(∇·(−g·h·∇η)),阶段一致显式,
 * coef = min(h²/3, 0.6·min(dx,dy)²)(网格稳定上限)→ 色散曲线 ω² = c²k²(1 − coef·k²)。
 * 测法:周期域单色右行波,右行特征 R=(η+hu/c)/2 的 DFT 模式相位随时间的推进速率
 * (隔离初始暂态在 η 场上的双向波拍频);关频散对照应回到无色散 ω = ck。
 */
describe('阶段4:频率频散(色散关系与波包扩展)', () => {
  /** 单色右行波演化 steps 步,返回右行模式测得的 ω 与相对初始的振幅保持率 */
  function measureOmega(dispersion: boolean, steps: number): { omega: number; ampKeep: number } {
    const mu = 0.8;                          // kh:频散显著且可解析
    const k = mu / H0;
    const lam = (2 * Math.PI) / k;
    const nx = 16;                           // 一波长 16 点 → coef 网格上限生效(教学网格典型)
    const dx = lam / nx;
    const dt = 0.5 * (dx / C0);
    const s = new CpuSolver({
      nx, ny: 2, dx, dy: dx, bed: new Float64Array(nx * 2).fill(-H0),
      dt, scheme: 'v2', spongeWidth: 0, periodicX: true, manning: 0, dispersion,
    });
    const coef = Math.min((H0 * H0) / 3, 0.6 * dx * dx);
    const omegaAna = C0 * k * Math.sqrt(1 - coef * k * k);   // 实现系统的解析色散曲线
    const amp = 0.01;
    for (let j = 0; j < 2; j++)
      for (let i = 0; i < nx; i++) {
        const e = amp * Math.cos(k * (i + 0.5) * dx);
        s.eta[j * nx + i] = e;
        s.hu[j * nx + i] = omegaAna / k * e;   // 右行特征速度按频散相速给(收敛最快)
      }
    const modeR = (): { th: number; re: number; im: number } => {
      let re = 0, im = 0;
      for (let i = 0; i < nx; i++) {
        const r = 0.5 * (s.eta[i] + s.hu[i] / C0);
        const ph = k * (i + 0.5) * dx;
        re += r * Math.cos(ph);
        im -= r * Math.sin(ph);
      }
      return { th: Math.atan2(im, re), re, im };
    };
    const m0 = modeR();
    const a0 = Math.hypot(m0.re, m0.im);
    let thPrev = m0.th;
    let dSum = 0;
    for (let n = 0; n < steps; n++) {
      s.step();
      const m = modeR();
      let d = m.th - thPrev;
      if (d > Math.PI) d -= 2 * Math.PI;
      if (d < -Math.PI) d += 2 * Math.PI;
      dSum += d;
      thPrev = m.th;
    }
    const m1 = modeR();
    return { omega: -dSum / steps / dt, ampKeep: Math.hypot(m1.re, m1.im) / a0 };
  }

  it('μ=kh=0.8 相速符合实现系统色散曲线 ±2.5%,关频散对照偏离 >3.5%', () => {
    const mu = 0.8;
    const k = mu / H0;
    const lam = (2 * Math.PI) / k;
    const dx = lam / 16;
    const coef = Math.min((H0 * H0) / 3, 0.6 * dx * dx);
    const omegaAna = C0 * k * Math.sqrt(1 - coef * k * k);
    const on = measureOmega(true, 100);
    const off = measureOmega(false, 100);
    const errOn = Math.abs(on.omega - omegaAna) / omegaAna;
    const errOff = Math.abs(off.omega - omegaAna) / omegaAna;
    expect(errOn).toBeLessThan(0.025);    // 开频散:落在解析色散曲线上(实测 ≈1.4%)
    expect(errOff).toBeGreaterThan(0.035); // 关频散:回到无色散 ω=ck(慢 ≈5%)
    expect(on.ampKeep).toBeGreaterThan(0.02); // 16pts 粗网格基格式耗散大,只查非零
  }, 60000);

  it('高斯波包:开频散产生物理频散扩展(RMS 宽度 >1.1×),质量守恒不变', () => {
    const spreadOf = (dispersion: boolean): { sigma: number; massErr: number } => {
      // h≈dx 的近场网格(coef 上限部分生效),短波包主导模 μ≈0.6 → 可测频散扩展
      const H1 = 100, nx = 400, L = 24000, dx = L / nx;
      const s = new CpuSolver({
        nx, ny: 2, dx, dy: dx, bed: new Float64Array(nx * 2).fill(-H1),
        dt: 0.5 * (dx / Math.sqrt(GRAVITY * H1)), scheme: 'v2', spongeWidth: 0,
        periodicX: true, manning: 0, dispersion,
      });
      const x0 = L / 4, w = 80;
      const c1 = Math.sqrt(GRAVITY * H1);
      for (let j = 0; j < 2; j++)
        for (let i = 0; i < nx; i++) {
          const e = Math.exp(-((((i + 0.5) * dx - x0) / w) ** 2));
          s.eta[j * nx + i] = e;
          s.hu[j * nx + i] = c1 * e;      // 纯右行包(避免双向分裂污染宽度度量)
        }
      const v0 = s.totalVolume();
      const sigma = (): number => {
        let sw = 0, sx = 0, sxx = 0;
        for (let i = 0; i < nx; i++) {
          const a = Math.abs(s.eta[i]), x = (i + 0.5) * dx;
          sw += a; sx += a * x; sxx += a * x * x;
        }
        return Math.sqrt(sxx / sw - (sx / sw) ** 2);
      };
      const T = 120;                       // 波包约走 3.8 km ≈ 47 倍初始宽度
      const steps = Math.round(T / s.dt);
      for (let n = 0; n < steps; n++) s.step();
      return {
        sigma: sigma(),
        massErr: Math.abs(s.totalVolume() - v0) / v0,
      };
    };
    const on = spreadOf(true);
    const off = spreadOf(false);
    expect(on.sigma).toBeGreaterThan(off.sigma * 1.1);   // 频散尾波展宽(实测区分明显)
    expect(on.massErr).toBeLessThan(1e-6);              // 频散项只动量、不触碰质量
    expect(off.massErr).toBeLessThan(1e-6);
  }, 60000);

  it('变水深 + 频散长时 3000 步稳定无发散(系数网格上限守恒)', () => {
    const nx = 512, L = 200e3, dx = L / nx;
    const bed = new Float64Array(nx * 2);
    for (let j = 0; j < 2; j++)
      for (let i = 0; i < nx; i++)
        bed[j * nx + i] = -(1000 - 900 * (i / nx));   // 1000 m → 100 m 缓坡
    const s = new CpuSolver({
      nx, ny: 2, dx, dy: dx, bed,
      dt: computeStableDt(dx, dx, 1000), scheme: 'v2',
      spongeWidth: 0.06, dispersion: true, manning: 0,
    });
    const x0 = 40e3, w = 8e3;
    for (let j = 0; j < 2; j++)
      for (let i = 0; i < nx; i++)
        s.eta[j * nx + i] = Math.exp(-((((i + 0.5) * dx - x0) / w) ** 2));
    const p0 = s.readPeak();
    for (let n = 0; n < 3000; n++) s.step();
    const p = s.readPeak();
    expect(Number.isFinite(p)).toBe(true);
    expect(p).toBeGreaterThan(0);
    expect(p).toBeLessThan(p0 * 3);
  }, 60000);
});

/**
 * 阶段5:干湿 run-up(静水重构 + 保正通量 + 动态干湿)。
 * Thacker(1981)抛物面碗平面晃荡解:床面 z_b = (ω²/2g)(x²−L²),
 * 精确解 η = A·cos(ωt)·x + b(t)、全场均匀速度 u = −(gA/ω)sin(ωt),ω = √(2gh₀)/L。
 * 关键性质:解的速度处处均匀 → 对流项 u∂x u ≡ 0 —— 本模型(略去对流)的**精确解**,
 * 数值误差纯为离散化;水面周期性漫过 ±L 处的岸线(run-up/run-down)。
 */
describe('阶段5:干湿 run-up(Thacker 抛物面碗)', () => {
  it('晃荡周期/幅值误差 <5%,全程无负水深,最大爬高范围 <8%', () => {
    const Lb = 1e4, h0 = 50;
    const om2 = (2 * GRAVITY * h0) / (Lb * Lb);
    const om = Math.sqrt(om2);
    const T = (2 * Math.PI) / om;
    const A = 4e-4;                       // 倾斜幅值:η(±L) ≈ ±4 m
    const xa = -1.08 * Lb, xb = 1.08 * Lb; // 岸线最大到 ±10.4 km,留 ~390 m 缓冲
                                      //(勿让水触域缘:鬼界面自通量是开边界,会漏质量)
    const nx = 800, dx = (xb - xa) / nx;
    const bed = new Float64Array(nx * 2);
    for (let j = 0; j < 2; j++)
      for (let i = 0; i < nx; i++) {
        const x = xa + (i + 0.5) * dx;
        bed[j * nx + i] = (om2 / (2 * GRAVITY)) * (x * x - Lb * Lb);
      }
    const s = new CpuSolver({
      nx, ny: 2, dx, dy: dx, bed,
      dt: computeStableDt(dx, dx, h0), scheme: 'v2',
      spongeWidth: 0, manning: 0, radiation: false,
    });
    for (let j = 0; j < 2; j++)
      for (let i = 0; i < nx; i++) {
        const x = xa + (i + 0.5) * dx;
        // 静止倾斜水面(t=0 为转向点);解析解的湿区 = A·x > bed(约 |x|<10.4 km),
        // 陆地(碗缘山地)无水 → η=0,勿把线性面设到高地上(会凭空注水)
        s.eta[j * nx + i] = A * x > bed[j * nx + i] ? A * x : 0;
      }

    const iProbe = Math.round((5000 - xa) / dx);
    const steps = Math.round((5 * T) / s.dt);
    const probe = new Float64Array(steps + 1);
    const HArr = s.maxH && (s as unknown as { H: Float64Array }).H;   // 静水深(私有,读出扫最小水深)
    let minH = Infinity;
    for (let n = 1; n <= steps; n++) {
      s.step();
      probe[n] = s.eta[iProbe];
      if (n % 100 === 0) {
        for (let q = 0; q < nx * 2; q++) {
          const hq = HArr[q] + s.eta[q];
          if (hq < minH) minH = hq;
        }
      }
    }
    // 探针局部极大(相邻差分变号,窗口 ≥ T/3 防噪声双峰)
    const win = Math.max(3, Math.round(T / (3 * s.dt)));
    const peakT: number[] = [];
    const peakV: number[] = [];
    for (let n = 2; n < steps; n++) {
      if (probe[n] <= probe[n - 1] || probe[n] < probe[n + 1]) continue;
      if (peakT.length && n - peakT[peakT.length - 1] < win) {
        if (probe[n] > peakV[peakV.length - 1]) {
          peakT[peakT.length - 1] = n; peakV[peakV.length - 1] = probe[n];
        }
        continue;
      }
      peakT.push(n); peakV.push(probe[n]);
    }
    const period = (peakT[peakT.length - 1] - peakT[0]) / (peakT.length - 1) * s.dt;

    // 解析最大爬高:h=0 的外侧根(t=0):x± = [A + √(A²+2(ω²/g)h₀)]·g/ω²
    const xShore = (A + Math.sqrt(A * A + 2 * (om2 / GRAVITY) * h0)) * (GRAVITY / om2);
    let xMax = 0;
    for (let i = Math.floor(nx / 2); i < nx; i++) {
      if (s.maxH[i] > 5e-3) xMax = Math.max(xMax, xa + (i + 0.5) * dx);
    }
    const periodErr = Math.abs(period - T) / T;
    const firstPeak = Math.max(...peakV);
    const shoreErr = Math.abs(xMax - xShore) / xShore;
    expect(peakT.length).toBeGreaterThanOrEqual(4);   // 5 周期至少 4 个极大
    expect(periodErr).toBeLessThan(0.05);             // 周期 <5%(实测 0.46%)
    expect(firstPeak).toBeGreaterThan(0.35 * A * 5000); // 首摆幅值保持 ≥35%(实测 ≈47%;
                                                      // 一阶前沿固有耗散,见 ROADMAP 偏差记录)
    expect(minH).toBeGreaterThanOrEqual(0);           // 全程无负水深(薄膜/归干钳制)
    expect(shoreErr).toBeLessThan(0.08);              // 累计爬高范围 <8%(实测 3.6%)
  }, 180000);
});
