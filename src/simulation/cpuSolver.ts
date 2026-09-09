/**
 * CPU 镜像求解器(纯 TS,无 WebGL 依赖)。
 * 与 GPU STEP_FRAG(LF)/ STEP_FRAG_V2(MUSCL-HLL-RK2)语义同构,
 * 供 CI 基准测试(衰减、收敛阶、守恒、长时稳定)。
 * 网格行主序(自南向北);状态:eta(m)、hu/hv(深度积分通量 m²/s)。
 */
import { GRAVITY } from '../config';

export type Scheme = 'lf' | 'v2';

export interface CpuSolverOptions {
  nx: number;
  ny: number;
  dx: number;
  dy: number;
  /** 海床高程(m,负值在海面以下) */
  bed: Float64Array;
  dt: number;
  globe?: boolean;
  scheme?: Scheme;
  /** 每步动量阻尼系数(对应 GPU uDamping) */
  damping?: number;
  /** 边界海绵宽度(域比例,plane 默认 0.06、globe 0.03;0 关闭,供收敛基准) */
  spongeWidth?: number;
  /** x 方向周期边界(plane 网格下仅供收敛基准使用) */
  periodicX?: boolean;
}

function minmod(a: number, b: number): number {
  if (a * b <= 0) return 0;
  return Math.abs(a) < Math.abs(b) ? a : b;
}

function smoothstepEdge(edge: number, width: number): number {
  const t = Math.min(Math.max(edge / width, 0), 1);
  return t * t * (3 - 2 * t);
}

export class CpuSolver {
  readonly nx: number;
  readonly ny: number;
  readonly dx: number;
  readonly dy: number;
  readonly dt: number;
  readonly globe: boolean;
  private readonly periodicX: boolean;
  scheme: Scheme;
  damping: number;
  eta: Float64Array;
  hu: Float64Array;
  hv: Float64Array;

  private readonly H: Float64Array;
  /** 逐单元经向有效格距(球面极地保护,与着色器 cflFloor 同构) */
  private readonly dxEff: Float64Array;
  private readonly sponge: Float64Array;
  private readonly wet: Float64Array;
  private tEta: Float64Array;
  private tHu: Float64Array;
  private tHv: Float64Array;
  private readonly lEta: Float64Array;
  private readonly lHu: Float64Array;
  private readonly lHv: Float64Array;
  /** 界面通量暂存(x / y 方向各三分量) */
  private fx = [0, 0, 0];
  private fy = [0, 0, 0];

  constructor(o: CpuSolverOptions) {
    this.nx = o.nx;
    this.ny = o.ny;
    this.dx = o.dx;
    this.dy = o.dy;
    this.dt = o.dt;
    this.globe = o.globe ?? false;
    this.periodicX = o.periodicX ?? false;
    this.scheme = o.scheme ?? 'v2';
    this.damping = o.damping ?? 0.9998;
    const n = o.nx * o.ny;
    this.H = new Float64Array(n);
    this.dxEff = new Float64Array(n);
    this.sponge = new Float64Array(n);
    this.wet = new Float64Array(n);
    this.eta = new Float64Array(n);
    this.hu = new Float64Array(n);
    this.hv = new Float64Array(n);
    this.tEta = new Float64Array(n);
    this.tHu = new Float64Array(n);
    this.tHv = new Float64Array(n);
    this.lEta = new Float64Array(n);
    this.lHu = new Float64Array(n);
    this.lHv = new Float64Array(n);
    const cosDeg = Math.PI / 180;
    const width = o.spongeWidth ?? (this.globe ? 0.03 : 0.06);
    for (let j = 0; j < o.ny; j++) {
      const v = (j + 0.5) / o.ny;
      const lat = (v - 0.5) * 168.0;
      const cosLat = Math.cos(Math.max(Math.abs(lat), 5.0) * cosDeg);
      for (let i = 0; i < o.nx; i++) {
        const k = j * o.nx + i;
        this.H[k] = Math.max(-o.bed[k], 0);
        const floor = this.dt * Math.sqrt(GRAVITY * this.H[k]) * 2.2 + 100.0;
        this.dxEff[k] = this.globe ? Math.max(o.dx * cosLat, floor) : o.dx;
        const u = (i + 0.5) / o.nx;
        const edge = this.globe
          ? Math.min(v, 1 - v)
          : Math.min(Math.min(u, 1 - u), Math.min(v, 1 - v));
        this.sponge[k] = width > 0 ? smoothstepEdge(edge, width) : 1;
        this.wet[k] = this.H[k] > 1.0 ? 1.0 : 0.0;
      }
    }
  }

  private ix(i: number): number {
    if (this.globe || this.periodicX) return ((i % this.nx) + this.nx) % this.nx;
    return i < 0 ? 0 : i >= this.nx ? this.nx - 1 : i;
  }

  private iy(j: number): number {
    return j < 0 ? 0 : j >= this.ny ? this.ny - 1 : j;
  }

  /** 注入高斯型海底抬升(与 GPU inject 同构) */
  inject(u: number, v: number, ampMeters: number, radiusMeters: number): void {
    const rUv = radiusMeters / (this.dx * this.nx);
    const rr = rUv * rUv;
    for (let j = 0; j < this.ny; j++) {
      for (let i = 0; i < this.nx; i++) {
        let dvx = (i + 0.5) / this.nx - u;
        let dvy = (j + 0.5) / this.ny - v;
        if (this.globe) {
          dvx -= Math.floor(dvx + 0.5);
          dvy *= 0.4667;
        }
        this.eta[j * this.nx + i] +=
          ampMeters * Math.exp(-(dvx * dvx + dvy * dvy) / rr);
      }
    }
  }

  readPeak(): number {
    let p = 0;
    for (let k = 0; k < this.eta.length; k++) {
      const a = Math.abs(this.eta[k]);
      if (a > p) p = a;
    }
    return p;
  }

  /** 总水量 Σ(H+η)(守恒检查用) */
  totalVolume(): number {
    let s = 0;
    for (let k = 0; k < this.eta.length; k++) s += this.H[k] + this.eta[k];
    return s;
  }

  step(): void {
    if (this.scheme === 'lf') this.stepLf();
    else this.stepV2();
  }

  // ---------------------------------------------------------------- LF(镜像旧格式)

  private stepLf(): void {
    const { nx, ny, dy, dt, eta, hu, hv, tEta, tHu, tHv } = this;
    for (let j = 0; j < ny; j++) {
      const rm = this.iy(j - 1) * nx;
      const rp = this.iy(j + 1) * nx;
      for (let i = 0; i < nx; i++) {
        const k = j * nx + i;
        const l = j * nx + this.ix(i - 1);
        const r = j * nx + this.ix(i + 1);
        const d = rm + i;
        const u = rp + i;
        const dxe = this.dxEff[k];
        const Hc = this.H[k];
        const eNew = 0.25 * (eta[l] + eta[r] + eta[d] + eta[u]) -
          dt * ((hu[r] - hu[l]) / (2 * dxe) + (hv[u] - hv[d]) / (2 * dy));
        let uNew = 0.25 * (hu[l] + hu[r] + hu[d] + hu[u]) -
          (dt * GRAVITY * Hc * (eta[r] - eta[l])) / (2 * dxe);
        let vNew = 0.25 * (hv[l] + hv[r] + hv[d] + hv[u]) -
          (dt * GRAVITY * Hc * (eta[u] - eta[d])) / (2 * dy);
        const m = this.damping * this.sponge[k] * this.wet[k];
        uNew *= m;
        vNew *= m;
        tEta[k] = this.wet[k] > 0.5 ? eNew : eta[k];
        tHu[k] = uNew;
        tHv[k] = vNew;
      }
    }
    this.swap();
  }

  private swap(): void {
    let t = this.eta; this.eta = this.tEta; this.tEta = t;
    t = this.hu; this.hu = this.tHu; this.tHu = t;
    t = this.hv; this.hv = this.tHv; this.tHv = t;
  }

  // ---------------------------------------------------------------- V2(MUSCL-HLL-RK2)

  /** x 方向界面 i+1/2 通量 → this.fx(MUSCL 重构 + Rusanov) */
  private fluxX(i: number, j: number): void {
    const { nx, H } = this;
    const row = j * nx;
    const a = row + this.ix(i - 1);
    const b = row + this.ix(i);
    const c = row + this.ix(i + 1);
    const d = row + this.ix(i + 2);
    const e = [this.eta, this.hu, this.hv];
    const sL = [0, 0, 0];
    const sR = [0, 0, 0];
    for (let q = 0; q < 3; q++) {
      const f = e[q];
      sL[q] = minmod(f[b] - f[a], f[c] - f[b]);
      sR[q] = minmod(f[c] - f[b], f[d] - f[c]);
    }
    const HL = H[b];
    const HR = H[c];
    const s = Math.sqrt(GRAVITY * Math.max(HL, HR));
    const eL = this.eta[b] + 0.5 * sL[0];
    const eR = this.eta[c] - 0.5 * sR[0];
    const uL = this.hu[b] + 0.5 * sL[1];
    const uR = this.hu[c] - 0.5 * sR[1];
    const vL = this.hv[b] + 0.5 * sL[2];
    const vR = this.hv[c] - 0.5 * sR[2];
    this.fx[0] = 0.5 * (uL + uR) - 0.5 * s * (eR - eL);
    this.fx[1] = 0.5 * GRAVITY * (HL * eL + HR * eR) - 0.5 * s * (uR - uL);
    this.fx[2] = -0.5 * s * (vR - vL);
  }

  /** y 方向界面 j+1/2 通量 → this.fy */
  private fluxY(i: number, j: number): void {
    const { nx, H } = this;
    const a = this.iy(j - 1) * nx + i;
    const b = this.iy(j) * nx + i;
    const c = this.iy(j + 1) * nx + i;
    const d = this.iy(j + 2) * nx + i;
    const e = [this.eta, this.hu, this.hv];
    const sL = [0, 0, 0];
    const sR = [0, 0, 0];
    for (let q = 0; q < 3; q++) {
      const f = e[q];
      sL[q] = minmod(f[b] - f[a], f[c] - f[b]);
      sR[q] = minmod(f[c] - f[b], f[d] - f[c]);
    }
    const HL = H[b];
    const HR = H[c];
    const s = Math.sqrt(GRAVITY * Math.max(HL, HR));
    const eL = this.eta[b] + 0.5 * sL[0];
    const eR = this.eta[c] - 0.5 * sR[0];
    const uL = this.hu[b] + 0.5 * sL[1];
    const uR = this.hu[c] - 0.5 * sR[1];
    const vL = this.hv[b] + 0.5 * sL[2];
    const vR = this.hv[c] - 0.5 * sR[2];
    this.fy[0] = 0.5 * (vL + vR) - 0.5 * s * (eR - eL);
    this.fy[1] = -0.5 * s * (uR - uL);
    this.fy[2] = 0.5 * GRAVITY * (HL * eL + HR * eR) - 0.5 * s * (vR - vL);
  }

  /** 空间算子 L(U) = -div F → lEta/lHu/lHv */
  private computeL(e: Float64Array, u: Float64Array, v: Float64Array): void {
    // 通量函数读取 this.eta/hu/hv,故先别名交换
    const saveE = this.eta, saveU = this.hu, saveV = this.hv;
    this.eta = e; this.hu = u; this.hv = v;
    const { nx, ny, dy, lEta, lHu, lHv } = this;
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const k = j * nx + i;
        this.fluxX(i - 1, j);
        const xm = [this.fx[0], this.fx[1], this.fx[2]];
        this.fluxX(i, j);
        const xp = this.fx;
        this.fluxY(i, j - 1);
        const ym = [this.fy[0], this.fy[1], this.fy[2]];
        this.fluxY(i, j);
        const yp = this.fy;
        const dxe = this.dxEff[k];
        lEta[k] = -((xp[0] - xm[0]) / dxe + (yp[0] - ym[0]) / dy);
        lHu[k] = -((xp[1] - xm[1]) / dxe + (yp[1] - ym[1]) / dy);
        lHv[k] = -((xp[2] - xm[2]) / dxe + (yp[2] - ym[2]) / dy);
      }
    }
    this.eta = saveE; this.hu = saveU; this.hv = saveV;
  }

  private stepV2(): void {
    const { dt, eta, hu, hv, tEta, tHu, tHv, lEta, lHu, lHv } = this;
    // 阶段 A:U* = U + dt·L(U)
    this.computeL(eta, hu, hv);
    for (let k = 0; k < eta.length; k++) {
      tEta[k] = eta[k] + dt * lEta[k];
      tHu[k] = hu[k] + dt * lHu[k];
      tHv[k] = hv[k] + dt * lHv[k];
    }
    // 阶段 B:U' = ½U + ½(U* + dt·L(U*))
    this.computeL(tEta, tHu, tHv);
    for (let k = 0; k < eta.length; k++) {
      const eC = 0.5 * (eta[k] + tEta[k] + dt * lEta[k]);
      const uC = 0.5 * (hu[k] + tHu[k] + dt * lHu[k]);
      const vC = 0.5 * (hv[k] + tHv[k] + dt * lHv[k]);
      const m = this.damping * this.sponge[k] * this.wet[k];
      eta[k] = this.wet[k] > 0.5 ? eC : eta[k];
      hu[k] = uC * m;
      hv[k] = vC * m;
    }
  }
}
