/**
 * CPU 镜像求解器(纯 TS,无 WebGL 依赖)。
 * 与 GPU STEP_FRAG(LF)/ STEP_FRAG_V2(MUSCL-HLL-RK2)语义同构,
 * 供 CI 基准测试(衰减、收敛阶、守恒、长时稳定)。
 * 网格行主序(自南向北);状态:eta(m)、hu/hv(深度积分通量 m²/s)。
 */
import { GRAVITY, H_MIN, MANNING_N, polarStride } from '../config';

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
  /** 曼宁摩擦系数 n(仅 v2;0 关闭,默认 MANNING_N) */
  manning?: number;
  /** 干单元阈值(m):总水深 h=H+η < hMin 视干(默认 H_MIN) */
  hMin?: number;
  /** 辐射边界(单向波 η 外推,plane 默认开;globe/periodicX 该向自动关闭) */
  radiation?: boolean;
}

function minmod(a: number, b: number): number {
  if (a * b <= 0) return 0;
  return Math.abs(a) < Math.abs(b) ? a : b;
}

function smoothstepEdge(edge: number, width: number): number {
  const t = Math.min(Math.max(edge / width, 0), 1);
  return t * t * (3 - 2 * t);
}

/** 2 单元薄海绵兜底因子:辐射边界单元(d=0)不阻尼(交由辐射 BC 精确透射),
 * 内侧 d=1,2 温和动量阻尼(带 floor 避免刚性壁反射),吸收斜入射等残余。 */
function thinSpongeFactor(d: number): number {
  if (d <= 0 || d >= 3) return 1;
  return 1 - 0.08 * (3 - d) / 2;
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
  /** 曼宁摩擦系数 n(仅 v2;lf 仍用 damping) */
  manning: number;
  /** 干单元阈值(m) */
  hMin: number;
  eta: Float64Array;
  hu: Float64Array;
  hv: Float64Array;

  private readonly H: Float64Array;
  /** 逐单元经向有效格距(缩减纬网 dxEff=k·dx·cosφ,与着色器同构) */
  private readonly dxEff: Float64Array;
  /** 逐纬向行缩减纬网 stride k(|φ|>74°→2,>80°→4;plane 全 1) */
  private readonly kLat: Int32Array;
  private readonly sponge: Float64Array;
  private readonly wet: Float64Array;
  /** 辐射边界:开关与逐单元内向邻居索引(-1 表示非辐射边界单元) */
  private readonly radiation: boolean;
  private readonly radXnb: Int32Array;
  private readonly radYnb: Int32Array;
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
    this.manning = o.manning ?? MANNING_N;
    this.hMin = o.hMin ?? H_MIN;
    this.radiation = o.radiation ?? !(o.globe ?? false);
    const n = o.nx * o.ny;
    this.H = new Float64Array(n);
    this.dxEff = new Float64Array(n);
    this.kLat = new Int32Array(o.ny).fill(1);
    this.sponge = new Float64Array(n);
    this.wet = new Float64Array(n);
    this.radXnb = new Int32Array(n).fill(-1);
    this.radYnb = new Int32Array(n).fill(-1);
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
    // 辐射边界开启且未显式指定海绵宽度时,改用 2 单元薄海绵兜底(逐方向,守卫小网格)。
    // 仅限 v2:辐射 BC 是二阶格式特性;lf 保留旧宽海绵作教学基线(与 GPU 两着色器分工一致)
    const thinSponge = this.radiation && this.scheme === 'v2' && o.spongeWidth === undefined;
    const radX = !this.globe && !this.periodicX && o.nx > 2;
    const radY = !this.globe && o.ny > 2;
    for (let j = 0; j < o.ny; j++) {
      const v = (j + 0.5) / o.ny;
      const lat = (v - 0.5) * 168.0;
      const cosLat = Math.cos(Math.max(Math.abs(lat), 5.0) * cosDeg);
      // 缩减纬网:高纬经向每 k 列合并(stride-k 采样),dxEff=k·dx·cosφ 保持有限;
      // 去除旧 cflFloor hack → 极地波速恢复正确(74° 误差 <3%),dt 不再被极地拖垮
      const kj = this.globe ? polarStride(lat) : 1;
      this.kLat[j] = kj;
      for (let i = 0; i < o.nx; i++) {
        const k = j * o.nx + i;
        this.H[k] = Math.max(-o.bed[k], 0);
        this.dxEff[k] = this.globe ? kj * o.dx * cosLat : o.dx;
        if (thinSponge) {
          let sp = 1;
          if (radX) sp *= thinSpongeFactor(Math.min(i, o.nx - 1 - i));
          if (radY) sp *= thinSpongeFactor(Math.min(j, o.ny - 1 - j));
          this.sponge[k] = sp;
        } else {
          const u = (i + 0.5) / o.nx;
          const edge = this.globe
            ? Math.min(v, 1 - v)
            : Math.min(Math.min(u, 1 - u), Math.min(v, 1 - v));
          this.sponge[k] = width > 0 ? smoothstepEdge(edge, width) : 1;
        }
        this.wet[k] = this.H[k] > 1.0 ? 1.0 : 0.0;
        // 辐射边界预计算:边界单元记录内向邻居索引(方向决定出射特征;陆地 H=0 在步进中跳过)
        if (radX && i === 0) this.radXnb[k] = k + 1;
        else if (radX && i === o.nx - 1) this.radXnb[k] = k - 1;
        if (radY && j === 0) this.radYnb[k] = k + o.nx;
        else if (radY && j === o.ny - 1) this.radYnb[k] = k - o.nx;
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
      const s = this.kLat[j];   // 缩减纬网 stride(经向邻居 i±s)
      for (let i = 0; i < nx; i++) {
        const k = j * nx + i;
        const l = j * nx + this.ix(i - s);
        const r = j * nx + this.ix(i + s);
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

  /** x 方向界面通量 → this.fx(MUSCL 重构 + Rusanov)。
   * st=缩减纬网 stride:界面位于 cell i 与 i+st 之间,重构取 i-st/i/i+st/i+2st。 */
  private fluxX(i: number, j: number, st: number): void {
    const { nx, H } = this;
    const row = j * nx;
    const a = row + this.ix(i - st);
    const b = row + this.ix(i);
    const c = row + this.ix(i + st);
    const d = row + this.ix(i + 2 * st);
    const e = [this.eta, this.hu, this.hv];
    const sL = [0, 0, 0];
    const sR = [0, 0, 0];
    for (let q = 0; q < 3; q++) {
      const f = e[q];
      sL[q] = minmod(f[b] - f[a], f[c] - f[b]);
      sR[q] = minmod(f[c] - f[b], f[d] - f[c]);
    }
    const eL = this.eta[b] + 0.5 * sL[0];
    const eR = this.eta[c] - 0.5 * sR[0];
    const uL = this.hu[b] + 0.5 * sL[1];
    const uR = this.hu[c] - 0.5 * sR[1];
    const vL = this.hv[b] + 0.5 * sL[2];
    const vR = this.hv[c] - 0.5 * sR[2];
    // 非线性总水深 h=H+η;干床(h<hMin)界面通量置零(与 GPU musclFluxX 同构)
    const hL = H[b] + eL;
    const hR = H[c] + eR;
    if (hL < this.hMin || hR < this.hMin) {
      this.fx[0] = 0; this.fx[1] = 0; this.fx[2] = 0;
      return;
    }
    const s = Math.sqrt(GRAVITY * Math.max(hL, hR));
    this.fx[0] = 0.5 * (uL + uR) - 0.5 * s * (eR - eL);
    this.fx[1] = 0.5 * GRAVITY * (hL * eL + hR * eR) - 0.5 * s * (uR - uL);
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
    const eL = this.eta[b] + 0.5 * sL[0];
    const eR = this.eta[c] - 0.5 * sR[0];
    const uL = this.hu[b] + 0.5 * sL[1];
    const uR = this.hu[c] - 0.5 * sR[1];
    const vL = this.hv[b] + 0.5 * sL[2];
    const vR = this.hv[c] - 0.5 * sR[2];
    // 非线性总水深 h=H+η;干床(h<hMin)界面通量置零(与 GPU musclFluxY 同构)
    const hL = H[b] + eL;
    const hR = H[c] + eR;
    if (hL < this.hMin || hR < this.hMin) {
      this.fy[0] = 0; this.fy[1] = 0; this.fy[2] = 0;
      return;
    }
    const s = Math.sqrt(GRAVITY * Math.max(hL, hR));
    this.fy[0] = 0.5 * (vL + vR) - 0.5 * s * (eR - eL);
    this.fy[1] = -0.5 * s * (uR - uL);
    this.fy[2] = 0.5 * GRAVITY * (hL * eL + hR * eR) - 0.5 * s * (vR - vL);
  }

  /** 空间算子 L(U) = -div F → lEta/lHu/lHv */
  private computeL(e: Float64Array, u: Float64Array, v: Float64Array): void {
    // 通量函数读取 this.eta/hu/hv,故先别名交换
    const saveE = this.eta, saveU = this.hu, saveV = this.hv;
    this.eta = e; this.hu = u; this.hv = v;
    const { nx, ny, dy, lEta, lHu, lHv, H } = this;
    for (let j = 0; j < ny; j++) {
      const s = this.kLat[j];   // 缩减纬网 stride(经向界面/源项均用 i±s)
      for (let i = 0; i < nx; i++) {
        const k = j * nx + i;
        this.fluxX(i - s, j, s);
        const xm = [this.fx[0], this.fx[1], this.fx[2]];
        this.fluxX(i, j, s);
        const xp = this.fx;
        this.fluxY(i, j - 1);
        const ym = [this.fy[0], this.fy[1], this.fy[2]];
        this.fluxY(i, j);
        const yp = this.fy;
        const dxe = this.dxEff[k];
        // 井平衡源项:g·η·∂h/∂x(h=H+η),抵消 g·h·η 通量多出的 -g·η·∂h/∂x,
        // 使动量方程精确回到 -g·h·∂η/∂x(正确 Green 定律浅水放大);η=0 时源项为 0,严格静水平衡
        const im1 = j * nx + this.ix(i - s);
        const ip1 = j * nx + this.ix(i + s);
        const jm1 = this.iy(j - 1) * nx + i;
        const jp1 = this.iy(j + 1) * nx + i;
        const dhdx = ((H[ip1] + e[ip1]) - (H[im1] + e[im1])) / (2 * dxe);
        const dhdy = ((H[jp1] + e[jp1]) - (H[jm1] + e[jm1])) / (2 * dy);
        lEta[k] = -((xp[0] - xm[0]) / dxe + (yp[0] - ym[0]) / dy);
        lHu[k] = -((xp[1] - xm[1]) / dxe + (yp[1] - ym[1]) / dy) + GRAVITY * e[k] * dhdx;
        lHv[k] = -((xp[2] - xm[2]) / dxe + (yp[2] - ym[2]) / dy) + GRAVITY * e[k] * dhdy;
      }
    }
    this.eta = saveE; this.hu = saveU; this.hv = saveV;
  }

  private stepV2(): void {
    const { dt, eta, hu, hv, tEta, tHu, tHv, lEta, lHu, lHv, H, hMin, manning } = this;
    // 阶段 A:U* = U + dt·L(U)
    this.computeL(eta, hu, hv);
    for (let k = 0; k < eta.length; k++) {
      tEta[k] = eta[k] + dt * lEta[k];
      tHu[k] = hu[k] + dt * lHu[k];
      tHv[k] = hv[k] + dt * lHv[k];
    }
    // 阶段 B:U' = ½U + ½(U* + dt·L(U*)),再叠加算子分裂的修改项
    this.computeL(tEta, tHu, tHv);
    for (let k = 0; k < eta.length; k++) {
      // 陆地(静水深 H < hMin)为干单元:η 冻结、动量清零(不参与通量 → 质量守恒)
      if (H[k] < hMin) {
        hu[k] = 0;
        hv[k] = 0;
        continue;
      }
      const eC = 0.5 * (eta[k] + tEta[k] + dt * lEta[k]);
      let uC = 0.5 * (hu[k] + tHu[k] + dt * lHu[k]);
      let vC = 0.5 * (hv[k] + tHv[k] + dt * lHv[k]);
      // 边界海绵(仅动量)
      const sp = this.sponge[k];
      uC *= sp;
      vC *= sp;
      // 隐式曼宁摩擦:hu /= 1 + dt·g·n²·|u|/h^(4/3),|u| = √(u²+v²)/h
      if (manning > 0) {
        const h = Math.max(H[k] + eC, hMin);
        const speed = Math.sqrt(uC * uC + vC * vC) / h;
        const cf = (dt * GRAVITY * manning * manning * speed) / Math.pow(h, 4 / 3);
        uC /= 1 + cf;
        vC /= 1 + cf;
      }
      eta[k] = eC;
      hu[k] = uC;
      hv[k] = vC;
    }
    // 辐射边界(特征投影):把边界单元投影到出射特征、令入射特征为零。
    // 右边界出右行波 w+=η+hu/c → η=w+/2, hu=c·w+/2;左边界出左行波 w−=η−hu/c 对称。
    // 等价于单向波方程 ∂tφ=−c·∂nφ 的稳态投影,让出海波透射、把反射压到 <2%。
    if (this.radiation) {
      const { radXnb, radYnb, nx } = this;
      for (let k = 0; k < eta.length; k++) {
        const hasX = radXnb[k] >= 0, hasY = radYnb[k] >= 0;
        if (!hasX && !hasY) continue;
        const c = Math.sqrt(GRAVITY * H[k]);
        if (c <= 0) continue; // 陆地边界:η 冻结、无出射波
        if (hasX) {
          if (radXnb[k] === k - 1) {           // 右边界:出射右行波
            const wp = eta[k] + hu[k] / c;
            eta[k] = 0.5 * wp; hu[k] = 0.5 * c * wp;
          } else {                              // 左边界:出射左行波
            const wm = eta[k] - hu[k] / c;
            eta[k] = 0.5 * wm; hu[k] = -0.5 * c * wm;
          }
        }
        if (hasY) {
          if (radYnb[k] === k - nx) {          // 上边界:出射上行波
            const wp = eta[k] + hv[k] / c;
            eta[k] = 0.5 * wp; hv[k] = 0.5 * c * wp;
          } else {                              // 下边界:出射下行波
            const wm = eta[k] - hv[k] / c;
            eta[k] = 0.5 * wm; hv[k] = -0.5 * c * wm;
          }
        }
      }
    }
  }
}
