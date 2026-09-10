/** 全局仿真配置(教学级简化参数) */

/** 默认(程序化地形)求解器网格分辨率(N×N) */
export const SIM_SIZE = 256;

/** 真实地形网格单边上限(性能保护) */
export const MAX_GRID = 384;

/** 默认模拟域边长,单位 km */
export const DOMAIN_KM = 240;

/** 重力加速度 m/s² */
export const GRAVITY = 9.81;

/**
 * CFL 安全因子:dt = CFL_SAFETY·min(dx,dy)/√(g·Hmax)。
 * 二维格式稳定条件为两方向库朗数之和 νx+νy ≤ 1,方形网格下取 0.5 即贴近上限。
 */
export const CFL_SAFETY = 0.5;

/** 由网格间距与最大水深计算稳定时间步长(s) */
export function computeStableDt(
  dxM: number, dyM: number, maxDepthM: number, cfl = CFL_SAFETY
): number {
  const c = Math.sqrt(GRAVITY * Math.max(maxDepthM, 1));
  return (cfl * Math.min(dxM, dyM)) / c;
}

/** 全球域纬度范围(墨卡托瓦片极限 ±84°) */
export const GLOBE_LAT_SPAN = 168.0;

/**
 * 缩减纬网(reduced-latitude grid)极地经向合并因子 k。
 * 高纬经向格距 dx·cosφ 随纬度收缩,迫使 dt→0;改为每 k 列合并(stride-k 采样),
 * 有效格距 k·dx·cosφ 保持有限,dt 不再被极地拖垮且波速正确。
 * |φ|>80° 每 4 列、>74° 每 2 列、其余每 1 列(与着色器 stride 同构)。
 */
export const POLAR_REDUCE_LAT2 = 74.0;
export const POLAR_REDUCE_LAT4 = 80.0;
export function polarStride(latDeg: number): number {
  const a = Math.abs(latDeg);
  return a > POLAR_REDUCE_LAT4 ? 4 : a > POLAR_REDUCE_LAT2 ? 2 : 1;
}

/**
 * 缩减纬网下最小经向有效格距因子 min(k·cosφ)(逐行扫描),用于 globe CFL dt。
 * 与 cpuSolver/着色器的 dxEff=k·dx·cos(max(|φ|,5°)) 同构;最小值出现在 k=1 带边缘(约 74°)。
 */
export function reducedGridMinFactor(latSpanDeg: number, ny: number): number {
  const deg = Math.PI / 180;
  let minF = Infinity;
  for (let j = 0; j < ny; j++) {
    const lat = ((j + 0.5) / ny - 0.5) * latSpanDeg;
    const f = polarStride(lat) * Math.cos(Math.max(Math.abs(lat), 5.0) * deg);
    if (f < minF) minF = f;
  }
  return Number.isFinite(minF) ? minF : 1;
}

/**
 * 曼宁摩擦系数 n(s/m^(1/3))。海面/海底粗糙度的教学级默认值:
 * 开阔深水取 0.025;近岸浅滩可增大。UI 滑杆范围 0–0.05。
 */
export const MANNING_N = 0.025;

/** 干单元判定阈值(m):总水深 h = H+η < H_MIN 视为干,通量置零、η 冻结 */
export const H_MIN = 1e-3;

/**
 * 默认时间倍速(相对真实时间的倍数,1 = 真实时间)。
 * 步频 = 倍速 / dt,与显示器帧率无关;科研对照场景选 1× 实时。
 */
export const DEFAULT_TIME_SCALE = 64;

// ---------------- 地理投影常数(等距圆柱近似,教学级) ----------------

/** 每纬度对应的 km 数 */
export const KM_PER_DEG_LAT = 111.32;

/** 经度 1° 对应的 km 数(随纬度变化,取域中心纬度) */
export function kmPerDegLon(latDeg: number): number {
  return 111.32 * Math.cos((latDeg * Math.PI) / 180);
}
