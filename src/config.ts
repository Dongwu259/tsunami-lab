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
