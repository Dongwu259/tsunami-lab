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
 * 单个子步时间步长(s)。
 * 越接近 CFL 上限(dx/√(gH) ≈ 4.9 s),Lax–Friedrichs 格式的数值耗散越小。
 */
export const SIM_DT = 3.0;

/**
 * 全球球面模式时间步长(s)。
 * 二维 LF 格式稳定条件为两方向库朗数之和 νx+νy ≤ 1:
 *   νy = dt·√(gH)/dy ≈ 0.39(最深海域),需给极地经向保护留出余量,
 *   取 50 s 而非逼近单向上限,避免长时间运行后高纬模式累积爆炸。
 */
export const GLOBE_DT = 50.0;

/** 全球域纬度范围(墨卡托瓦片极限 ±84°) */
export const GLOBE_LAT_SPAN = 168.0;

/**
 * 默认时间倍速(相对真实时间的倍数,1 = 真实时间)。
 * 步频 = 倍速 / dt,与显示器帧率无关;科研对照场景选 1× 实时。
 * 64×:dt=3s 平面域约 21 步/秒,dt=50s 全球域约 1.3 步/秒。
 */
export const DEFAULT_TIME_SCALE = 64;

// ---------------- 地理投影常数(等距圆柱近似,教学级) ----------------

/** 每纬度对应的 km 数 */
export const KM_PER_DEG_LAT = 111.32;

/** 经度 1° 对应的 km 数(随纬度变化,取域中心纬度) */
export function kmPerDegLon(latDeg: number): number {
  return 111.32 * Math.cos((latDeg * Math.PI) / 180);
}
