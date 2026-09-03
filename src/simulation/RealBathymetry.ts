import { MAX_GRID, kmPerDegLon, KM_PER_DEG_LAT } from '../config';

/**
 * .tsunami 二进制格式(小端),头部共 78 字节:
 *   magic 'TSNB'(4) | u16 版本(2) | u32 宽(4) | u32 高(4)
 *   | f64 西(8) | f64 东(8) | f64 南(8) | f64 北(8)
 *   | char[32] 数据源名(32) | float32[宽×高] 高程(米,行主序,自南向北)
 * 高程数据紧随 78 字节头之后;读取偏移必须与写入端 tools/fetch_bathy.py 一致。
 */

const MAGIC = 0x424e5354; // 'TSNB' 小端

/** 头部字节数:4 + 2 + 4 + 4 + 8×4 + 32 = 78,高程数据从此偏移开始 */
const HEADER_BYTES = 78;
/** 数据源名在头部中的偏移与长度(char[32]) */
const NAME_OFFSET = 46;
const NAME_LEN = 32;

export interface RealRegion {
  /** 源网格尺寸 */
  width: number;
  height: number;
  /** 经纬度边界 */
  west: number;
  east: number;
  south: number;
  north: number;
  source: string;
  /** 高程(米,自南向北行主序) */
  grid: Float32Array;
}

/** 仿真域几何(由区域换算而来) */
export interface RegionGeometry {
  sizeX: number;
  sizeY: number;
  domainKmX: number;
  domainKmY: number;
  dxM: number;
  dyM: number;
  /** 重采样到求解器网格后的海床(米) */
  grid: Float32Array;
}

/** 解析 .tsunami 二进制;格式非法时抛错 */
export function parseTsunamiBinary(buf: ArrayBuffer): RealRegion {
  const view = new DataView(buf);
  if (buf.byteLength < HEADER_BYTES || view.getUint32(0, true) !== MAGIC) {
    throw new Error('不是有效的 .tsunami 地形文件');
  }
  const version = view.getUint16(4, true);
  if (version !== 1) {
    throw new Error(`不支持的文件版本 v${version}`);
  }
  const width = view.getUint32(6, true);
  const height = view.getUint32(10, true);
  const west = view.getFloat64(14, true);
  const east = view.getFloat64(22, true);
  const south = view.getFloat64(30, true);
  const north = view.getFloat64(38, true);

  const nameBytes = new Uint8Array(buf, NAME_OFFSET, NAME_LEN);
  const end = nameBytes.indexOf(0);
  const source = new TextDecoder().decode(
    nameBytes.subarray(0, end < 0 ? NAME_LEN : end)
  );

  const expected = HEADER_BYTES + width * height * 4;
  if (buf.byteLength < expected) {
    throw new Error('文件损坏:数据长度不足');
  }
  const grid = new Float32Array(
    buf.slice(HEADER_BYTES, HEADER_BYTES + width * height * 4)
  );

  return { width, height, west, east, south, north, source, grid };
}

/**
 * 由经纬度区域计算仿真域几何,并双线性重采样到求解器网格。
 * 长边固定 MAX_GRID,短边按纵横比取偶数。
 */
export function regionToGeometry(region: RealRegion): RegionGeometry {
  const latMid = (region.south + region.north) / 2;
  const kmX = (region.east - region.west) * kmPerDegLon(latMid);
  const kmY = (region.north - region.south) * KM_PER_DEG_LAT;
  if (kmX <= 0 || kmY <= 0) {
    throw new Error('区域范围非法');
  }

  const aspect = kmX / kmY;
  let sizeX: number;
  let sizeY: number;
  if (aspect >= 1) {
    sizeX = MAX_GRID;
    sizeY = Math.max(64, Math.round(MAX_GRID / aspect / 2) * 2);
  } else {
    sizeY = MAX_GRID;
    sizeX = Math.max(64, Math.round((MAX_GRID * aspect) / 2) * 2);
  }

  const dxM = (kmX * 1000) / sizeX;
  const dyM = (kmY * 1000) / sizeY;

  // --- 双线性重采样(源网格 → sizeX×sizeY) ---
  const out = new Float32Array(sizeX * sizeY);
  const sx = (region.width - 1) / (sizeX - 1);
  const sy = (region.height - 1) / (sizeY - 1);
  for (let j = 0; j < sizeY; j++) {
    const fy = j * sy;
    const y0 = Math.min(Math.floor(fy), region.height - 2);
    const ty = fy - y0;
    for (let i = 0; i < sizeX; i++) {
      const fx = i * sx;
      const x0 = Math.min(Math.floor(fx), region.width - 2);
      const tx = fx - x0;
      const r0 = y0 * region.width;
      const r1 = (y0 + 1) * region.width;
      const a = region.grid[r0 + x0] * (1 - tx) + region.grid[r0 + x0 + 1] * tx;
      const b = region.grid[r1 + x0] * (1 - tx) + region.grid[r1 + x0 + 1] * tx;
      out[j * sizeX + i] = a * (1 - ty) + b * ty;
    }
  }

  return { sizeX, sizeY, domainKmX: kmX, domainKmY: kmY, dxM, dyM, grid: out };
}
