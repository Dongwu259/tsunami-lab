/**
 * 真实地震震源:由断层几何参数计算初始海面位移场 η。
 *
 * 物理模型(教学级,Okada 1985 的远场主导结构):
 *  - 逆冲/正断型(倾滑分量)在海面产生"隆起 + 沉降"双极型位移,
 *    双极沿断层倾向排列,波长 ~ 破裂宽度的水平投影——这是真实
 *    弹性位错解的远场主导项,也是跨洋海啸传播的主要能量来源;
 *  - 走向滑移分量对海面垂直位移无一级贡献(走滑型地震海啸弱);
 *  - **非均匀滑动分布**:大破裂沿走向拆分为多个子断层,滑量权重
 *    按 asperity(滑量集中区)高斯凸包分布,均值归一保持总矩不变。
 *
 * 坐标约定:断层走向自北顺时针,断层向走向右侧倾斜(与 Aki & Richards 一致);
 * 逆冲(rake=90°)时上盘(右侧)隆起、海沟侧(左侧)沉降。
 */

import { CatalogEarthquake } from '../data/earthquakes';

/** 指定纬度处每经度对应的 km 数 */
function kmPerDegAt(latDeg: number): number {
  return 111.32 * Math.max(Math.cos((latDeg * Math.PI) / 180), 1e-3);
}

/** 经纬度描述的求解域(与 RealRegion / 全球网格对齐,自南向北行主序) */
export interface GeoDomain {
  west: number;
  east: number;
  south: number;
  north: number;
  sizeX: number;
  sizeY: number;
}

/** 震级 → 平均滑量(m):M0 = 10^(1.5Mw+9.1),slip = M0/(μ·L·W),
 * 用断层真实破裂尺度(而非经验长度公式)避免滑量失真 */
export function magnitudeToSlip(mw: number, lengthKm: number, widthKm: number): number {
  const m0 = Math.pow(10, 1.5 * mw + 9.1);
  return m0 / (3e10 * lengthKm * 1000 * widthKm * 1000);
}

/** 跨断层剖面:ξ = 距断层顶缘投影的水平距离 / 特征波长 */
function crossProfile(xi: number): number {
  return 2.6 * xi * Math.exp(-2.2 * xi * xi);
}

/**
 * 计算地震产生的初始海面位移场(m,自南向北行主序)。
 * 纯走滑断层返回 null(无一级垂直位移)。
 */
export function quakeSurfaceEta(
  q: CatalogEarthquake,
  domain: GeoDomain
): Float32Array | null {
  const D2R = Math.PI / 180;
  const strikeR = q.strike * D2R;
  const dipR = q.dip * D2R;
  const rakeR = q.rake * D2R;

  // 倾向滑移分量(逆冲为正);走滑分量对 η 无一级贡献
  const dipSlipFrac = Math.sin(rakeR);
  if (Math.abs(dipSlipFrac) < 0.15) return null;

  // 断层局部坐标基(x 沿走向,y 指向倾斜侧),单位:度(局部近似)
  const sx = Math.sin(strikeR);
  const cx = Math.cos(strikeR);

  const cosLatEpic = Math.max(Math.cos(q.lat * D2R), 1e-3);
  const kmPerLon = 111.32 * cosLatEpic;
  const kmPerLat = 111.32;

  // 断层顶/底缘在地表的投影(沿倾向的水平距离)
  const topY = q.depthKm / Math.tan(dipR);
  const botY = topY + q.widthKm * Math.cos(dipR);
  const yMid = (topY + botY) / 2;
  let lambda = Math.max(botY - topY, q.depthKm, 25); // 特征波长 km

  // 网格欠采样保护:双极波长不足 ~2.5 个网格时会被采样成单格尖峰,
  // 此时平滑展宽到可分辨尺度(峰值幅值相应下降,总能量趋势不变)
  const cellKmLon = ((domain.east - domain.west) * kmPerDegAt(q.lat)) / domain.sizeX;
  const cellKmLat = ((domain.north - domain.south) * 111.32) / domain.sizeY;
  const cellKm = Math.max(cellKmLon, cellKmLat);
  lambda = Math.max(lambda, cellKm * 2.5);

  // 幅值:滑量 × 倾滑分量 × 海底耦合系数,限幅到观测典型区间
  const slip = magnitudeToSlip(q.mw, q.lengthKm, q.widthKm);
  const amp = Math.min(Math.max(slip * Math.abs(dipSlipFrac) * 0.35, 0.2), 15);
  const sign = dipSlipFrac >= 0 ? 1 : -1; // 逆冲隆起在上盘,正断反之

  // --- 沿走向拆分子断层,滑量按 asperity 高斯凸包非均匀分布 ---
  const nSeg = Math.min(6, Math.max(1, Math.round(q.lengthKm / 120)));
  const segL = q.lengthKm / nSeg;
  const asperity = q.asperity ?? 0.5;
  const segFx0: number[] = [];
  const segW: number[] = [];
  let wSum = 0;
  for (let k = 0; k < nSeg; k++) {
    segFx0.push((k + 0.5 - nSeg / 2) * segL);
    const c = (k + 0.5) / nSeg;
    const d = ((c - asperity) * nSeg) / 1.4;
    const w = 0.35 + 0.9 * Math.exp(-d * d);
    segW.push(w);
    wSum += w;
  }
  // 权重均值归一到 1:总矩不变,仅重新分配滑量
  const wNorm = nSeg / wSum;

  const lonSpan = domain.east - domain.west;
  const eta = new Float32Array(domain.sizeX * domain.sizeY);

  for (let j = 0; j < domain.sizeY; j++) {
    const lat = domain.south + ((j + 0.5) / domain.sizeY) * (domain.north - domain.south);
    const kmLat = (lat - q.lat) * kmPerLat;
    for (let i = 0; i < domain.sizeX; i++) {
      let lon = domain.west + ((i + 0.5) / domain.sizeX) * lonSpan;
      // 跨 180° 经线取最短差
      let dLon = lon - q.lon;
      if (dLon > 180) dLon -= 360;
      if (dLon < -180) dLon += 360;
      const kmLon = dLon * kmPerLon;

      // 转到断层局部坐标
      const fx = kmLon * sx + kmLat * cx; // 沿走向
      const fy = kmLon * cx - kmLat * sx; // 垂直走向(倾向侧为正)

      const halfL = q.lengthKm / 2;
      if (Math.abs(fx) > halfL * 1.6) continue;
      const xi = (fy - yMid) / lambda;
      if (Math.abs(xi) > 3) continue;

      // 各子断层贡献:自身余弦锥度 × 非均匀滑量权重
      let acc = 0;
      for (let k = 0; k < nSeg; k++) {
        const lfx = fx - segFx0[k];
        if (Math.abs(lfx) > segL * 0.8) continue;
        const taper = Math.abs(lfx) < segL / 2
          ? Math.cos((Math.PI * lfx) / segL)
          : 0;
        acc += segW[k] * wNorm * taper;
      }
      if (acc > 1e-4) {
        eta[j * domain.sizeX + i] += sign * amp * crossProfile(xi) * acc;
      }
    }
  }
  return eta;
}
