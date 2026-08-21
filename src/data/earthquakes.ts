/**
 * 历史著名海啸地震目录(教学级)。
 *
 * 断层几何参数取自 USGS 震源机制解与公开文献的近似值(走向/倾角/滑动角
 * 取整数,破裂尺度按 Wells & Coppersmith 1994 经验关系圆整)。
 * 用于驱动"真实地震震源"计算,不可用于科研或预警。
 */

export interface CatalogEarthquake {
  id: string;
  /** 年份(标识用) */
  year: number;
  name: string;
  /** 震中经纬度(度) */
  lon: number;
  lat: number;
  /** 矩震级 */
  mw: number;
  /** 断层走向(自北顺时针,度);断层向走向右侧倾斜 */
  strike: number;
  /** 倾角(度,0 = 水平) */
  dip: number;
  /** 滑动角(度,90 = 逆冲,0 = 走滑) */
  rake: number;
  /** 破裂长度 km(沿走向) */
  lengthKm: number;
  /** 破裂宽度 km(沿倾向) */
  widthKm: number;
  /** 断层顶缘深度 km */
  depthKm: number;
  /** 滑量集中区(asperity)沿走向的相对位置 0–1,缺省 0.5 居中 */
  asperity?: number;
}

export const EARTHQUAKE_CATALOG: CatalogEarthquake[] = [
  {
    id: 'tohoku2011',
    year: 2011,
    name: '日本东北海域(东日本大地震)',
    lon: 142.9, lat: 38.1, mw: 9.0,
    strike: 16, dip: 12, rake: 90,
    lengthKm: 500, widthKm: 200, depthKm: 5,
    asperity: 0.62,
  },
  {
    id: 'sumatra2004',
    year: 2004,
    name: '印尼苏门答腊(印度洋大海啸)',
    lon: 95.9, lat: 3.3, mw: 9.1,
    strike: 320, dip: 12, rake: 90,
    lengthKm: 1200, widthKm: 160, depthKm: 5,
    asperity: 0.42,
  },
  {
    id: 'chile1960',
    year: 1960,
    name: '智利瓦尔迪维亚(史上最强地震)',
    lon: -73.1, lat: -38.1, mw: 9.5,
    strike: 8, dip: 15, rake: 90,
    lengthKm: 1000, widthKm: 180, depthKm: 10,
    asperity: 0.55,
  },
  {
    id: 'alaska1964',
    year: 1964,
    name: '阿拉斯加威廉王子湾',
    lon: -147.6, lat: 61.0, mw: 9.2,
    strike: 223, dip: 9, rake: 90,
    lengthKm: 700, widthKm: 200, depthKm: 10,
  },
  {
    id: 'maule2010',
    year: 2010,
    name: '智利马乌莱',
    lon: -72.7, lat: -35.8, mw: 8.8,
    strike: 16, dip: 18, rake: 90,
    lengthKm: 500, widthKm: 150, depthKm: 10,
  },
  {
    id: 'cascadia1700',
    year: 1700,
    name: '卡斯卡迪亚(假想全破裂)',
    lon: -125.0, lat: 44.0, mw: 9.0,
    strike: 8, dip: 12, rake: 90,
    lengthKm: 1000, widthKm: 150, depthKm: 5,
  },
  {
    id: 'palu2018',
    year: 2018,
    name: '印尼帕卢(走滑型对照)',
    lon: 119.9, lat: -0.2, mw: 7.5,
    strike: 352, dip: 85, rake: 0,
    lengthKm: 150, widthKm: 20, depthKm: 5,
  },
];

/** 面板下拉选项(键 → 显示名) */
export function catalogOptions(): Record<string, string> {
  const opts: Record<string, string> = {};
  for (const q of EARTHQUAKE_CATALOG) {
    opts[q.id] = `${q.year} ${q.name} Mw${q.mw}`;
  }
  return opts;
}

export function findQuake(id: string): CatalogEarthquake | undefined {
  return EARTHQUAKE_CATALOG.find((q) => q.id === id);
}
