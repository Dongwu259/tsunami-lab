/**
 * USGS 实时地震目录客户端。
 *
 * 数据源:FDSN Event Web Service(geojson 格式,原生支持 CORS)。
 * 实时目录只有震中/震级/深度,没有断层机制解——触发模拟时按
 * "逆冲型海啸地震"的经验参数自动估算断层几何(见 toLiveQuakeCatalog),
 * 面板会如实标注"(参数估算)"。
 */

export const USGS_QUERY_URL = 'https://earthquake.usgs.gov/fdsnws/event/1/query';

/** USGS 实时地震记录(已裁剪为模拟所需字段) */
export interface LiveEarthquake {
  id: string;
  /** 发震时刻(UTC 毫秒时间戳) */
  time: number;
  lon: number;
  lat: number;
  mag: number;
  depthKm: number;
  place: string;
  /** USGS 事件详情页 */
  url: string;
}

/**
 * 拉取最近 days 天内 M ≥ minMag 的全球地震(按震级降序)。
 * 网络失败或超时抛错,由调用方提示。
 */
export async function fetchRecentQuakes(
  minMag: number,
  days: number,
  limit = 20
): Promise<LiveEarthquake[]> {
  const end = new Date();
  const start = new Date(end.getTime() - days * 24 * 3600 * 1000);
  const params = new URLSearchParams({
    format: 'geojson',
    starttime: start.toISOString(),
    endtime: end.toISOString(),
    minmagnitude: minMag.toFixed(1),
    orderby: 'magnitude',
    limit: String(limit),
  });
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 12000);
  try {
    const resp = await fetch(`${USGS_QUERY_URL}?${params}`, { signal: ctrl.signal });
    if (!resp.ok) throw new Error(`USGS 服务 HTTP ${resp.status}`);
    const body = (await resp.json()) as {
      features?: {
        id: string;
        properties: {
          mag: number | null;
          place: string | null;
          time: number;
          url?: string;
        };
        geometry: { coordinates: [number, number, number] };
      }[];
    };
    const list: LiveEarthquake[] = [];
    for (const f of body.features ?? []) {
      const p = f.properties;
      if (p.mag == null) continue;
      const [lon, lat, depth] = f.geometry.coordinates;
      list.push({
        id: f.id,
        time: p.time,
        lon,
        lat,
        mag: p.mag,
        depthKm: Math.max(depth, 0),
        place: p.place || '未知地点',
        url: p.url || '',
      });
    }
    return list;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * 实时记录 → 目录震源格式。
 * 无机制解,按逆冲型海啸地震经验参数估算:
 *  - 破裂长度用 Wells & Coppersmith 1994 逆冲关系;宽度 ≈ L/2.2
 *  - 走向取 0°(无信息,仅影响双极方向),倾角 12°,滑动角 90°
 *  - 顶缘深度取震源深度(上限 30 km,过深断层海面位移弱)
 */
export function toLiveQuakeCatalog(q: LiveEarthquake) {
  const lengthKm = Math.pow(10, (q.mag - 4.44) / 1.49);
  const widthKm = lengthKm / 2.2;
  return {
    id: `live-${q.id}`,
    year: new Date(q.time).getFullYear(),
    name: `${q.place}(实时·参数估算)`,
    lon: q.lon,
    lat: q.lat,
    mw: q.mag,
    strike: 0,
    dip: 12,
    rake: 90,
    lengthKm,
    widthKm,
    depthKm: Math.min(Math.max(q.depthKm, 2), 30),
  };
}
