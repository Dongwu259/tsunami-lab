/**
 * 海啸观测点(验潮站)目录。
 * 经纬度取自真实沿海城镇的近似值,用于统计波的到达时间与峰值波高。
 * 仅教学用途,真实预警请以各国官方机构为准。
 */

export interface ObserverSite {
  id: string;
  name: string;
  lon: number;
  lat: number;
}

export const OBSERVER_SITES: ObserverSite[] = [
  { id: 'miyako', name: '宮古(日本三陆海岸)', lon: 141.95, lat: 39.64 },
  { id: 'hilo', name: '希洛(夏威夷)', lon: -155.08, lat: 19.72 },
  { id: 'crescent', name: '克雷森特城(美国西海岸)', lon: -124.2, lat: 41.75 },
  { id: 'kodiak', name: '科迪亚克(阿拉斯加)', lon: -152.41, lat: 57.79 },
  { id: 'valparaiso', name: '瓦尔帕莱索(智利)', lon: -71.62, lat: -33.05 },
  { id: 'callao', name: '卡亚俄(秘鲁)', lon: -77.15, lat: -12.05 },
  { id: 'padang', name: '巴东(苏门答腊)', lon: 100.35, lat: -0.95 },
  { id: 'hualien', name: '花莲(台湾东岸)', lon: 121.61, lat: 23.99 },
  { id: 'manila', name: '马尼拉湾', lon: 120.98, lat: 14.6 },
  { id: 'tauranga', name: '陶朗加(新西兰)', lon: 176.17, lat: -37.69 },
];
