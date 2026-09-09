import GUI from 'lil-gui';
import { DEFAULT_TIME_SCALE } from '../config';
import { catalogOptions, EARTHQUAKE_CATALOG } from '../data/earthquakes';

export interface SimParams {
  playing: boolean;
  /** 时间倍速:相对真实时间的倍数(1 = 实时) */
  timeScale: number;
  /** 数值格式:'v2' = MUSCL-RK2(二阶),'lf' = Lax–Friedrichs(一阶对照) */
  scheme: string;
  exaggeration: number;
  colorRange: number;
  waterOpacity: number;
  showTerrain: boolean;
  /** P/S 波圈(地震体波扩散可视化) */
  showSeismic: boolean;
  wireframe: boolean;
  magnitude: number;
  radiusKm: number;
  /** 真实地震目录选中的 id */
  quakeId: string;
  /** USGS 实时目录 */
  usgsMinMag: number;
  usgsDays: number;
  liveQuakeId: string;
  /** 科研模式 */
  geoUrl: string;
  geoSimMinutes: number;
  geoFrames: number;
  geoAutoPlay: boolean;
}

export interface PanelCallbacks {
  onReset(): void;
  onSchemeChange(scheme: string): void;
  onRandomQuake(): void;
  onTohokuQuake(): void;
  onExaggerationChange(ex: number): void;
  onColorRangeChange(meters: number): void;
  onWaterOpacityChange(opacity: number): void;
  onTerrainVisible(visible: boolean): void;
  onSeismicVisible(visible: boolean): void;
  onWireframe(on: boolean): void;
  onLoadTohoku(): void;
  onLoadGlobe(): void;
  onImportFile(file: File): void;
  onResetProcedural(): void;
  onRealQuake(): void;
  onFetchLive(): void;
  onTriggerLive(): void;
  onGeoConnect(): void;
  onGeoSubmit(): void;
  onGeoNextFrame(): void;
}

export interface PanelHandle {
  params: SimParams;
  /** 更新区域信息行 */
  setRegionInfo(text: string): void;
  /** 同步震级滑条显示(外部改 params.magnitude 后调用) */
  refreshMagnitude(): void;
  /** 更新科研模式状态行 */
  setGeoStatus(text: string): void;
  /** 更新 USGS 实时目录下拉选项 */
  setLiveOptions(options: Record<string, string>, firstId: string): void;
  /** 更新 USGS 实时目录状态行 */
  setLiveStatus(text: string): void;
}

/** 创建控制面板 */
export function createPanel(cb: PanelCallbacks): PanelHandle {
  const params: SimParams = {
    playing: true,
    timeScale: DEFAULT_TIME_SCALE,
    scheme: 'v2',
    exaggeration: 3,
    colorRange: 2,
    waterOpacity: 0.8,
    showTerrain: true,
    showSeismic: true,
    wireframe: false,
    magnitude: 8.5,
    radiusKm: 18,
    quakeId: EARTHQUAKE_CATALOG[0].id,
    usgsMinMag: 7.0,
    usgsDays: 7,
    liveQuakeId: '',
    geoUrl: 'http://localhost:8100',
    geoSimMinutes: 60,
    geoFrames: 24,
    geoAutoPlay: true,
  };

  const gui = new GUI({ title: '控制台' });
  // 窄屏(移动端)默认折叠面板,避免遮挡场景
  if (window.innerWidth < 760) gui.close();

  // --- 模拟 ---
  const sim = gui.addFolder('模拟');
  sim.add(params, 'playing').name('播放');
  sim
    .add(params, 'timeScale', {
      '实时 1×': 1, '2×': 2, '4×': 4, '8×': 8, '16×': 16,
      '32×': 32, '64×': 64, '128×': 128, '256×': 256, '512×': 512,
    })
    .name('时间倍速(相对实时)');
  sim
    .add(params, 'scheme', {
      'MUSCL-RK2(二阶低耗散)': 'v2',
      'Lax–Friedrichs(一阶对照)': 'lf',
    })
    .name('数值格式')
    .onChange((v: string) => cb.onSchemeChange(v));
  sim.add({ reset: cb.onReset }, 'reset').name('重置海面');

  // --- 地震源 ---
  const quake = gui.addFolder('地震源');
  const magCtrl = quake.add(params, 'magnitude', 6.5, 9.5, 0.1).name('矩震级 Mw');
  quake.add(params, 'radiusKm', 5, 40, 1).name('破裂半径 (km)');
  quake.add({ trigger: cb.onRandomQuake }, 'trigger').name('随机触发地震');
  quake.add({ t2011: cb.onTohokuQuake }, 't2011').name('预设:东北海域 Mw9.0');
  quake
    .add({ hint: '提示:直接点击海面也可触发' }, 'hint')
    .name('提示:直接点击海面也可触发')
    .disable();

  // --- 地形数据 ---
  const data = gui.addFolder('地形数据');
  data.add({ g: cb.onLoadGlobe }, 'g').name('加载:3D 全球地球');
  data.add({ t: cb.onLoadTohoku }, 't').name('加载:东北海域真实地形');
  data
    .add(
      {
        i: () => {
          const input = document.createElement('input');
          input.type = 'file';
          input.accept = '.tsunami';
          input.onchange = () => {
            const file = input.files?.[0];
            if (file) cb.onImportFile(file);
          };
          input.click();
        },
      },
      'i'
    )
    .name('导入 .tsunami 文件…');
  data.add({ r: cb.onResetProcedural }, 'r').name('恢复程序化地形');
  const regionObj = { info: '程序化理想地形 240×240 km' };
  const regionCtrl = data.add(regionObj, 'info').name('区域').disable();

  // --- 真实地震(历史目录) ---
  const quakeCat = gui.addFolder('真实地震(历史目录)');
  quakeCat.add(params, 'quakeId', catalogOptions()).name('选择地震');
  quakeCat.add({ go: cb.onRealQuake }, 'go').name('触发选中的真实地震');
  quakeCat
    .add({ hint: '按真实断层参数计算隆起/沉降双极震源' }, 'hint')
    .name('按真实断层参数计算双极震源')
    .disable();

  // --- 实时地震目录(USGS) ---
  const live = gui.addFolder('实时地震目录 (USGS)');
  live.add(params, 'usgsMinMag', 6.0, 8.0, 0.5).name('最小震级');
  live
    .add(params, 'usgsDays', {
      '近 1 天': 1, '近 3 天': 3, '近 7 天': 7, '近 14 天': 14, '近 30 天': 30,
    })
    .name('时间范围');
  live.add({ f: cb.onFetchLive }, 'f').name('拉取最新地震');
  const liveCtrl = live
    .add(params, 'liveQuakeId', { '(先拉取)': '' })
    .name('选择地震');
  live.add({ t: cb.onTriggerLive }, 't').name('触发选中的实时地震');
  const liveObj = { status: '未拉取 · 断层参数自动估算' };
  const liveStatusCtrl = live.add(liveObj, 'status').name('状态').disable();

  // --- 科研模式(GeoClaw) ---
  const geo = gui.addFolder('科研模式 (GeoClaw)');
  geo.add(params, 'geoUrl').name('后端地址');
  geo.add({ c: cb.onGeoConnect }, 'c').name('测试连接');
  geo.add(params, 'geoSimMinutes', 10, 240, 10).name('模拟时长 (分钟)');
  geo.add(params, 'geoFrames', 8, 100, 4).name('输出帧数');
  geo.add({ s: cb.onGeoSubmit }, 's').name('提交 GeoClaw 模拟');
  geo.add(params, 'geoAutoPlay').name('回放自动播放');
  geo.add({ n: cb.onGeoNextFrame }, 'n').name('回放:下一帧');
  const geoObj = { status: '未连接' };
  const geoCtrl = geo.add(geoObj, 'status').name('状态').disable();

  // --- 显示 ---
  const view = gui.addFolder('显示');
  view
    .add(params, 'exaggeration', 1, 8, 0.5)
    .name('垂直夸张')
    .onChange((v: number) => cb.onExaggerationChange(v));
  view
    .add(params, 'colorRange', 0.5, 20, 0.5)
    .name('色标量程 ±(m)')
    .onChange((v: number) => cb.onColorRangeChange(v));
  view
    .add(params, 'waterOpacity', 0.15, 1, 0.05)
    .name('海面不透明度')
    .onChange((v: number) => cb.onWaterOpacityChange(v));
  view
    .add(params, 'showTerrain')
    .name('显示海底地形')
    .onChange((v: boolean) => cb.onTerrainVisible(v));
  view
    .add(params, 'showSeismic')
    .name('P/S 波圈')
    .onChange((v: boolean) => cb.onSeismicVisible(v));
  view
    .add(params, 'wireframe')
    .name('线框模式')
    .onChange((v: boolean) => cb.onWireframe(v));

  return {
    params,
    setRegionInfo: (text: string) => {
      regionObj.info = text;
      regionCtrl.updateDisplay();
    },
    refreshMagnitude: () => magCtrl.updateDisplay(),
    setGeoStatus: (text: string) => {
      geoObj.status = text;
      geoCtrl.updateDisplay();
    },
    setLiveOptions: (options: Record<string, string>, firstId: string) => {
      params.liveQuakeId = firstId;
      liveCtrl.options(Object.keys(options).length ? options : { '(无记录)': '' });
      liveCtrl.updateDisplay();
    },
    setLiveStatus: (text: string) => {
      liveObj.status = text;
      liveStatusCtrl.updateDisplay();
    },
  };
}

/** 震级 → 初始波幅的教学级经验换算(M6.5≈0.5m,每 +1 级翻倍) */
export function magnitudeToAmplitude(mw: number): number {
  return 0.5 * Math.pow(2, mw - 6.5);
}
