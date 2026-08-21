import * as THREE from 'three';
import {
  checkHealth,
  fetchFrame,
  fetchFrames,
  FrameMeta,
  jobStatus,
  magnitudeToSlip,
  submitJob,
} from './api/geoclawClient';
import {
  fetchRecentQuakes,
  LiveEarthquake,
  toLiveQuakeCatalog,
} from './api/usgsClient';
import { DOMAIN_KM, GLOBE_DT, GLOBE_LAT_SPAN, KM_PER_DEG_LAT, SIM_DT, SIM_SIZE, kmPerDegLon } from './config';
import { CatalogEarthquake, findQuake } from './data/earthquakes';
import { ObserverSite, OBSERVER_SITES } from './data/observers';
import { SceneApp, ViewMode } from './scene/SceneApp';
import { generateBathymetry } from './simulation/bathymetry';
import {
  GeoDomain,
  quakeSurfaceEta,
} from './simulation/quakeSource';
import {
  parseTsunamiBinary,
  RealRegion,
  regionToGeometry,
} from './simulation/RealBathymetry';
import { TsunamiSolver } from './simulation/TsunamiSolver';
import {
  createPanel,
  magnitudeToAmplitude,
  PanelHandle,
} from './ui/ControlPanel';

/**
 * 应用总装:渲染器 → 求解器 → 场景 → 控制面板 → 主循环。
 * 支持在程序化地形与真实地形(.tsunami)之间切换,切换时重建求解器与网格。
 */
export class TsunamiApp {
  private renderer: THREE.WebGLRenderer;
  private sceneApp: SceneApp;
  private solver: TsunamiSolver;
  private panel: PanelHandle;

  private simSeconds = 0;
  private frames = 0;
  private fpsTimer = performance.now();

  /** 帧率无关的步进累加器(单位:子步) */
  private stepAcc = 0;
  private lastFrameTime = performance.now();

  // --- 科研模式(GeoClaw)状态 ---
  private regionBuf: ArrayBuffer | null = null;
  private regionMeta: RealRegion | null = null;
  /** 当前视图模式(平面区域 / 全球球面) */
  private viewMode: ViewMode = 'plane';
  private geoJobId: string | null = null;
  private geoPollTimer: number | null = null;
  private geoPlayTimer: number | null = null;
  private geoFrames: { meta: FrameMeta; eta: Float32Array }[] = [];
  private geoFrameIdx = 0;

  // --- 观测点(验潮站)状态 ---
  private observers: { site: ObserverSite; u: number; v: number; arrivedAt: number | null; peak: number }[] = [];
  private peakBuf: Float32Array | null = null;
  private observerRows: { arr: HTMLElement; pk: HTMLElement }[] = [];

  // --- USGS 实时目录 ---
  private liveQuakes: LiveEarthquake[] = [];

  /** 震源序列(倒计时→视角→标记→注入)忙标志,防止并发触发 */
  private quakeSeqBusy = false;

  // --- P/S 波圈(地震体波可视化)---
  /** 发震时刻的模拟时钟(s);null = 尚未记录 */
  private quakeSimT: number | null = null;
  /** 高分辨率渲染用海床纹理(全球地形升级) */
  private hiBathyTex: THREE.DataTexture | null = null;

  private elTime = document.getElementById('stat-time')!;
  private elPeak = document.getElementById('stat-peak')!;
  private elFps = document.getElementById('stat-fps')!;
  private elRegion = document.getElementById('stat-region')!;
  private elLegendMin = document.getElementById('legend-min')!;
  private elLegendMax = document.getElementById('legend-max')!;
  private elObserverPanel = document.getElementById('observer-panel')!;
  private elObserverTable = document.getElementById('observer-table')!;
  private elCountdown = document.getElementById('countdown')!;

  constructor(container: HTMLElement) {
    // --- 渲染器 ---
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(this.renderer.domElement);

    // --- 初始:程序化地形 ---
    const bathy = generateBathymetry();
    const dx = (DOMAIN_KM * 1000) / SIM_SIZE;
    this.solver = new TsunamiSolver(this.renderer, bathy, SIM_SIZE, SIM_SIZE, dx, dx);
    this.sceneApp = new SceneApp(container, this.renderer, this.solver);
    this.sceneApp.setExaggeration(3);

    // --- 控制面板 ---
    this.panel = createPanel({
      onReset: () => this.resetSea(),
      onRandomQuake: () => {
        const u = 0.12 + Math.random() * 0.34;
        const v = 0.12 + Math.random() * 0.76;
        this.quakeAt(u, v);
      },
      onTohokuQuake: () => {
        // 2011 东北海域地震震中(约 142.9E, 38.1N)
        this.panel.params.magnitude = 9.0;
        this.panel.refreshMagnitude();
        const u = this.viewMode === 'globe' ? (142.9 + 180) / 360 : 0.29;
        const v = this.viewMode === 'globe' ? (38.1 + 84) / GLOBE_LAT_SPAN : 0.45;
        void this.runEpicenterSequence(u, v, () => this.quakeAt(u, v));
      },
      onExaggerationChange: (ex) => this.sceneApp.setExaggeration(ex),
      onColorRangeChange: (m) => {
        this.sceneApp.setColorRange(m);
        this.elLegendMin.textContent = `-${m} m`;
        this.elLegendMax.textContent = `+${m} m`;
      },
      onWaterOpacityChange: (o) => this.sceneApp.setWaterOpacity(o),
      onTerrainVisible: (v) => this.sceneApp.setTerrainVisible(v),
      onSeismicVisible: (v) => this.sceneApp.setSeismicVisible(v),
      onWireframe: (v) => this.sceneApp.setWireframe(v),
      onLoadTohoku: () => this.loadTohoku(),
      onLoadGlobe: () => this.loadGlobe(),
      onImportFile: (file) => this.importFile(file),
      onResetProcedural: () => this.resetProcedural(),
      onRealQuake: () => this.triggerRealQuake(),
      onFetchLive: () => this.fetchLiveQuakes(),
      onTriggerLive: () => this.triggerLiveQuake(),
      onGeoConnect: () => this.geoConnect(),
      onGeoSubmit: () => this.geoSubmit(),
      onGeoNextFrame: () => this.geoNextFrame(),
    });

    // --- 点击海面触发地震(区分点击与拖拽) ---
    let downX = 0;
    let downY = 0;
    const canvas = this.renderer.domElement;
    canvas.addEventListener('pointerdown', (e) => {
      downX = e.clientX;
      downY = e.clientY;
    });
    canvas.addEventListener('pointerup', (e) => {
      const moved = Math.hypot(e.clientX - downX, e.clientY - downY);
      if (moved > 6) return; // 视为拖拽旋转
      const uv = this.sceneApp.pickUv(e.clientX, e.clientY);
      if (uv) this.quakeAt(uv.u, uv.v);
    });

    // --- 拖拽文件导入地形 ---
    container.addEventListener('dragover', (e) => e.preventDefault());
    container.addEventListener('drop', (e) => {
      e.preventDefault();
      const file = e.dataTransfer?.files?.[0];
      if (file) this.importFile(file);
    });

    // --- 开场演示:海沟处一次 M8.5 地震 ---
    setTimeout(() => this.quakeAt(0.3, 0.5), 800);

    this.loop();
  }

  // ---------------------------------------------------------------- 地形切换

  /** 用新海床重建求解器与场景域。
   * renderSize + hiResGrid:可选的高分辨率渲染通道(仅提升地形视觉细节,
   * 求解网格仍用降采样后的 bathy,保持计算性能) */
  private rebuild(
    bathy: Float32Array,
    sizeX: number,
    sizeY: number,
    dxM: number,
    dyM: number,
    regionText: string,
    mode: ViewMode = 'plane',
    dtSeconds = SIM_DT,
    renderSize?: { x: number; y: number },
    hiResGrid?: Float32Array
  ): void {
    this.stopGeoReplay();
    this.viewMode = mode;
    this.solver.dispose();
    this.solver = new TsunamiSolver(
      this.renderer, bathy, sizeX, sizeY, dxM, dyM, mode === 'globe', dtSeconds
    );
    // 高分辨率渲染纹理:RGBA 浮点(r = 高程),与求解器 bathyTexture 格式一致;
    // 旧纹理由 setRenderBathy 内部统一 dispose
    if (renderSize && hiResGrid) {
      const tex = new THREE.DataTexture(
        new Float32Array(renderSize.x * renderSize.y * 4),
        renderSize.x, renderSize.y, THREE.RGBAFormat, THREE.FloatType
      );
      const d = tex.image.data as unknown as Float32Array;
      for (let i = 0; i < hiResGrid.length; i++) {
        d[i * 4] = hiResGrid[i];
        d[i * 4 + 3] = 1;
      }
      if (this.renderer.extensions.get('OES_texture_float_linear')) {
        tex.minFilter = THREE.LinearFilter;
        tex.magFilter = THREE.LinearFilter;
      }
      tex.needsUpdate = true;
      this.hiBathyTex = tex;
      this.sceneApp.setRenderBathy(tex, renderSize);
    } else {
      this.hiBathyTex = null;
      this.sceneApp.setRenderBathy(null);
    }
    this.sceneApp.setMode(mode, this.solver);
    this.sceneApp.setDomain(this.solver, renderSize);
    this.sceneApp.setExaggeration(this.panel.params.exaggeration);
    this.simSeconds = 0;
    this.quakeSimT = null;                 // 域切换后旧震源波圈失效
    this.sceneApp.setSeismicTime(0);
    this.elRegion.textContent = regionText;
    this.panel.setRegionInfo(regionText);
    this.sceneApp.setEpicenterMarker(null); // 域切换后旧震源标记失效
    this.rebuildObservers(bathy);
    this.updateStats(true);
  }

  /** 加载随包的真实地形(2011 东北海域) */
  private async loadTohoku(): Promise<void> {
    try {
      const resp = await fetch('data/tohoku.tsunami');
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const buf = await resp.arrayBuffer();
      this.applyRegion(parseTsunamiBinary(buf), buf);
    } catch (err) {
      alert(`加载真实地形失败：${err}\n请先运行 tools/fetch_bathy.py 生成数据。`);
    }
  }
  
  /** 用户导入 .tsunami 文件 */
  private async importFile(file: File): Promise<void> {
    try {
      const buf = await file.arrayBuffer();
      this.applyRegion(parseTsunamiBinary(buf), buf);
    } catch (err) {
      alert(`导入失败：${err}`);
    }
  }
  
  private applyRegion(region: RealRegion, rawBuf: ArrayBuffer): void {
    this.regionMeta = region;
    this.regionBuf = rawBuf;
    const geo = regionToGeometry(region);
    const text =
      `${region.source} ` +
      `[${region.west.toFixed(1)}°~${region.east.toFixed(1)}°E, ` +
      `${region.south.toFixed(1)}°~${region.north.toFixed(1)}°N] ` +
      `${geo.domainKmX.toFixed(0)}×${geo.domainKmY.toFixed(0)} km`;
    this.rebuild(geo.grid, geo.sizeX, geo.sizeY, geo.dxM, geo.dyM, text);
  }

  /** 加载全球地形并切换到 3D 地球视图。
   * 高分辨率(2048×1024)数据仅用于渲染;求解网格降采样到 1024×512,
   * 保持 GPU 步进与峰值回读的开销不变。 */
  private async loadGlobe(): Promise<void> {
    try {
      const resp = await fetch('data/globe.tsunami');
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const buf = await resp.arrayBuffer();
      const region = parseTsunamiBinary(buf);
      this.regionMeta = region;
      this.regionBuf = buf;
      const hiW = region.width;
      const hiH = region.height;
      // 求解网格上限 1024×512(按偶数倍降采样,保持对齐)
      let k = 1;
      while (hiW / k > 1024 || hiH / k > 512) k *= 2;
      const sizeX = Math.floor(hiW / k);
      const sizeY = Math.floor(hiH / k);
      const grid = k === 1 ? region.grid : downsample(region.grid, hiW, hiH, k);
      const dxM = (kmPerDegLon(0) * 360 * 1000) / sizeX;
      const dyM = (GLOBE_LAT_SPAN * KM_PER_DEG_LAT * 1000) / sizeY;
      this.rebuild(
        grid, sizeX, sizeY, dxM, dyM,
        `3D 全球地形 ±84° (求解 ${sizeX}×${sizeY} · 渲染 ${hiW}×${hiH})`,
        'globe', GLOBE_DT,
        k === 1 ? undefined : { x: hiW, y: hiH },
        k === 1 ? undefined : region.grid
      );
    } catch (err) {
      alert(`加载全球地形失败：${err}\n请先运行 tools/fetch_bathy.py --tiles --globe 生成数据。`);
    }
  }

  /** 恢复程序化理想地形 */
  private resetProcedural(): void {
    this.regionMeta = null;
    this.regionBuf = null;
    const bathy = generateBathymetry();
    const dx = (DOMAIN_KM * 1000) / SIM_SIZE;
    this.rebuild(
      bathy, SIM_SIZE, SIM_SIZE, dx, dx,
      `程序化理想地形 ${DOMAIN_KM}×${DOMAIN_KM} km`
    );
  }

  // ---------------------------------------------------------------- 仿真控制

  private resetSea(): void {
    this.solver.reset();
    this.simSeconds = 0;
    this.resetObserverStats();
    this.updateStats(true);
  }

  /** 在指定 uv 位置触发地震并自动开始播放 */
  private quakeAt(u: number, v: number): void {
    this.stopGeoReplay();
    this.resetObserverStats();
    const amp = magnitudeToAmplitude(this.panel.params.magnitude);
    this.solver.inject(u, v, amp, this.panel.params.radiusKm * 1000);
    this.recordQuakeOrigin(u, v);
    this.panel.params.playing = true;
  }

  /** 记录发震时刻与震中(P/S 波圈从此刻开始按波速扩散) */
  private recordQuakeOrigin(u: number, v: number): void {
    this.quakeSimT = this.simSeconds;
    this.sceneApp.setSeismicOrigin(u, v);
  }

  /** 触发真实地震:按目录断层参数计算双极震源场并叠加注入 */
  private async triggerRealQuake(): Promise<void> {
    const q = findQuake(this.panel.params.quakeId);
    if (!q) return;
    await this.triggerCatalogQuake(q);
  }

  /** 历史目录与 USGS 实时目录共用的震源注入流程 */
  private async triggerCatalogQuake(q: CatalogEarthquake): Promise<void> {
    let domain: GeoDomain;
    if (this.viewMode === 'globe') {
      domain = { west: -180, east: 180, south: -84, north: 84,
        sizeX: this.solver.sizeX, sizeY: this.solver.sizeY };
    } else if (this.regionMeta) {
      const r = this.regionMeta;
      const inside =
        q.lon >= r.west && q.lon <= r.east && q.lat >= r.south && q.lat <= r.north;
      if (!inside) {
        const goGlobe = confirm(
          `震中 (${q.lon.toFixed(1)}°, ${q.lat.toFixed(1)}°) 不在当前区域内。\n切换到 3D 全球视图触发?`
        );
        if (goGlobe) {
          await this.loadGlobe();
          return this.triggerCatalogQuake(q);
        }
        return;
      }
      domain = { west: r.west, east: r.east, south: r.south, north: r.north,
        sizeX: this.solver.sizeX, sizeY: this.solver.sizeY };
    } else {
      const goGlobe = confirm(
        '真实地震需要地理坐标域。切换到 3D 全球视图触发?'
      );
      if (goGlobe) {
        await this.loadGlobe();
        return this.triggerCatalogQuake(q);
      }
      return;
    }

    const eta = quakeSurfaceEta(q, domain);
    if (!eta) {
      alert('该地震以走滑为主,海底垂直位移极小,几乎不激发海啸(这正是走滑型海啸弱的原因)。');
      return;
    }

    // 震中经纬度 → 当前域 uv(用于视角跳转与叉号标记)
    let eu: number;
    let ev: number;
    if (this.viewMode === 'globe') {
      eu = (q.lon + 180) / 360;
      eu -= Math.floor(eu);
      ev = (q.lat + 84) / GLOBE_LAT_SPAN;
    } else {
      const r = this.regionMeta!;
      eu = (q.lon - r.west) / (r.east - r.west);
      ev = (q.lat - r.south) / (r.north - r.south);
    }

    await this.runEpicenterSequence(eu, ev, () => {
      this.stopGeoReplay();
      this.resetObserverStats();
      this.solver.injectField(eta, domain.sizeX, domain.sizeY);
      this.recordQuakeOrigin(eu, ev);
      // 同步面板震级与色标
      this.panel.params.magnitude = q.mw;
      this.panel.refreshMagnitude();
      if (Math.abs(q.mw) >= 8.8 && this.panel.params.colorRange < 5) {
        this.panel.params.colorRange = 8;
        this.sceneApp.setColorRange(8);
        this.elLegendMin.textContent = '-8 m';
        this.elLegendMax.textContent = '+8 m';
      }
      this.panel.params.playing = true;
      this.updateStats(true);
    });
  }

  /** 震源序列:相机立即起飞(倒计时期间同步运镜)→ 震源叉号 →
   * 倒计时 3 秒 → 注入。序列进行中再次触发会被忽略。 */
  private async runEpicenterSequence(
    u: number, v: number, inject: () => void
  ): Promise<void> {
    if (this.quakeSeqBusy) return;
    this.quakeSeqBusy = true;
    try {
      this.sceneApp.focusUv(u, v);
      this.sceneApp.setEpicenterMarker({ u, v });
      await this.countdown(3);
      inject();
    } finally {
      this.quakeSeqBusy = false;
    }
  }

  /** 屏幕中央倒计时(每秒递减,结束隐藏) */
  private countdown(seconds: number): Promise<void> {
    return new Promise((resolve) => {
      let n = seconds;
      this.elCountdown.textContent = String(n);
      this.elCountdown.classList.add('show');
      const timer = window.setInterval(() => {
        n -= 1;
        if (n <= 0) {
          window.clearInterval(timer);
          this.elCountdown.classList.remove('show');
          resolve();
        } else {
          this.elCountdown.textContent = String(n);
        }
      }, 1000);
    });
  }

  // ---------------------------------------------------------------- USGS 实时目录

  /** 拉取 USGS 近 N 天全球地震并填充下拉列表 */
  private async fetchLiveQuakes(): Promise<void> {
    const p = this.panel.params;
    this.panel.setLiveStatus('拉取 USGS 中…');
    try {
      const list = await fetchRecentQuakes(p.usgsMinMag, p.usgsDays);
      this.liveQuakes = list;
      const opts: Record<string, string> = {};
      for (const q of list) {
        const d = new Date(q.time);
        const mmdd = `${String(d.getUTCMonth() + 1).padStart(2, '0')}-` +
          `${String(d.getUTCDate()).padStart(2, '0')}`;
        opts[`live-${q.id}`] = `M${q.mag.toFixed(1)} ${q.place} (${mmdd})`;
      }
      this.panel.setLiveOptions(opts, list.length ? `live-${list[0].id}` : '');
      this.panel.setLiveStatus(
        list.length ? `共 ${list.length} 条 · 近 ${p.usgsDays} 天 M≥${p.usgsMinMag}` : '无记录'
      );
      if (!list.length) {
        alert(`近 ${p.usgsDays} 天没有 M ≥ ${p.usgsMinMag} 的地震,可降低最小震级重试。`);
      }
    } catch (err) {
      this.panel.setLiveStatus('拉取失败');
      alert(`拉取 USGS 地震目录失败：${err}\n请检查网络连接。`);
    }
  }

  /** 触发选中的实时地震(断层参数为经验估算) */
  private async triggerLiveQuake(): Promise<void> {
    const sel = this.panel.params.liveQuakeId;
    const q = this.liveQuakes.find((x) => `live-${x.id}` === sel);
    if (!q) {
      alert('请先点击"拉取最新地震"。');
      return;
    }
    await this.triggerCatalogQuake(toLiveQuakeCatalog(q));
  }

  // ---------------------------------------------------------------- 观测点

  /** 按当前域重建观测点列表(自动吸附到最近的海洋单元) */
  private rebuildObservers(bathy: Float32Array): void {
    this.observers = [];
    const { sizeX, sizeY } = this.solver;
    const sites: { site: ObserverSite; u: number; v: number }[] = [];
    if (this.viewMode === 'globe') {
      for (const s of OBSERVER_SITES) {
        sites.push({ site: s, u: (s.lon + 180) / 360, v: (s.lat + 84) / GLOBE_LAT_SPAN });
      }
    } else if (this.regionMeta) {
      const r = this.regionMeta;
      for (const s of OBSERVER_SITES) {
        if (s.lon < r.west || s.lon > r.east || s.lat < r.south || s.lat > r.north) continue;
        sites.push({
          site: s,
          u: (s.lon - r.west) / (r.east - r.west),
          v: (s.lat - r.south) / (r.north - r.south),
        });
      }
    }
    for (const { site, u, v } of sites) {
      let i = Math.min(sizeX - 1, Math.floor(u * sizeX));
      let j = Math.min(sizeY - 1, Math.floor(v * sizeY));
      // 低分辨率下海岸站点可能落在陆地单元,向最近海洋单元偏移
      if (bathy[j * sizeX + i] >= -1) {
        [i, j] = this.snapToWater(bathy, sizeX, sizeY, i, j);
      }
      this.observers.push({
        site, u: (i + 0.5) / sizeX, v: (j + 0.5) / sizeY, arrivedAt: null, peak: 0,
      });
    }
    this.sceneApp.setObserverMarkers(this.observers.map((o) => ({ u: o.u, v: o.v })));
    this.renderObserverTable();
    this.elObserverPanel.style.display = this.observers.length ? '' : 'none';
  }

  /** 以 (i,j) 为中心向外逐圈搜索海洋单元(最远 24 格) */
  private snapToWater(
    bathy: Float32Array, sizeX: number, sizeY: number, i0: number, j0: number
  ): [number, number] {
    for (let rad = 1; rad <= 24; rad++) {
      for (let dj = -rad; dj <= rad; dj++) {
        for (let di = -rad; di <= rad; di++) {
          if (Math.max(Math.abs(di), Math.abs(dj)) !== rad) continue;
          const i = i0 + di;
          const j = j0 + dj;
          if (i < 0 || i >= sizeX || j < 0 || j >= sizeY) continue;
          if (bathy[j * sizeX + i] < -1) return [i, j];
        }
      }
    }
    return [i0, j0];
  }

  /** 新事件开始:清空到达时间与峰值统计 */
  private resetObserverStats(): void {
    for (const o of this.observers) {
      o.arrivedAt = null;
      o.peak = 0;
    }
    this.updateObserverRows();
  }

  private renderObserverTable(): void {
    let html = '';
    for (const o of this.observers) {
      html += `<tr><td class="ob-name">${o.site.name}</td>` +
        `<td class="ob-val ob-arr">—</td><td class="ob-val ob-pk">--</td></tr>`;
    }
    this.elObserverTable.innerHTML = html;
    this.observerRows = Array.from(
      this.elObserverTable.querySelectorAll('tr')
    ).map((tr) => ({
      arr: tr.querySelector('.ob-arr') as HTMLElement,
      pk: tr.querySelector('.ob-pk') as HTMLElement,
    }));
  }

  /** 每秒采样一次观测点波高(复用 readPeak 的回读缓冲,不额外回读) */
  private sampleObservers(): void {
    if (this.observers.length === 0 || !this.peakBuf) return;
    if (this.geoFrames.length > 0) return; // 回放模式时间不连续,不统计
    const { sizeX, sizeY } = this.solver;
    for (const o of this.observers) {
      const i = Math.min(sizeX - 1, Math.floor(o.u * sizeX));
      const j = Math.min(sizeY - 1, Math.floor(o.v * sizeY));
      const a = Math.abs(this.peakBuf[(j * sizeX + i) * 4]);
      if (this.panel.params.playing) {
        if (o.arrivedAt === null && a >= 0.05 && this.simSeconds > 0) {
          o.arrivedAt = this.simSeconds;
        }
      }
      if (a > o.peak) o.peak = a;
    }
    this.updateObserverRows();
  }

  private updateObserverRows(): void {
    for (let k = 0; k < this.observers.length; k++) {
      const o = this.observers[k];
      const row = this.observerRows[k];
      if (!row) continue;
      if (o.arrivedAt === null) {
        row.arr.textContent = '未到达';
      } else {
        const m = Math.floor(o.arrivedAt / 60);
        const s = Math.floor(o.arrivedAt % 60);
        row.arr.textContent = `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
      }
      row.pk.textContent = o.peak < 0.005 ? '--' : `${o.peak.toFixed(2)} m`;
    }
  }

  // ---------------------------------------------------------------- 科研模式

  /** 探测 GeoClaw 后端环境 */
  private async geoConnect(): Promise<void> {
    const url = this.panel.params.geoUrl.replace(/\/$/, '');
    this.panel.setGeoStatus('连接中…');
    try {
      const env = await checkHealth(url);
      this.panel.setGeoStatus(
        env.geoclaw
          ? `已连接 · clawpack ${env.clawpack_version}`
          : `已连接 · ${env.note || 'clawpack 未安装'}`
      );
    } catch (err) {
      this.panel.setGeoStatus('连接失败');
      alert(
        `无法连接 GeoClaw 后端：${err}\n` +
          `启动方法：python3 -m uvicorn server.app:app --port 8100`
      );
    }
  }

  /** 提交 GeoClaw 模拟任务 */
  private async geoSubmit(): Promise<void> {
    if (!this.regionMeta || !this.regionBuf) {
      alert('请先加载真实地形(如“东北海域真实地形”),再提交 GeoClaw 模拟。');
      return;
    }
    const url = this.panel.params.geoUrl.replace(/\/$/, '');
    const r = this.regionMeta;
    const p = this.panel.params;

    // 断层:优先使用"真实地震目录"选中地震的完整 Okada 参数
    let fault;
    const q = findQuake(p.quakeId);
    if (q) {
      const inside = q.lon >= r.west && q.lon <= r.east && q.lat >= r.south && q.lat <= r.north;
      if (!inside) {
        const go = confirm(
          `选中地震 ${q.year} ${q.name} 的震中不在当前区域内,` +
            `区域内可能看不到明显波形。仍要提交?`
        );
        if (!go) {
          this.panel.setGeoStatus('未连接');
          return;
        }
      }
      const L = q.lengthKm * 1000;
      const W = q.widthKm * 1000;
      const m0 = Math.pow(10, 1.5 * q.mw + 9.1);
      fault = {
        longitude: q.lon, latitude: q.lat,
        depth: q.depthKm * 1000, length: L, width: W,
        strike: q.strike, dip: q.dip, rake: q.rake,
        slip: m0 / (3e10 * L * W),
      };
      this.panel.setGeoStatus(`提交中·${q.year} ${q.name}…`);
    } else {
      fault = {
        longitude: (r.west + r.east) / 2,
        latitude: (r.south + r.north) / 2,
        slip: magnitudeToSlip(p.magnitude),
      };
      this.panel.setGeoStatus('提交中…');
    }

    try {
      const resp = await submitJob(url, {
        sim_time: p.geoSimMinutes * 60,
        nframes: p.geoFrames,
        fault,
        bathy_b64: bufToBase64(this.regionBuf),
      });
      this.geoJobId = resp.job_id;
      this.panel.setGeoStatus('运行中…');
      this.geoPoll(url);
    } catch (err) {
      this.panel.setGeoStatus('提交失败');
      alert(`提交失败：${err}`);
    }
  }

  /** 轮询任务状态,完成后下载全部帧 */
  private geoPoll(url: string): void {
    if (this.geoPollTimer) window.clearInterval(this.geoPollTimer);
    this.geoPollTimer = window.setInterval(async () => {
      try {
        const st = await jobStatus(url, this.geoJobId!);
        if (st.status === 'running' || st.status === 'queued') {
          this.panel.setGeoStatus('GeoClaw 运行中…');
        } else if (st.status === 'error') {
          window.clearInterval(this.geoPollTimer!);
          this.geoPollTimer = null;
          this.panel.setGeoStatus('任务失败');
          alert(`GeoClaw 任务失败：\n${st.error}`);
        } else {
          window.clearInterval(this.geoPollTimer!);
          this.geoPollTimer = null;
          await this.geoDownloadFrames(url);
        }
      } catch (err) {
        window.clearInterval(this.geoPollTimer!);
        this.geoPollTimer = null;
        this.panel.setGeoStatus('状态查询失败');
        alert(`任务状态查询失败：${err}`);
      }
    }, 2000);
  }

  private async geoDownloadFrames(url: string): Promise<void> {
    const metas = await fetchFrames(url, this.geoJobId!);
    this.geoFrames = [];
    for (let i = 0; i < metas.length; i++) {
      this.panel.setGeoStatus(`下载帧 ${i + 1}/${metas.length}…`);
      const eta = await fetchFrame(url, this.geoJobId!, metas[i]);
      this.geoFrames.push({ meta: metas[i], eta });
    }
    this.geoFrameIdx = 0;
    this.panel.params.playing = false; // 暂停实时模拟
    this.showGeoFrame();
    if (this.panel.params.geoAutoPlay) this.startGeoAutoplay();
  }

  /** 回放下一帧(循环) */
  private geoNextFrame(): void {
    if (this.geoFrames.length === 0) return;
    this.geoFrameIdx = (this.geoFrameIdx + 1) % this.geoFrames.length;
    this.showGeoFrame();
  }

  private showGeoFrame(): void {
    const f = this.geoFrames[this.geoFrameIdx];
    if (!f) return;
    this.solver.loadEta(f.eta, f.meta.nx, f.meta.ny);
    this.simSeconds = f.meta.time;
    const min = (f.meta.time / 60).toFixed(0);
    this.panel.setGeoStatus(
      `回放 ${this.geoFrameIdx + 1}/${this.geoFrames.length} · t=${min} min`
    );
    this.updateStats(true);
  }

  private startGeoAutoplay(): void {
    this.stopGeoAutoplay();
    this.geoPlayTimer = window.setInterval(() => this.geoNextFrame(), 600);
  }

  private stopGeoAutoplay(): void {
    if (this.geoPlayTimer) {
      window.clearInterval(this.geoPlayTimer);
      this.geoPlayTimer = null;
    }
  }

  /** 退出回放(切换地形/触发新地震时调用) */
  private stopGeoReplay(): void {
    this.stopGeoAutoplay();
    this.geoFrames = [];
    this.geoFrameIdx = 0;
  }

  private loop = (): void => {
    requestAnimationFrame(this.loop);

    const now = performance.now();
    const delta = Math.min((now - this.lastFrameTime) / 1000, 0.1);
    this.lastFrameTime = now;

    if (this.panel.params.playing) {
      // 步频 = 时间倍速 / dt(真实时间语义),与显示器帧率无关
      this.stepAcc += delta * (this.panel.params.timeScale / this.solver.dtSeconds);
      const n = Math.min(Math.floor(this.stepAcc), 60);
      this.stepAcc -= n;
      if (this.stepAcc > 60) this.stepAcc = 0; // 防卡帧后追步雪崩
      if (n > 0) {
        this.solver.step(n);
        this.simSeconds += n * this.solver.dtSeconds;
      }
    } else {
      this.stepAcc = 0;
    }

    this.sceneApp.syncState(this.solver);
    // P/S 波圈时间 = 发震后经过的模拟时间(回放模式下不连续,保持冻结)
    if (this.quakeSimT !== null && this.geoFrames.length === 0) {
      this.sceneApp.setSeismicTime(this.simSeconds - this.quakeSimT);
    }
    this.sceneApp.render();

    this.frames++;
    if (now - this.fpsTimer >= 1000) {
      this.elFps.textContent = `${this.frames} fps`;
      this.frames = 0;
      this.fpsTimer = now;
      this.updateStats(false);
    }
  };

  /** 更新 HUD 统计(峰值波高需要 GPU 回读,每秒一次) */
  private updateStats(forcePeak: boolean): void {
    const m = Math.floor(this.simSeconds / 60);
    const s = Math.floor(this.simSeconds % 60);
    this.elTime.textContent = `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
    if (forcePeak || this.frames === 0) {
      const need = this.solver.sizeX * this.solver.sizeY * 4;
      if (!this.peakBuf || this.peakBuf.length !== need) {
        this.peakBuf = new Float32Array(need);
      }
      const peak = this.solver.readPeak(this.peakBuf);
      this.elPeak.textContent = peak < 0.005 ? '--' : `${peak.toFixed(2)} m`;
      this.sampleObservers();
    }
  }
}

/** 按 k×k 块均值降采样高程网格(k 为整数倍率) */
function downsample(
  src: Float32Array, w: number, h: number, k: number
): Float32Array {
  const w2 = Math.floor(w / k);
  const h2 = Math.floor(h / k);
  const out = new Float32Array(w2 * h2);
  const inv = 1 / (k * k);
  for (let j = 0; j < h2; j++) {
    for (let i = 0; i < w2; i++) {
      let sum = 0;
      for (let dj = 0; dj < k; dj++) {
        const row = (j * k + dj) * w + i * k;
        for (let di = 0; di < k; di++) sum += src[row + di];
      }
      out[j * w2 + i] = sum * inv;
    }
  }
  return out;
}

/** ArrayBuffer → base64(分块编码,避免超出 String.fromCharCode 参数上限) */
function bufToBase64(buf: ArrayBuffer): string {
  const bytes = new Uint8Array(buf);
  let bin = '';
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    bin += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  }
  return btoa(bin);
}
