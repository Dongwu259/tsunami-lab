/**
 * 全部 GLSL 着色器。
 *
 * 物理模型:线性浅水方程(Linear Shallow Water Equations)
 *   ∂η/∂t  = -∇·(H·u)            连续性方程
 *   ∂(H·u)/∂t = -g·H·∇η          动量方程
 * 状态纹理:r = η(海面位移 m),g = Hu,b = Hv(深度积分通量 m²/s)。
 * 数值格式(uScheme 切换):
 *   0 = 一阶 Lax–Friedrichs(教学对照,耗散大);
 *   1 = MUSCL 重构 + minmod 限制器 + Rusanov 界面通量 + SSP-RK2(二阶低耗散)。
 * 震源注入为独立 pass(INJECT_FRAG),与步进格式解耦。
 */

/** 单步推进着色器(LF 格式;手动 ping-pong 下 uState 需显式声明) */
export const STEP_FRAG = /* glsl */ `
uniform sampler2D uState;        // 当前状态(RK2 中为 U0)
uniform sampler2D uBathymetry;   // 海床高程(m,负值在海面以下)
uniform vec2 uTexel;
uniform vec2 uRes;               // 网格尺寸(像素)
uniform float uDx;               // x 方向网格边长(m)
uniform float uDy;               // y 方向网格边长(m)
uniform float uDt;               // 时间步长(s)
uniform float uG;                // 重力加速度
uniform float uDamping;          // 底摩擦阻尼系数
uniform float uGlobeMode;        // 1 = 全球球面(经向周期 + 纬度度量)

void main() {
  vec2 uv = gl_FragCoord.xy / uRes;
  float tx = uTexel.x;
  float ty = uTexel.y;

  vec4 c = texture2D(uState, uv);
  vec4 l = texture2D(uState, uv + vec2(-tx, 0.0));
  vec4 r = texture2D(uState, uv + vec2( tx, 0.0));
  vec4 d = texture2D(uState, uv + vec2(0.0, -ty));
  vec4 u = texture2D(uState, uv + vec2(0.0,  ty));

  float bed = texture2D(uBathymetry, uv).r;
  float H   = max(-bed, 0.0);          // 静水深度
  float wet = step(1.0, H);            // 水深 > 1 m 视为水域

  // 球面度量:经向格距随纬度收缩。
  // 二维 LF 稳定条件是两方向库朗数之和 νx+νy ≤ 1;纬向 νy ≈ 0.39,
  // 故极地下限取 2.2·dt·√(gH),使极点处 νx ≤ 1/2.2 ≈ 0.45,总和 < 0.85。
  // 代价:高纬有效格距变大,波速略失真(教学级可接受)
  float lat = (uv.y - 0.5) * 168.0;    // 全球域纬度范围 ±84°
  float cflFloor = uDt * sqrt(uG * max(H, 0.0)) * 2.2 + 100.0;
  float dxEff = uGlobeMode > 0.5
      ? max(uDx * cos(max(abs(lat), 5.0) * 0.017453293), cflFloor)
      : uDx;

  // Lax–Friedrichs 邻域平均,保证显式格式稳定
  vec4 avg = 0.25 * (l + r + d + u);

  // 连续性方程:η_t = -div(Hu)
  float etaNew = avg.r - uDt * ((r.g - l.g) / (2.0 * dxEff) + (u.b - d.b) / (2.0 * uDy));

  // 动量方程:(Hu)_t = -g·H·η_x
  float deta_dx = (r.r - l.r) / (2.0 * dxEff);
  float deta_dy = (u.r - d.r) / (2.0 * uDy);
  float huNew = avg.g - uDt * uG * H * deta_dx;
  float hvNew = avg.b - uDt * uG * H * deta_dy;

  // 底摩擦阻尼(曼宁摩擦的教学级近似)
  huNew *= uDamping;
  hvNew *= uDamping;

  // 边界海绵:平面域四边吸收;球面仅两极吸收(经向周期环绕)
  float edge = uGlobeMode > 0.5
      ? min(uv.y, 1.0 - uv.y)
      : min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
  float spongeWidth = uGlobeMode > 0.5 ? 0.03 : 0.06;
  float sponge = smoothstep(0.0, spongeWidth, edge);
  huNew *= sponge;
  hvNew *= sponge;

  // 干单元(陆地):无通量,η 保持不变
  huNew *= wet;
  hvNew *= wet;
  etaNew = mix(c.r, etaNew, wet);

  gl_FragColor = vec4(etaNew, huNew, hvNew, 1.0);
}
`;

/** MUSCL 界面通量公共代码段(与 cpuSolver.ts fluxX/fluxY 同构)。
 * minmod 限制器 + Rusanov(局部 Lax–Friedrichs)通量,非线性总水深 h=H+η,
 * 波速 s = √(g·max(hL,hR));干床(h<uHMin)界面通量置零。
 * 非线性浅水通量:Fx = (hu, g·h·η, 0),Fy = (hv, 0, g·h·η)。 */
const FLUX_GLSL = /* glsl */ `
uniform sampler2D uState;
uniform sampler2D uBathymetry;
uniform vec2 uRes;
uniform vec2 uTexel;
uniform float uDx;
uniform float uDy;
uniform float uDt;
uniform float uG;
uniform float uGlobeMode;
uniform float uHMin;

float minmod(float a, float b) {
  // 同号取绝对值较小者,异号(或含零)返回 0(与 cpuSolver.minmod 同构)
  return (a * b <= 0.0) ? 0.0 : sign(a) * min(abs(a), abs(b));
}

// 网格索引:经向 x 周期(globe)或钳制(plane),纬向 y 始终钳制(与 CPU ix/iy 同构)
ivec2 cellIdx(int i, int j) {
  float nx = uRes.x;
  float ny = uRes.y;
  float fx = uGlobeMode > 0.5 ? mod(float(i), nx) : clamp(float(i), 0.0, nx - 1.0);
  float fy = clamp(float(j), 0.0, ny - 1.0);
  return ivec2(int(fx), int(fy));
}

vec4 cellState(ivec2 c) { return texture2D(uState, (vec2(c) + 0.5) * uTexel); }
float cellDepth(ivec2 c) { return max(-texture2D(uBathymetry, (vec2(c) + 0.5) * uTexel).r, 0.0); }

// x 方向界面 (i+1/2, 行 j) 通量:入参 q=(i,j),与 CPU fluxX(i,j) 同构
vec3 musclFluxX(ivec2 q) {
  vec4 A = cellState(cellIdx(q.x - 1, q.y));
  vec4 B = cellState(cellIdx(q.x,     q.y));
  vec4 C = cellState(cellIdx(q.x + 1, q.y));
  vec4 D = cellState(cellIdx(q.x + 2, q.y));
  float HB = cellDepth(cellIdx(q.x,     q.y));
  float HC = cellDepth(cellIdx(q.x + 1, q.y));
  vec3 sL = vec3(minmod(B.r - A.r, C.r - B.r), minmod(B.g - A.g, C.g - B.g), minmod(B.b - A.b, C.b - B.b));
  vec3 sR = vec3(minmod(C.r - B.r, D.r - C.r), minmod(C.g - B.g, D.g - C.g), minmod(C.b - B.b, D.b - C.b));
  float eL = B.r + 0.5 * sL.x, eR = C.r - 0.5 * sR.x;
  float uL = B.g + 0.5 * sL.y, uR = C.g - 0.5 * sR.y;
  float vL = B.b + 0.5 * sL.z, vR = C.b - 0.5 * sR.z;
  // 非线性总水深 h=H+η;干床(h<uHMin)界面通量置零(与 cpuSolver.fluxX 同构)
  float hL = HB + eL, hR = HC + eR;
  if (hL < uHMin || hR < uHMin) return vec3(0.0);
  float s = sqrt(uG * max(hL, hR));
  return vec3(
    0.5 * (uL + uR) - 0.5 * s * (eR - eL),
    0.5 * uG * (hL * eL + hR * eR) - 0.5 * s * (uR - uL),
    -0.5 * s * (vR - vL));
}

// y 方向界面 (列 i, j+1/2) 通量:入参 q=(i,j),与 CPU fluxY(i,j) 同构
vec3 musclFluxY(ivec2 q) {
  vec4 A = cellState(cellIdx(q.x, q.y - 1));
  vec4 B = cellState(cellIdx(q.x, q.y));
  vec4 C = cellState(cellIdx(q.x, q.y + 1));
  vec4 D = cellState(cellIdx(q.x, q.y + 2));
  float HB = cellDepth(cellIdx(q.x, q.y));
  float HC = cellDepth(cellIdx(q.x, q.y + 1));
  vec3 sL = vec3(minmod(B.r - A.r, C.r - B.r), minmod(B.g - A.g, C.g - B.g), minmod(B.b - A.b, C.b - B.b));
  vec3 sR = vec3(minmod(C.r - B.r, D.r - C.r), minmod(C.g - B.g, D.g - C.g), minmod(C.b - B.b, D.b - C.b));
  float eL = B.r + 0.5 * sL.x, eR = C.r - 0.5 * sR.x;
  float uL = B.g + 0.5 * sL.y, uR = C.g - 0.5 * sR.y;
  float vL = B.b + 0.5 * sL.z, vR = C.b - 0.5 * sR.z;
  // 非线性总水深 h=H+η;干床(h<uHMin)界面通量置零(与 cpuSolver.fluxY 同构)
  float hL = HB + eL, hR = HC + eR;
  if (hL < uHMin || hR < uHMin) return vec3(0.0);
  float s = sqrt(uG * max(hL, hR));
  return vec3(
    0.5 * (vL + vR) - 0.5 * s * (eR - eL),
    -0.5 * s * (uR - uL),
    0.5 * uG * (hL * eL + hR * eR) - 0.5 * s * (vR - vL));
}
`;

/** RK2 推进着色器(uScheme=1)。uStage 切换两个 SSP-RK2 阶段:
 *   0: U* = U + dt·L(U)              (uState = U0)
 *   1: U' = ½U0 + ½(U* + dt·L(U*))   (uState = U*,u0 = U0)
 * 非线性总水深 h=H+η + 井平衡源项;修改项(海绵·曼宁摩擦·干单元)按算子分裂
 * 在阶段 1 合并后一次性应用,与 cpuSolver.stepV2 同构。 */
export const STEP_FRAG_V2 = /* glsl */ `
${FLUX_GLSL}
uniform float uStage;
uniform float uManning;   // 曼宁摩擦系数 n(0 关闭)
uniform float uDiag;      // DEV 诊断:1 = 直接输出 L 算子(不推进)
uniform sampler2D u0;   // 阶段 B 的 RK2 基态 U0

void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec2 uv = gl_FragCoord.xy / uRes;

  float bed = texture2D(uBathymetry, uv).r;
  float H = max(-bed, 0.0);
  float wet = step(uHMin, H);   // 1 = 水域(静水深 ≥ hMin),0 = 永久陆地

  // 球面度量:经向格距随纬度收缩 + 极地 CFL 下限保护(与 LF/CPU 镜像同构)
  float lat = (uv.y - 0.5) * 168.0;
  float cflFloor = uDt * sqrt(uG * H) * 2.2 + 100.0;
  float dxEff = uGlobeMode > 0.5
      ? max(uDx * cos(max(abs(lat), 5.0) * 0.017453293), cflFloor)
      : uDx;

  // 单元 p 的四界面通量:musclFluxX(q)=界面 q.x+1/2,musclFluxY(q)=界面 q.y+1/2。
  // 故单元 p 的左/右界面用 q=(p.x-1,p.y)/(p.x,p.y),下/上界面用 q=(p.x,p.y-1)/(p.x,p.y)。
  vec3 fxm = musclFluxX(ivec2(p.x - 1, p.y));
  vec3 fxp = musclFluxX(ivec2(p.x,     p.y));
  vec3 fym = musclFluxY(ivec2(p.x, p.y - 1));
  vec3 fyp = musclFluxY(ivec2(p.x, p.y));
  vec3 L = -vec3(
    (fxp.x - fxm.x) / dxEff + (fyp.x - fym.x) / uDy,
    (fxp.y - fxm.y) / dxEff + (fyp.y - fym.y) / uDy,
    (fxp.z - fxm.z) / dxEff + (fyp.z - fym.z) / uDy);

  // 井平衡源项 g·η·∂h/∂x(h=H+η):抵消 g·h·η 通量多出的地形项,使动量方程
  // 精确回到 -g·h·∂η/∂x;η=0 时源项为 0 → 严格静水平衡(与 cpuSolver.computeL 同构)
  float etaC = texture2D(uState, uv).r;
  float hIp = cellDepth(cellIdx(p.x + 1, p.y)) + cellState(cellIdx(p.x + 1, p.y)).r;
  float hIm = cellDepth(cellIdx(p.x - 1, p.y)) + cellState(cellIdx(p.x - 1, p.y)).r;
  float hJp = cellDepth(cellIdx(p.x, p.y + 1)) + cellState(cellIdx(p.x, p.y + 1)).r;
  float hJm = cellDepth(cellIdx(p.x, p.y - 1)) + cellState(cellIdx(p.x, p.y - 1)).r;
  L.y += uG * etaC * (hIp - hIm) / (2.0 * dxEff);
  L.z += uG * etaC * (hJp - hJm) / (2.0 * uDy);

  if (uDiag > 0.5) {
    // 诊断:r=Lη·1e3, g=Lhu, b=Lhv, a=fxp.y-fxm.y(压力通量差)
    gl_FragColor = vec4(L.x * 1000.0, L.y, L.z, fxp.y - fxm.y);
    return;
  }

  vec4 outState;
  if (uStage < 0.5) {
    // 阶段 A:uState = U0
    outState = vec4(texture2D(uState, uv).rgb + uDt * L, 1.0);
  } else {
    // 阶段 B:uState = U*,u0 = U0
    vec4 U0 = texture2D(u0, uv);
    vec4 Us = texture2D(uState, uv);
    outState = vec4(0.5 * U0.rgb + 0.5 * (Us.rgb + uDt * L), 1.0);
    // 边界海绵(仅动量):平面域四边吸收;球面仅两极吸收
    float edge = uGlobeMode > 0.5
        ? min(uv.y, 1.0 - uv.y)
        : min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
    float spongeWidth = uGlobeMode > 0.5 ? 0.03 : 0.06;
    float sponge = smoothstep(0.0, spongeWidth, edge);
    outState.g *= sponge;
    outState.b *= sponge;
    // 隐式曼宁摩擦:hu /= 1 + dt·g·n²·|u|/h^(4/3),|u| = √(u²+v²)/h
    float h = max(H + outState.r, uHMin);
    float speed = length(outState.gb) / h;
    float cf = uDt * uG * uManning * uManning * speed / pow(h, 4.0 / 3.0);
    outState.g /= 1.0 + cf;
    outState.b /= 1.0 + cf;
    // 永久陆地(H < hMin):动量清零、η 冻结(干界面通量已置零 → 质量守恒)
    outState.g *= wet;
    outState.b *= wet;
    outState.r = mix(U0.r, outState.r, wet);
  }
  gl_FragColor = outState;
}
`;

/** 震源注入 pass(独立于步进格式):η += 高斯型海底抬升。
 * 读 uState(当前)、写另一缓冲后交换,避免自读自写。 */
export const INJECT_FRAG = /* glsl */ `
uniform sampler2D uState;
uniform vec2 uRes;
uniform vec3  uSource;           // xy = 震中 uv,z = 半径(uv 单位)
uniform float uSourceAmp;        // 初始波幅(m)
uniform float uGlobeMode;

void main() {
  vec2 uv = gl_FragCoord.xy / uRes;
  vec4 c = texture2D(uState, uv);
  vec2 dv = uv - uSource.xy;
  if (uGlobeMode > 0.5) {
    // 经向环绕:取最短跨缝距离,保证 180° 经线两侧震源连续
    dv.x -= floor(dv.x + 0.5);
    dv.y *= 0.4667;   // 纬度范围 168°:把 v 距离折算成等效经度距离
  }
  float rr = uSource.z * uSource.z;
  c.r += uSourceAmp * exp(-dot(dv, dv) / rr);
  gl_FragColor = c;
}
`;

/** 海面顶点着色器:按状态纹理位移网格并计算法线 */
export const WATER_VERT = /* glsl */ `
uniform sampler2D uState;
uniform vec2  uTexel;
uniform float uWaveScale;    // 显示比例:m → 场景单位(km)
uniform vec2  uDomainKm;     // 域尺度(km)

varying vec2  vUv;
varying float vEta;
varying vec3  vNormal;

void main() {
  vUv = uv;

  float eta = texture2D(uState, uv).r;
  float eL  = texture2D(uState, uv + vec2(-uTexel.x, 0.0)).r;
  float eR  = texture2D(uState, uv + vec2( uTexel.x, 0.0)).r;
  float eD  = texture2D(uState, uv + vec2(0.0, -uTexel.y)).r;
  float eU  = texture2D(uState, uv + vec2(0.0,  uTexel.y)).r;

  vEta = eta;

  vec2 cellKm = uDomainKm * uTexel;
  vec3 n = normalize(vec3(
    (eL - eR) * uWaveScale / (2.0 * cellKm.x),
    (eD - eU) * uWaveScale / (2.0 * cellKm.y),
    1.0
  ));
  vNormal = normalMatrix * n;

  vec3 pos = position;
  pos.z += eta * uWaveScale;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
}
`;

/** 地震体波(P/S 波圈)叠加代码段,供海面与地形片元着色器共用。
 * P 波 6.0 km/s(白青),S 波 3.5 km/s(橙红),环半径 = 波速 × 发震后模拟时间;
 * 波圈越过对拓点(约 20000 km)后逐渐淡出。球面用大圆弧距,平面用域尺度换算。 */
export const SEISMIC_GLSL = /* glsl */ `
uniform float uSeisOn;     // 1 = 显示 P/S 波圈
uniform float uSeisT;      // 发震后经过的模拟时间(s)
uniform vec2  uEpi;        // 震中 uv
uniform float uEpiGlobe;   // 1 = 球面大圆弧距;0 = 平面域 km
uniform vec2  uSeisDomKm;  // 平面域尺度(km)

vec3 seisDir(vec2 suv) {
  float phi = 6.2831853 * suv.x;
  float theta = 3.1415926 * (1.0 - suv.y);
  float st = sin(theta);
  return vec3(-cos(phi) * st, cos(theta), sin(phi) * st);
}

float seisDistKm(vec2 suv, vec2 epi) {
  if (uEpiGlobe > 0.5) {
    float c = clamp(dot(seisDir(suv), seisDir(epi)), -1.0, 1.0);
    return 6371.0 * acos(c);
  }
  vec2 dv = suv - epi;
  return length(dv * uSeisDomKm);
}

float seisRing(float distKm, float radiusKm) {
  float w = max(60.0, radiusKm * 0.05);   // 环宽随半径适度展宽
  return smoothstep(w, 0.0, abs(distKm - radiusKm));
}

vec3 seismicOverlay(vec3 col, vec2 suv) {
  if (uSeisOn < 0.5 || uSeisT <= 0.0) return col;
  float distKm = seisDistKm(suv, uEpi);
  float rP = uSeisT * 6.0;
  float rS = uSeisT * 3.5;
  // P 波达对拓点后淡出,S 波随后淡出
  float fade = 1.0 - smoothstep(19000.0, 26000.0, rP);
  float rPw = seisRing(distKm, rP);
  float rSw = seisRing(distKm, rS);
  col += vec3(0.75, 0.95, 1.0) * rPw * 0.85 * fade;
  col += vec3(1.0, 0.55, 0.18) * rSw * 0.9 * fade;
  // 震源初始闪光(前 60 s)
  float flash = exp(-uSeisT / 25.0) * smoothstep(400.0, 0.0, distKm);
  col += vec3(1.0, 0.9, 0.6) * flash;
  return col;
}
`;

/** 海面片元着色器:按波高着色(波谷深蓝 → 平静青蓝 → 波峰红橙) */
export const WATER_FRAG = /* glsl */ `
uniform float uColorScale;   // 1 / 色标半量程(m)
uniform float uOpacity;      // 海面不透明度(0–1),调低可透视海底地形

varying vec2  vUv;
varying float vEta;
varying vec3  vNormal;

${SEISMIC_GLSL}

vec3 colormap(float t) {
  vec3 trough = vec3(0.02, 0.16, 0.45);
  vec3 calm   = vec3(0.05, 0.45, 0.65);
  vec3 crest  = vec3(0.98, 0.36, 0.12);
  if (t < 0.0) return mix(calm, trough, clamp(-t, 0.0, 1.0));
  return mix(calm, crest, clamp(t, 0.0, 1.0));
}

void main() {
  float t = clamp(vEta * uColorScale, -1.0, 1.0);
  vec3 col = colormap(t);

  float diff = clamp(dot(normalize(vNormal), normalize(vec3(0.4, 0.6, 0.8))), 0.0, 1.0);
  col *= 0.55 + 0.55 * diff;

  col = seismicOverlay(col, vUv);

  gl_FragColor = vec4(col, uOpacity);
}
`;

/** 地形顶点着色器:按海床纹理位移网格 */
export const TERRAIN_VERT = /* glsl */ `
uniform sampler2D uBathymetry;
uniform vec2  uTexel;
uniform float uTerrainScale; // 显示比例:m → 场景单位(km)
uniform vec2  uDomainKm;     // 域尺度(km)

varying float vBed;
varying vec3  vNormal;
varying vec2  vUv;

void main() {
  vUv = uv;
  float bed = texture2D(uBathymetry, uv).r;
  float bL  = texture2D(uBathymetry, uv + vec2(-uTexel.x, 0.0)).r;
  float bR  = texture2D(uBathymetry, uv + vec2( uTexel.x, 0.0)).r;
  float bD  = texture2D(uBathymetry, uv + vec2(0.0, -uTexel.y)).r;
  float bU  = texture2D(uBathymetry, uv + vec2(0.0,  uTexel.y)).r;

  vBed = bed;

  vec2 cellKm = uDomainKm * uTexel;
  vec3 n = normalize(vec3(
    (bL - bR) * uTerrainScale / (2.0 * cellKm.x),
    (bD - bU) * uTerrainScale / (2.0 * cellKm.y),
    1.0
  ));
  vNormal = normalMatrix * n;

  vec3 pos = position;
  pos.z += bed * uTerrainScale;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
}
`;

/** 地形片元着色器:按高程着色(深海墨蓝 → 浅海青 → 陆地绿棕) */
export const TERRAIN_FRAG = /* glsl */ `
varying float vBed;
varying vec3  vNormal;
varying vec2  vUv;

${SEISMIC_GLSL}

void main() {
  vec3 col;
  if (vBed < 0.0) {
    float d = clamp(-vBed / 4000.0, 0.0, 1.0);
    col = mix(vec3(0.10, 0.30, 0.38), vec3(0.01, 0.04, 0.10), pow(d, 0.5));
  } else {
    float h = clamp(vBed / 180.0, 0.0, 1.0);
    col = mix(vec3(0.24, 0.42, 0.24), vec3(0.52, 0.45, 0.34), h);
  }

  float diff = clamp(dot(normalize(vNormal), normalize(vec3(0.4, 0.6, 0.8))), 0.0, 1.0);
  col *= 0.5 + 0.6 * diff;

  col = seismicOverlay(col, vUv);

  gl_FragColor = vec4(col, 1.0);
}
`;

// ================================================================ 球面(3D 地球)

/** 球面海面顶点着色器:沿径向位移并按 η 梯度扰动法线。
 * 东向 = normalize(up×n),北向 = n×东;u 跨 2π 弧度,v 跨 168°(2.932 弧度)。 */
export const GLOBE_WATER_VERT = /* glsl */ `
uniform sampler2D uState;
uniform vec2  uTexel;
uniform float uWaveScale;    // m → 场景单位

varying vec2  vUv;
varying float vEta;
varying vec3  vNormal;

vec3 sphereEastW(vec3 n) {
  vec3 e = cross(vec3(0.0, 1.0, 0.0), n);
  return normalize(e + vec3(1e-6, 0.0, 1e-6));
}

void main() {
  vUv = uv;
  float eta = texture2D(uState, uv).r;
  float eL  = texture2D(uState, uv + vec2(-uTexel.x, 0.0)).r;
  float eR  = texture2D(uState, uv + vec2( uTexel.x, 0.0)).r;
  float eD  = texture2D(uState, uv + vec2(0.0, -uTexel.y)).r;
  float eU  = texture2D(uState, uv + vec2(0.0,  uTexel.y)).r;
  vEta = eta;

  vec3 n = normalize(position);
  vec3 east  = sphereEastW(n);
  vec3 north = cross(n, east);

  // 切向坡度(弧度制,与球半径无关)
  float slopeE = (eR - eL) * uWaveScale / (2.0 * uTexel.x * 6.2831853);
  float slopeN = (eU - eD) * uWaveScale / (2.0 * uTexel.y * 2.9321532);
  vec3 nDef = normalize(n - east * slopeE - north * slopeN);
  vNormal = normalMatrix * nDef;

  vec3 pos = position + n * (eta * uWaveScale);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
}
`;

/** 球面地形顶点着色器:按海床高程径向位移(陆地凸出、海盆凹陷) */
export const GLOBE_TERRAIN_VERT = /* glsl */ `
uniform sampler2D uBathymetry;
uniform vec2  uTexel;
uniform float uTerrainScale; // m → 场景单位
uniform float uLandLift;     // 陆地径向抬升(场景单位),防水面球遮挡低海岸

varying float vBed;
varying vec3  vNormal;
varying vec2  vUv;

vec3 sphereEastT(vec3 n) {
  vec3 e = cross(vec3(0.0, 1.0, 0.0), n);
  return normalize(e + vec3(1e-6, 0.0, 1e-6));
}

void main() {
  vUv = uv;
  float bed = texture2D(uBathymetry, uv).r;
  float bL  = texture2D(uBathymetry, uv + vec2(-uTexel.x, 0.0)).r;
  float bR  = texture2D(uBathymetry, uv + vec2( uTexel.x, 0.0)).r;
  float bD  = texture2D(uBathymetry, uv + vec2(0.0, -uTexel.y)).r;
  float bU  = texture2D(uBathymetry, uv + vec2(0.0,  uTexel.y)).r;
  vBed = bed;

  vec3 n = normalize(position);
  vec3 east  = sphereEastT(n);
  vec3 north = cross(n, east);

  float slopeE = (bR - bL) * uTerrainScale / (2.0 * uTexel.x * 6.2831853);
  float slopeN = (bU - bD) * uTerrainScale / (2.0 * uTexel.y * 2.9321532);
  vec3 nDef = normalize(n - east * slopeE - north * slopeN);
  vNormal = normalMatrix * nDef;

  // 陆地加微小径向抬升:低海拔海岸(高程×缩放 < 球壳半径差)若不加抬升
  // 会被水面球遮挡并与之 z-fighting,表现为大陆碎裂/缺块
  float disp = bed * uTerrainScale + step(0.0, bed) * uLandLift;
  vec3 pos = position + n * disp;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
}
`;
