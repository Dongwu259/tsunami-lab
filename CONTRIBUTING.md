# 贡献与开发指南(CONTRIBUTING)

> 面向**人类开发者与 AI agent** 的接手文档。读完本文即可安全地修改代码、验证、提交与发布。
> 项目定位、物理模型与功能说明见 [README.md](README.md);数值格式的分阶段升级计划见 [ROADMAP.md](ROADMAP.md)。

---

## 1. 环境与常用命令

- **运行时**:Node.js **22+**(CI 基准版本)、npm。前端全 MIT 技术栈:Vite 8 + TypeScript 5(ES2020,`strict`)+ Three.js 0.160 + lil-gui + vitest 5。
- **可选后端**:GeoClaw 科研模式需 Python 3 + gfortran + clawpack(见 README「科研模式」)。

| 命令 | 作用 |
|------|------|
| `npm install` | 安装依赖 |
| `npm run dev` | 启动开发服务器(Vite,默认 http://localhost:5173) |
| `npm run check` | 类型检查(`tsc --noEmit`) |
| `npm test` | 运行单元测试与数值基准(`vitest run`) |
| `npm run build` | 类型检查 + 生产构建(`tsc && vite build`) |
| `npm run preview` | 预览生产构建 |

> **提交前必须三步全绿**:`npm run check && npm test && npm run build`(与 CI 门禁一致,见 §5)。

构建细节:`vite.config.ts` 的 `copy-bathy-data` 插件在 `build` 后把随包地形 `data/` 拷入 `dist/data/`(`data/` 非 `public/`,dev 模式由 Vite 直接从根目录提供)。TypeScript `include` 仅 `src`。

---

## 2. 代码结构速览

完整逐文件地图见 README「架构」章节。数值核心集中在 `src/simulation/`:

```
src/simulation/
  cpuSolver.ts        GPU 格式的 CPU/TS 镜像(float64,纯函数,CI 无 WebGL 测试的真理来源)
  shaders.ts          全部 GLSL:STEP_FRAG(LF)、STEP_FRAG_V2(V2)、INJECT_FRAG、渲染着色器
  TsunamiSolver.ts    GPU ping-pong 求解器(手动 3 个 WebGLRenderTarget,RK2 中间级暂存)
  benchmarks.test.ts  数值基准(衰减/收敛/守恒/Green/辐射反射/极地波速/球面稳定)
  RealBathymetry.ts   .tsunami 二进制解析 + 经纬度域→仿真网格重采样
  bathymetry.ts       程序化理想海床地形
  quakeSource.ts      断层几何 → 初始海面位移场(Okada 远场近似)
src/debug-gpu.ts      DEV 专用 GPU↔CPU 64² 逐步一致性对比(经 /debug.html 运行,不进 CI)
src/config.ts         全局仿真参数 + 数值辅助(computeStableDt/polarStride/reducedGridMinFactor)
```

数据流:`quakeSource`/点击 → `TsunamiSolver.inject`(或 `CpuSolver.inject`)→ 状态纹理 RGBA=(η, hu, hv, 1) ping-pong 步进 → `SceneApp` 渲染网格采样 η 位移 + 切向梯度法线。

---

## 3. ⚠️ 核心不变量:CPU ↔ GPU 求解器镜像(最重要)

**`cpuSolver.ts`(TS/float64)与 `shaders.ts` 的 GLSL 求解器必须逐式同构。** 二者是同一数值格式的两份实现:

| CPU(`cpuSolver.ts`) | GPU(`shaders.ts`) | 说明 |
|---|---|---|
| `step()` → `stepLf()`(`scheme='lf'`) | `STEP_FRAG`(`uScheme=0`) | 一阶 Lax–Friedrichs 教学对照(保留 `uDamping` 宽海绵) |
| `step()` → `stepV2()`(`scheme='v2'`) | `STEP_FRAG_V2`(`uScheme=1`,两 stage RK2) | 二阶 MUSCL-Rusanov-SSP-RK2 默认路径 |
| `computeL(e,u,v)` | `STEP_FRAG_V2` main 的 `L` 组装 | 空间算子(通量散度 + 井平衡源) |
| `fluxX(i,j,st)` / `fluxY(i,j)` | `musclFluxX(q,k)` / `musclFluxY(q)` | 界面通量(MUSCL 重构 + minmod + Rusanov) |
| `inject(u,v,amp,r)` | `INJECT_FRAG` | 高斯型海底抬升震源 |
| 薄海绵 `thinSpongeFactor` + 辐射 pass(特征投影) | `thinSponge()` + V2 特征投影辐射 BC | 平面域边界处理 |
| `kLat` / `dxEff=k·dx·cosφ`(缩减纬网) | V2/LF 的 `int k` / `dxEff` | 极地 stride-k 合并 |
| `computeL` 频散源(`vxAt/vyAt/divAt`) | V2 的 `dispVx/dispVy/dispDiv`(`uDispersion>0.5`) | 可选频率频散(η 位势形式源项,coef=min(h²/3, 0.6·min(dx,dy)²)) |

- **uniform 语义一致**:`uDx/uDy/uDt/uG/uHMin/uManning/uGlobeMode/uDispersion/uStage/u0/uDiag`(LF 另有 `uDamping`)。`uScheme` 是 JS 侧控制量(`>0.5` 绑定 V2 程序,否则 LF),运行时可切换。
- **为什么这条不变量是硬性的**:CI 环境**无 WebGL**,只能运行 `cpuSolver` 镜像(`benchmarks.test.ts`)。若只改 GLSL 而不改 CPU 镜像,**CI 仍会全绿,但实际 GPU 渲染行为可能与"通过的测试"背离** —— 测试将不再代表真实求解器。

### 改求解器的标准流程(checklist)

1. **先在 `cpuSolver.ts` 实现**改动,并在 `benchmarks.test.ts` 增/改对应数值基准用例(float64 精确、易断点调试、可量化验收)。
2. `npm test` 确认 CPU 侧达标(验收阈值见 [ROADMAP.md](ROADMAP.md))。
3. **镜像到 `shaders.ts`** 对应着色器(LF→`STEP_FRAG`、V2→`STEP_FRAG_V2`),注意 GLSL ES 1.00 限制(见 §8)。
4. 用 **`/debug.html`**(即 `src/debug-gpu.ts`)做 64² **GPU↔CPU 逐步对比**:先比 `L` 算子诊断,再比 20 步 η/hu 剖面,确认二者一致(典型 L 算子差 ≤3e-6)。
5. 浏览器目视验收(触发 tohoku 预设看波形锐度 / 边界吸收 / 极地无爆炸斑块)。
6. `npm run check && npm test && npm run build` 全绿后再提交。

---

## 4. 测试与验证体系

| 层 | 文件 / 工具 | 覆盖 | 是否进 CI |
|----|------------|------|:--:|
| 数值基准 | `src/simulation/benchmarks.test.ts` | 行波衰减、二阶收敛阶、静水平衡(lake-at-rest)、质量守恒、Green 定律浅水放大、辐射边界反射、缩减纬网极地波速、球面长时稳定 | ✅ |
| 数据管线 | `src/simulation/RealBathymetry.test.ts` | `.tsunami` 二进制解析 + 经纬度域重采样 | ✅ |
| GPU↔CPU 一致性 | `/debug.html`(`src/debug-gpu.ts`) | 64² 小网格 L 算子 + 20 步剖面对比 | ❌(DEV 手动) |
| 浏览器验证 | 目视 + 无头脚本 | 着色器零编译错误、无 NaN/发散、极地稳定;窗口 `hidden` 时用 hook `compileShader`/离屏 `render()`+`readPixels` 等效替代 | ❌(DEV 手动) |

- CI 仅跑 **CPU 镜像 vitest**(无 WebGL)。GPU 侧正确性靠 §3 的镜像不变量 + `/debug.html` + 浏览器验证保证。
- 数值基准的量化验收阈值(每阶段)记录在 [ROADMAP.md](ROADMAP.md)。

---

## 5. CI 门禁

`.github/workflows/ci.yml`:push 到 `main` 或 PR 时触发,Node 22,顺序:

```
npm ci → npm run check(tsc) → npm test(vitest) → npm run build(tsc && vite build)
```

任一步失败即红。**本地提交前请跑同样三步**(§1)。`npm ci` 严格要求 `package-lock.json` 与 `package.json` 依赖树自洽(见 §8 的 lock 陷阱)。

---

## 6. 提交规范

采用 **Conventional Commits + 中文描述**。历史示例:

| 前缀 | 用途 | 示例 |
|------|------|------|
| `feat(sim):` | 求解器/仿真功能(阶段性升级用「阶段N」) | `feat(sim): 阶段3 辐射边界(特征投影)+ 缩减纬网极地处理` |
| `fix:` / 纯中文 | 缺陷修复 | `修复 .tsunami 头部偏移读写不一致(54→78)` |
| `docs:` | 文档 | `docs: 重截 3D 视角与线框截图` |
| `ci:` | CI 配置 | `ci: 升级 actions 至 v7` |
| `chore(deps):` | 依赖 | `chore(deps): 升级 vite8+vitest5 清零漏洞` |

**求解器阶段性提交的 body 建议包含**:改了什么(要点列表)、CPU/GPU 镜像说明、验收结果(基准数值)、版本号变化(如 `版本 0.3.0 → 0.4.0`)。

---

## 7. 发布流程(打 tag)

每完成一个里程碑阶段:

1. **bump `package.json` 的 `version`**(semver:阶段级 minor,修复级 patch)。
2. `git add -A && git commit`(§6 规范)。
3. **附注 tag**:`git tag -a vX.Y.Z -m "阶段N:<摘要>"`(历史 tag 均为 annotated)。
4. **push 分支 + tag**:`git push origin main && git push origin vX.Y.Z`。

现有 tag:`v0.1.0`(开源首发)→ `v0.2.0`(阶段1)→ `v0.3.0`(阶段2)→ `v0.4.0`(阶段3)。

**本机 git 未配置 `user.name`/`user.email`**:提交/打 tag 时用**内联身份**,勿改全局配置:

```bash
git -c user.name="Dongwu259" -c user.email="Dongwu259@users.noreply.github.com" commit -F <msgfile>
git -c user.name="Dongwu259" -c user.email="Dongwu259@users.noreply.github.com" tag -a vX.Y.Z -m "..."
```

**push 凭据**:见 §8 的两类 403 陷阱(需用内联 `credential.helper` 重置走 gh 身份)。

---

## 8. 已知陷阱(改代码前务必阅读)

**求解器 / 数值**
- **只改 CPU 或只改 GPU** → CI 假绿而实际行为背离(§3)。任何数值改动两边都要改并跑 `/debug.html` 对比。
- **缩减纬网 stride-k 一致性**:`dxEff = k·dx·cosφ` 必须与经向邻居的 stride `k` 匹配。若 `dxEff` 用 `k·dx` 而邻居仍取 `i±1`,波速会变成 `c/k`(错误)。`k=polarStride(lat)`:`|φ|>80°→4`、`>74°→2`、否则 `1`。y 方向保持 stride-1(dy 恒定)。
- **极地波速测量的剪切耦合伪影**:球面上不同纬度列速 `c/(dx·cosφ)` 不同,y 耦合会把极侧更快东移的 η 喂入目标行,使质心东偏 → **长时间测速偏快(误差 ∝ T)**。量测时用 **y-均匀注入(初始无 y 梯度)+ 早期窗口**;赤道无剪切,可验证格式内禀波速(误差应 ≈0)。
- **球面 CFL 是二维的**:稳定条件 `νx+νy ≤ 1`(非单向 `ν≤1`)。去掉旧 `cflFloor` hack 后,globe 的 `dt` 必须用**缩减后最小经向格距** `dxM·reducedGridMinFactor(GLOBE_LAT_SPAN, sizeY)` 计算,否则高纬模式缓慢累积、数十分钟后**延迟爆炸**(峰值波高显示天文数字是最直接信号)。
- **GLSL ES 1.00 限制**:无 `int` 版 `min`/`max`(`genIType` 是 ES 3.00),整数取小/大用三元 `a<b?a:b`;支持 int 算术、三元、`float(k)` 转换。
- **显式 Boussinesq 频散的稳定性**(阶段4):频散系数必须带**网格上限** `coef = min(h²/3, 0.6·min(dxEff,dy)²)`——显式处理的稳定条件 `b = coef·k² ≤ 1` 要对一切网格模式成立。试过并否决的方案:半隐式 Δq 修正(慢驻波寄生)、滞后 L⁰ 修正(共振爆炸)、逐点比值限制器 `|S|≤0.9|L|`(|L| 过零处削顶 → 谐波级联耗散)。改动频散项须重跑阶段4 三基准。

**工程 / 发布**
- **push 403「Write access to repository not granted」**:git 走了系统 `osxkeychain` 里缓存的**另一身份**(它先于临时 helper 被尝试)。用**内联重置** helper 列表再挂 gh,使 gh 成为唯一凭据源(勿持久化改配置):
  ```bash
  GIT_TERMINAL_PROMPT=0 git -c credential.helper= -c credential.helper='!gh auth git-credential' push origin <ref>
  ```
  排查:先 `gh api repos/Dongwu259/tsunami-lab --jq .permissions` 确认 token 确有 `push:true`(排除权限问题),再 `git config --get-all credential.helper` 查 osxkeychain 抢占。403(非 401)通常代表认证成功但账号错误。
- **push 403「without workflow scope」**(改 `.github/workflows/` 时):gh token 缺 `workflow` scope。`gh auth refresh -h github.com -s workflow`(**后台运行**读一次性设备码转达用户去 github.com/login/device 输入;前台会死锁),补 scope 后重推。
- **`npm ci` 报 EUSAGE「not in sync」**:新增依赖时增量 `npm install` 可能为包私带嵌套副本、optional 传递依赖未写入 lock 顶层。修复:选与核心工具主版本兼容的依赖版本 → 删 `node_modules` + `package-lock.json` → 全新 `npm install` 重建自洽 lock → 验证后连同 `package.json`/`package-lock.json` 一起提交。
- **`.tsunami` 头部偏移**:头部共 **78 字节**(magic4 + 版本2 + 宽4 + 高4 + 四至 4×8 + 数据源名 32)。读写偏移必须一致,否则地形错位 + 垃圾尖峰。
- **`package-lock.json` 根 `version` 字段**长期停留在 `0.1.0`(各阶段只 bump `package.json`):`npm ci` 只校验依赖树、不校验根 version,故 CI 不受影响。**沿用此惯例即可**,无需同步 lock 根版本。

**渲染 / 交互**(见 `SceneApp.ts` / `ControlPanel.ts`)
- 水面渲染启用透明混合;控制台在窄屏默认折叠;新增功能应提供用户可调参数。
- 模拟速度需与刷新率解耦(固定步频推进时间),保证不同设备行为一致。
- 球面着色器中 `east` 向量在极点退化,需加微小偏移防 NaN。

---

## 9. 许可

代码 MIT(见 [LICENSE](LICENSE));随包数据与外部服务署名要求见 README「许可证与数据源署名」。贡献即同意以 MIT 许可分发。
