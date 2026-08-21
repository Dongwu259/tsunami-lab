"""
geoclaw_runner.py —— GeoClaw 科研后端运行器。

职责:
  1. 检测 clawpack 环境是否可用;
  2. 由 .tsunami 地形生成 GeoClaw topofile(topotype 3);
  3. 生成 setrun.py(断层参数 + 全域强制细化,保证输出为单一均匀网格);
  4. 运行模拟并解析 fort.q#### 帧,降采样为紧凑 float32 帧。

说明:本模块在 clawpack 未安装时仍可导入,仅在 run() 时报错。
"""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- 环境检测


def check_environment() -> dict:
    """返回 GeoClaw 环境状态。"""
    info = {"geoclaw": False, "gfortran": False, "clawpack_version": None, "note": ""}
    info["gfortran"] = shutil.which("gfortran") is not None
    try:
        import clawpack  # noqa: F401
        from clawpack import geoclaw  # noqa: F401

        info["geoclaw"] = True
        info["clawpack_version"] = getattr(clawpack, "__version__", "unknown")
    except Exception as exc:  # noqa: BLE001
        info["note"] = f"clawpack 未安装或导入失败: {exc}"
    if info["geoclaw"] and not info["gfortran"]:
        info["note"] = "clawpack 已安装但缺少 gfortran,运行前请安装(brew install gfortran)"
    return info


# ---------------------------------------------------------------- 数据转换


def tsunami_to_topofile(tsunami_path: Path, out_path: Path) -> dict:
    """.tsunami 二进制 → GeoClaw topofile(topotype 3,与 ESRI ASCII 头相同)。"""
    buf = tsunami_path.read_bytes()
    version, nx, ny = struct.unpack_from("<HII", buf, 4)
    west, east, south, north = struct.unpack_from("<dddd", buf, 14)
    grid = np.frombuffer(buf[54 : 54 + nx * ny * 4], dtype="<f4").reshape(ny, nx)

    cell_x = (east - west) / (nx - 1)
    cell_y = (north - south) / (ny - 1)
    with open(out_path, "w") as f:
        f.write(f"{nx}\t\t\t ncols\n")
        f.write(f"{ny}\t\t\t nrows\n")
        f.write(f"{west:.6f}\t xllcorner\n")
        f.write(f"{south:.6f}\t yllcorner\n")
        f.write(f"{cell_x:.8f}\t cellsize\n")
        f.write("-9999\t\t nodata_value\n")
        # GeoClaw 按自北向南写行
        for j in range(ny - 1, -1, -1):
            f.write(" ".join(f"{v:.2f}" for v in grid[j]) + "\n")
    return {
        "west": west, "east": east, "south": south, "north": north,
        "nx": nx, "ny": ny,
    }


SETRUN_TEMPLATE = '''"""由 tsunami-lab 服务自动生成的 GeoClaw 运行脚本。"""
import numpy as np
from clawpack.geoclaw.data import LatLongData, SurgeData  # noqa: F401
from clawpack.geoclaw import util
from clawpack.clawutil.data import ClawData

def setrun(claw_pkg="geoclaw"):
    from clawpack.geoclaw.data import ClawRunData
    rundata = ClawRunData(claw_pkg, ndim=2)

    clawdata = rundata.clawdata
    clawdata.lower[0] = {west}
    clawdata.upper[0] = {east}
    clawdata.lower[1] = {south}
    clawdata.upper[1] = {north}
    clawdata.num_cells[0] = {nx}
    clawdata.num_cells[1] = {ny}

    clawdata.output_style = 1
    clawdata.num_output_times = {nframes}
    clawdata.tfinal = {tfinal}
    clawdata.output_format = "ascii"

    clawdata.cfl_max = 0.9
    clawdata.cfl_desired = 0.75
    clawdata.steps_max = 50000

    amrdata = rundata.amrdata
    amrdata.amr_levels = [1, 2]
    amrdata.refinement_ratios = [2]
    # 全域强制细化到 level 2 → 输出为单一均匀网格,便于回放
    amrdata.aux_type = ["capacity", "center", "center", "left"]  # noqa
    amrdata.variable_dt_refinement_ratios = False

    regions = amrdata.regions
    regions.append([2, 2, 0.0, 1.0e9, {west}, {east}, {south}, {north}])

    geo_data = rundata.geo_data
    geo_data.gravity = 9.81
    geo_data.coordinate_system = 2
    geo_data.manning_coefficient = [0.025]
    geo_data.manning_break = []
    geo_data.sea_level = 0.0
    geo_data.dry_tolerance = 0.001

    topo = geo_data.topo_files
    topo.append([2, 1, 3, 0.0, 1.0e9, "{topofile}"])

    # 断层源(Okada)
    dtopo = rundata.dtopo_data
    dtopo.dtopofiles = []
    fault = {fault_repr}
    dtopo.dtopofiles.append(fault)

    return rundata
'''


def write_job(jobdir: Path, params: dict) -> dict:
    """在任务目录生成 topofile 与 setrun.py,返回区域元信息。"""
    jobdir.mkdir(parents=True, exist_ok=True)

    # 地形:优先使用请求内嵌 base64,否则用仓库随包数据
    import base64

    if params.get("bathy_b64"):
        tsunami_path = jobdir / "region.tsunami"
        tsunami_path.write_bytes(base64.b64decode(params["bathy_b64"]))
    else:
        default = Path(__file__).resolve().parent.parent / "data" / "tohoku.tsunami"
        if not default.exists():
            raise RuntimeError("未找到默认地形 data/tohoku.tsunami,请先运行 tools/fetch_bathy.py")
        tsunami_path = default

    region = tsunami_to_topofile(tsunami_path, jobdir / "topo.tt3")

    sim_time = float(params.get("sim_time", 3600))
    nframes = int(params.get("nframes", 24))
    fault = params.get("fault") or {}
    fault_repr = repr({
        "dtopofile": None,
        "coordinate_specification": "hypocenter",
        "rupture_type": "static",
        "longitude": fault.get("longitude", (region["west"] + region["east"]) / 2),
        "latitude": fault.get("latitude", (region["south"] + region["north"]) / 2),
        "depth": fault.get("depth", 20e3),
        "length": fault.get("length", 200e3),
        "width": fault.get("width", 80e3),
        "strike": fault.get("strike", 20.0),
        "dip": fault.get("dip", 10.0),
        "rake": fault.get("rake", 90.0),
        "slip": fault.get("slip", 10.0),
    })

    setrun = SETRUN_TEMPLATE.format(
        west=region["west"], east=region["east"],
        south=region["south"], north=region["north"],
        nx=region["nx"], ny=region["ny"],
        tfinal=sim_time, nframes=nframes,
        topofile=str(jobdir / "topo.tt3"),
        fault_repr=fault_repr,
    )
    (jobdir / "setrun.py").write_text(setrun)
    region.update({"sim_time": sim_time, "nframes": nframes})
    return region


# ---------------------------------------------------------------- 运行


def run(jobdir: Path, on_progress=None) -> None:
    """运行 GeoClaw 模拟(阻塞)。失败时抛出异常。"""
    if not check_environment()["geoclaw"]:
        raise RuntimeError(
            "clawpack/geoclaw 不可用。请先安装:brew install gfortran && "
            "pip install clawpack(详见 server/README)"
        )

    sys.path.insert(0, str(jobdir))
    try:
        import setrun  # type: ignore  # noqa: E402

        rundata = setrun.setrun()
        # 使用 clawpack 的控制器在纯 Python 中运行
        from clawpack.amrclaw.controller import ClawController

        controller = ClawController()
        controller.rundata = rundata
        controller.outdir = str(jobdir / "_output")
        controller.run()
    finally:
        sys.path.remove(str(jobdir))
        for mod in ("setrun",):
            sys.modules.pop(mod, None)


# ---------------------------------------------------------------- 结果解析


def parse_fort_q(path: Path) -> tuple[float, np.ndarray, tuple]:
    """解析 clawpack ascii fort.q####,取最高层最大网格。返回 (t, eta, (x0,dx,ny...))。"""
    lines = path.read_text().splitlines()
    idx = 0
    ngrids = int(lines[idx].split()[0]); idx += 1
    t = float(lines[idx].split()[0]); idx += 1
    # 跳过 meqn/nvar 等头行(格式:每网格 13 行头)
    best = None
    for _ in range(ngrids):
        hdr = []
        for _ in range(13):
            hdr.append(lines[idx].split()); idx += 1
        level = int(hdr[0][0])
        nvar = int(hdr[1][0])
        maux = int(hdr[11][0])
        mx, my = int(hdr[3][0]), int(hdr[4][0])
        lower = (float(hdr[5][0]), float(hdr[6][0]))
        dxy = (float(hdr[7][0]), float(hdr[8][0]))
        data_lines = mx * my
        vals = np.array(
            [float(v) for ln in lines[idx : idx + data_lines] for v in ln.split()],
            dtype=np.float32,
        )
        idx += data_lines
        vals = vals.reshape(my, mx, nvar + maux)
        eta = vals[:, :, 3]  # h + b - b? 见下方说明
        # clawpack geoclaw q 变量: [h, hu, hv, b, ...];海面位移 = h + b
        eta = vals[:, :, 0] + vals[:, :, 3]
        if best is None or (level, mx * my) > (best[0], best[1].size):
            best = (level, eta[::-1, :], lower, dxy)  # 翻转成自南向北
    if best is None:
        raise RuntimeError("fort.q 中无网格数据")
    _, eta, lower, dxy = best
    return t, eta, lower + dxy


def export_frames(jobdir: Path, outdir: Path, max_width: int = 256) -> list:
    """把 _output/fort.q#### 转为紧凑帧(float32 + 元信息 json)。"""
    outdir.mkdir(exist_ok=True)
    outs = sorted((jobdir / "_output").glob("fort.q*"))
    frames = []
    for i, q in enumerate(outs):
        t, eta, geom = parse_fort_q(q)
        ny, nx = eta.shape
        if nx > max_width:
            stride = int(np.ceil(nx / max_width))
            eta = eta[::stride, ::stride]
        fp = outdir / f"frame_{i:04d}.bin"
        fp.write_bytes(eta.astype("<f4").tobytes())
        frames.append({
            "index": i, "time": t, "nx": eta.shape[1], "ny": eta.shape[0],
            "file": fp.name,
        })
    (outdir / "frames.json").write_text(json.dumps(frames))
    return frames
