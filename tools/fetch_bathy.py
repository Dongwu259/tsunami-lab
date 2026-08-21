#!/usr/bin/env python3
"""
fetch_bathy.py —— 真实海床地形数据管线。

将 GEBCO / ETOPO 等真实全球地形数据裁切为 Web 端可加载的 .tsunami 二进制:

    支持的输入:
      --tiles    在线高程瓦片(Terrarium 编码,默认 AWS elevation-tiles-prod,
                 源自 NOAA/SRTM 全球地形水深,无需密钥)
      --netcdf   GEBCO_2024.nc / ETOPO 2022 .nc 等 NetCDF 文件(自动识别变量名)
      --ascii    ESRI ASCII grid(.asc,GeoClaw 原生 topofile 格式)
      --png      全球地形灰度图(8bit bump map,线性标定,教学级精度)

    输出 .tsunami 格式(小端):
      magic 'TSNB' | u16 版本 | u32 宽 | u32 高
      | float64 西 | float64 东 | float64 南 | float64 北
      | char[32] 数据源名 | float32[宽×高] 高程(米,行主序,自南向北)

用法示例:
    python3 tools/fetch_bathy.py --tiles \
        --west 140 --east 150 --south 35 --north 42 \
        --size 320 --name tohoku --out data/tohoku.tsunami

    # 全球模式(3D 地球):东西跨 360° 即全球,墨卡托纬限 ±84°
    python3 tools/fetch_bathy.py --tiles --globe \
        --size 1024 --name globe --out data/globe.tsunami

    python3 tools/fetch_bathy.py --netcdf ~/GEBCO_2024.nc \
        --west 140 --east 150 --south 35 --north 42 \
        --size 320 --name gebco-tohoku --out data/tohoku.tsunami

    python3 tools/fetch_bathy.py --png data/earth-topology.png \
        --west 140 --east 150 --south 35 --north 42 \
        --size 320 --name relief-tohoku --out data/tohoku.tsunami

依赖:pip install netCDF4 pillow numpy(ascii/png 模式只需 numpy + pillow)
"""

import argparse
import io
import math
import struct
import sys
import time
import urllib.request

import numpy as np

MAGIC = b"TSNB"
VERSION = 1

DEFAULT_TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"


# ---------------------------------------------------------------- 输入读取

def read_tiles(west, east, south, north, url_template, arcsec=30.0):
    """拉取 Terrarium 编码高程瓦片并拼接。
    编码:elev = R*256 + G + B/256 - 32768(米)。"""
    from PIL import Image

    # 目标分辨率反推缩放级别(瓦片分辨率为 360°/(256·2^z))
    target_deg = arcsec / 3600.0
    z = int(math.ceil(math.log2(360.0 / 256.0 / target_deg)))
    z = min(max(z, 2), 12)

    def tile_xy(lon, lat):
        x = int((lon + 180.0) / 360.0 * (1 << z)) % (1 << z)
        rad = math.radians(lat)
        y = int((1.0 - math.asinh(math.tan(rad)) / math.pi) / 2.0 * (1 << z))
        y = min(max(y, 0), (1 << z) - 1)
        return x, y

    x0, y0 = tile_xy(west, north)
    # 东边界按开区间处理:避免 east=180° 时经度取模环绕回 0,
    # 导致全球模式只取到 1 列瓦片、其余经度被边缘列钳制复制
    x1, y1 = tile_xy(east - 1e-6, south)
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    print(f"瓦片层级 z={z},需拉取 {(x1 - x0 + 1) * (y1 - y0 + 1)} 张瓦片...")

    n = (1 << z) * 256
    stitched = np.full(((y1 - y0 + 1) * 256, (x1 - x0 + 1) * 256), np.nan)
    total = (x1 - x0 + 1) * (y1 - y0 + 1)
    done = 0
    for ty in range(y0, y1 + 1):
        for tx in range(x0, x1 + 1):
            url = url_template.format(z=z, x=tx, y=ty)
            elev = None
            for attempt in range(4):
                try:
                    req = urllib.request.Request(
                        url, headers={"User-Agent": "tsunami-lab/0.1"}
                    )
                    with urllib.request.urlopen(req, timeout=20) as resp:
                        im = Image.open(io.BytesIO(resp.read())).convert("RGB")
                    rgb = np.asarray(im, dtype=np.float64)
                    elev = rgb[:, :, 0] * 256 + rgb[:, :, 1] + rgb[:, :, 2] / 256 - 32768
                    break
                except Exception as exc:  # noqa: BLE001 - 网络抖动重试
                    if attempt == 3:
                        raise RuntimeError(f"瓦片 {url} 拉取失败: {exc}") from exc
                    time.sleep(1.5 * (attempt + 1))
            stitched[(ty - y0) * 256 : (ty - y0 + 1) * 256,
                     (tx - x0) * 256 : (tx - x0 + 1) * 256] = elev
            done += 1
            if done % 8 == 0 or done == total:
                print(f"  进度 {done}/{total}")

    # 瓦片全局像素 → 经纬度
    lon0 = x0 * 256 / n * 360.0 - 180.0
    lat_top = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y0 * 256) / n))))
    lat_bot = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * ((y1 + 1) * 256) / n))))
    h, w = stitched.shape
    lons = lon0 + (np.arange(w) + 0.5) * 360.0 / n
    # 墨卡托行距不等纬,逐行反算纬度
    rows = np.arange(y0 * 256, (y1 + 1) * 256) + 0.5
    lats = np.degrees(np.arctan(np.sinh(math.pi * (1 - 2 * rows / n))))[::-1]
    stitched = stitched[::-1, :]  # 翻转成自南向北
    return stitched, lons, lats


def _pick_variable(ds):
    """在 NetCDF 数据集中寻找高程变量(GEBCO/ETOPO 常见命名)。"""
    for name in ("elevation", "z", "Band1", "topo", "ROSE", "height", "bathymetry"):
        if name in ds.variables:
            return name
    # 兜底:取二维变量中尺寸最大的
    candidates = [
        (v, ds.variables[v]) for v in ds.variables
        if len(ds.variables[v].dimensions) == 2
    ]
    if not candidates:
        raise RuntimeError("NetCDF 中未找到二维高程变量")
    candidates.sort(key=lambda kv: np.prod(kv[1].shape), reverse=True)
    return candidates[0][0]


def _pick_coords(ds, var):
    """根据高程变量的维度名寻找经纬度坐标变量。"""
    dims = ds.variables[var].dimensions  # 通常为 (lat, lon) 或 (y, x)
    lat = lon = None
    for d in dims:
        low = d.lower()
        if low in ("lat", "latitude", "y"):
            lat = d
        elif low in ("lon", "longitude", "x"):
            lon = d
    if lat is None or lon is None:
        raise RuntimeError(f"无法识别坐标维度: {dims}")
    return lat, lon


def read_netcdf(path, west, east, south, north):
    import netCDF4

    ds = netCDF4.Dataset(path)
    var = _pick_variable(ds)
    lat_name, lon_name = _pick_coords(ds, var)
    lats = np.asarray(ds.variables[lat_name][:], dtype=np.float64)
    lons = np.asarray(ds.variables[lon_name][:], dtype=np.float64)

    # 归一化经度到 [-180, 180)
    lons = np.where(lons > 180.0, lons - 360.0, lons)

    lat_asc = lats[-1] > lats[0]
    lon_asc = lons[-1] > lons[0]

    def index_range(coord, lo, hi, asc):
        c = coord if asc else coord[::-1]
        idx = np.where((c >= lo) & (c <= hi))[0]
        if idx.size < 2:
            raise RuntimeError("所选区域在数据范围之外或过小")
        i0, i1 = idx[0], idx[-1]
        return (i0, i1) if asc else (len(coord) - 1 - i1, len(coord) - 1 - i0)

    j0, j1 = index_range(lats, south, north, lat_asc)
    i0, i1 = index_range(lons, west, east, lon_asc)

    grid = np.asarray(ds.variables[var][j0 : j1 + 1, i0 : i1 + 1], dtype=np.float64)
    if not lat_asc:
        grid = grid[::-1, :]
    if not lon_asc:
        grid = grid[:, ::-1]

    lat_sub = np.sort(lats[j0 : j1 + 1])
    lon_sub = np.sort(lons[i0 : i1 + 1])
    fill = getattr(ds.variables[var], "_FillValue", None)
    ds.close()

    if fill is not None:
        grid = np.where(np.isclose(grid, fill), np.nan, grid)

    return grid, lon_sub, lat_sub


def read_ascii(path):
    """ESRI ASCII grid(GeoClaw topofile 亦可用)。"""
    header = {}
    with open(path) as f:
        for _ in range(6):
            k, v = f.readline().split()
            header[k.lower()] = float(v)
    ncols, nrows = int(header["ncols"]), int(header["nrows"])
    xll, yll = header["xllcorner"], header["yllcorner"]
    cell = header["cellsize"]
    nodata = header.get("nodata_value", -9999)
    grid = np.loadtxt(path, skiprows=6, dtype=np.float64).reshape(nrows, ncols)
    grid = grid[::-1, :]  # ascii 自北向南 → 自南向北
    grid[grid <= nodata + 1] = 0.0
    lons = xll + np.arange(ncols) * cell
    lats = yll + np.arange(nrows) * cell
    return grid, lons, lats


def read_png(path):
    """8bit 全球地形灰度图,线性标定:-10994 m(最深海沟)→ +8850 m(珠峰)。"""
    from PIL import Image

    a = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
    a = a[::-1, :]  # 图像行自北向南 → 自南向北
    elev = -10994.0 + a * (8850.0 + 10994.0) / 255.0
    h, w = a.shape
    lons = -180.0 + (np.arange(w) + 0.5) * 360.0 / w
    lats = -90.0 + (np.arange(h) + 0.5) * 180.0 / h
    return elev, lons, lats


# ---------------------------------------------------------------- 重采样

def _fill_nan(grid, iterations=20):
    """无 scipy 依赖的迭代邻域填补(教学级够用)。"""
    for _ in range(iterations):
        mask = np.isnan(grid)
        if not mask.any():
            return grid
        padded = np.pad(grid, 1, mode="edge")
        neighbors = np.stack(
            [padded[:-2, 1:-1], padded[2:, 1:-1], padded[1:-1, :-2], padded[1:-1, 2:]]
        )
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(neighbors, axis=0)
        grid = np.where(mask, mean, grid)
    return np.nan_to_num(grid, nan=0.0)


def resample(grid, src_lons, src_lats, west, east, south, north, size, globe=False):
    """双线性重采样到目标区域,长边固定 size 网格,保持纵横比。
    支持非均匀源网格(如墨卡托瓦片纬度)。globe=True 时固定 2:1 等距圆柱。"""
    if np.isnan(grid).any():
        grid = _fill_nan(grid)

    if globe:
        nx, ny = size, size // 2
    else:
        aspect = (east - west) / max(north - south, 1e-9)
        if aspect >= 1.0:
            nx = size
            ny = max(32, int(round(size / aspect)) // 2 * 2 or 2)
        else:
            ny = size
            nx = max(32, int(round(size * aspect)) // 2 * 2 or 2)

    xs = np.linspace(west, east, nx)
    ys = np.linspace(south, north, ny)

    def axis_idx(coord, target):
        i = np.clip(np.searchsorted(coord, target) - 1, 0, len(coord) - 2)
        t = (target - coord[i]) / np.maximum(coord[i + 1] - coord[i], 1e-12)
        return i, np.clip(t, 0.0, 1.0)

    i0, tx = axis_idx(src_lons, xs)
    j0, ty = axis_idx(src_lats, ys)

    out = np.empty((ny, nx), dtype=np.float64)
    for j in range(ny):
        a = grid[j0[j], i0] * (1 - tx) + grid[j0[j], i0 + 1] * tx
        b = grid[j0[j] + 1, i0] * (1 - tx) + grid[j0[j] + 1, i0 + 1] * tx
        out[j] = a * (1 - ty[j]) + b * ty[j]
    return out


# ---------------------------------------------------------------- 输出

def write_tsunami(path, grid, west, east, south, north, name):
    ny, nx = grid.shape
    name_bytes = name.encode("utf-8")[:32].ljust(32, b"\0")
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<HII", VERSION, nx, ny))
        f.write(struct.pack("<dddd", west, east, south, north))
        f.write(name_bytes)
        f.write(grid.astype("<f4").tobytes())
    size_kb = grid.size * 4 / 1024
    print(f"✓ 已写出 {path}  ({nx}×{ny}, {size_kb:.0f} KB)")
    print(f"  区域: W{west:.3f} E{east:.3f} S{south:.3f} N{north:.3f}")
    print(f"  高程: {grid.min():.0f} m ~ {grid.max():.0f} m")


def main():
    p = argparse.ArgumentParser(description="GEBCO/ETOPO → .tsunami 裁切管线")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--tiles", action="store_true",
                     help="在线 Terrarium 高程瓦片(默认 AWS 公开桶)")
    src.add_argument("--netcdf", help="GEBCO/ETOPO NetCDF 文件")
    src.add_argument("--ascii", help="ESRI ASCII grid(.asc)")
    src.add_argument("--png", help="全球地形灰度图(8bit)")
    p.add_argument("--globe", action="store_true",
                   help="全球模式:区域固定为 ±180°/±84°,输出 2:1 等距圆柱网格")
    p.add_argument("--tile-url", default=DEFAULT_TILE_URL, help="瓦片 URL 模板")
    p.add_argument("--arcsec", type=float, default=30.0, help="瓦片目标分辨率(弧秒)")
    p.add_argument("--west", type=float, default=None)
    p.add_argument("--east", type=float, default=None)
    p.add_argument("--south", type=float, default=None)
    p.add_argument("--north", type=float, default=None)
    p.add_argument("--size", type=int, default=320, help="长边网格数(默认 320)")
    p.add_argument("--name", default="unknown", help="数据源标识(≤32 字符)")
    p.add_argument("--out", required=True, help="输出 .tsunami 路径")
    args = p.parse_args()

    # 全球模式(墨卡托瓦片纬度极限 ±84°,与 Web 地图一致)
    if args.globe:
        args.west, args.east = -180.0, 180.0
        args.south, args.north = -84.0, 84.0
        if args.tiles:
            # 720″ 落在 z=3 区间(633″–1265″,共 64 张瓦片),
            # 瓦片原生分辨率 633″ ≈ 2048×1024 全球网格的单元尺度
            args.arcsec = max(args.arcsec, 720.0)
    elif None in (args.west, args.east, args.south, args.north):
        p.error("非全球模式必须提供 --west/--east/--south/--north")

    if args.tiles:
        grid, lons, lats = read_tiles(
            args.west, args.east, args.south, args.north, args.tile_url, args.arcsec
        )
    elif args.netcdf:
        grid, lons, lats = read_netcdf(
            args.netcdf, args.west, args.east, args.south, args.north
        )
    elif args.ascii:
        grid, lons, lats = read_ascii(args.ascii)
    else:
        grid, lons, lats = read_png(args.png)

    out = resample(grid, lons, lats, args.west, args.east, args.south, args.north,
                   args.size, globe=args.globe)
    write_tsunami(args.out, out, args.west, args.east, args.south, args.north, args.name)


if __name__ == "__main__":
    main()
