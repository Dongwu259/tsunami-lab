import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { parseTsunamiBinary } from './RealBathymetry';

/** .tsunami 头部字节数:magic4 + ver2 + nx4 + ny4 + 4×f64(32) + name[32] = 78 */
const HEADER = 78;

/**
 * 按 .tsunami 格式构造合成文件缓冲(头部 78 字节 + float32 高程)。
 * 与 tools/fetch_bathy.py 的 write_tsunami 布局一致。
 */
function buildTsunami(
  width: number,
  height: number,
  name: string,
  grid: number[],
  bounds = { west: 140, east: 150, south: 35, north: 42 }
): ArrayBuffer {
  const buf = new ArrayBuffer(HEADER + grid.length * 4);
  const dv = new DataView(buf);
  dv.setUint32(0, 0x424e5354, true); // 'TSNB' 小端
  dv.setUint16(4, 1, true); // version
  dv.setUint32(6, width, true);
  dv.setUint32(10, height, true);
  dv.setFloat64(14, bounds.west, true);
  dv.setFloat64(22, bounds.east, true);
  dv.setFloat64(30, bounds.south, true);
  dv.setFloat64(38, bounds.north, true);
  // name char[32] @ 46(ArrayBuffer 已零初始化,不足部分自动补 \0)
  new Uint8Array(buf, 46, 32).set(new TextEncoder().encode(name).slice(0, 32));
  // 高程 float32 @ 78(逐值写,避开 Float32Array 视图的 4 字节对齐限制)
  for (let i = 0; i < grid.length; i++) dv.setFloat32(HEADER + i * 4, grid[i], true);
  return buf;
}

describe('parseTsunamiBinary — 头部偏移契约(78 字节)', () => {
  it('元数据与经纬度边界正确解析', () => {
    const r = parseTsunamiBinary(
      buildTsunami(4, 3, 'test-src', Array.from({ length: 12 }, (_, i) => i - 5))
    );
    expect(r.width).toBe(4);
    expect(r.height).toBe(3);
    expect(r.source).toBe('test-src');
    expect(r.west).toBeCloseTo(140);
    expect(r.east).toBeCloseTo(150);
    expect(r.south).toBeCloseTo(35);
    expect(r.north).toBeCloseTo(42);
    expect(r.grid.length).toBe(12);
  });

  it('高程从偏移 78 开始:首/尾单元与写入值逐一吻合(防偏移 54 回归)', () => {
    // 数据源名含非空字节:若误从偏移 54 读取,name 尾部 ASCII 会被当成高程,
    // grid[0] 将变成 1e27 级垃圾而非 -1234.5,本断言即失败。
    const grid = [-1234.5, 0, 42.25, -9832, 6296, 7, 8, 9, 10, 11, 12, 13.5];
    const r = parseTsunamiBinary(buildTsunami(4, 3, 'terrarium-globe', grid));
    for (let i = 0; i < grid.length; i++) {
      expect(r.grid[i]).toBeCloseTo(grid[i], 3);
    }
    expect(r.grid[0]).toBeCloseTo(-1234.5, 3);
    expect(r.grid[grid.length - 1]).toBeCloseTo(13.5, 3);
  });

  it('name 超过 32 字节时截断且不误读为高程', () => {
    const r = parseTsunamiBinary(
      buildTsunami(2, 2, 'x'.repeat(40), [-1, -2, -3, -4])
    );
    expect(r.source).toBe('x'.repeat(32));
    expect(Array.from(r.grid)).toEqual([-1, -2, -3, -4]);
  });

  it('magic 非法时抛错', () => {
    const buf = buildTsunami(2, 2, 'x', [0, 0, 0, 0]);
    new DataView(buf).setUint32(0, 0xdeadbeef, true);
    expect(() => parseTsunamiBinary(buf)).toThrow(/不是有效的/);
  });

  it('数据长度不足时抛错', () => {
    const buf = buildTsunami(4, 4, 'x', Array(16).fill(0));
    const truncated = buf.slice(0, HEADER + 15 * 4); // 少最后一个单元
    expect(() => parseTsunamiBinary(truncated)).toThrow(/数据长度不足/);
  });
});

describe('随包 data/*.tsunami — 头部与高程合理性', () => {
  const dataDir = fileURLToPath(new URL('../../data', import.meta.url));
  for (const file of ['globe.tsunami', 'tohoku.tsunami']) {
    const path = `${dataDir}/${file}`;
    if (!existsSync(path)) continue; // 数据未随包时跳过
    it(`${file}:size = 78 + nx·ny·4 且无异常高程`, () => {
      const bytes = readFileSync(path);
      const buf = bytes.buffer.slice(
        bytes.byteOffset,
        bytes.byteOffset + bytes.byteLength
      ) as ArrayBuffer;
      const r = parseTsunamiBinary(buf);
      expect(buf.byteLength).toBe(HEADER + r.width * r.height * 4);
      expect(r.grid.length).toBe(r.width * r.height);
      let mn = Infinity;
      let mx = -Infinity;
      for (const v of r.grid) {
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
      // 偏移读错时西南角会出现 1e27 级垃圾值,此处直接拦下
      expect(mx).toBeLessThan(9000); // 高于珠峰即异常
      expect(mn).toBeGreaterThan(-12000); // 深于马里亚纳海沟即异常
    });
  }
});
