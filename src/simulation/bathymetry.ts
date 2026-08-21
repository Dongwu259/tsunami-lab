import { SIM_SIZE } from '../config';

/**
 * 程序化生成理想化海床地形(教学用途,非真实地理数据)。
 * 布局:左侧深海 + 俯冲海沟,中部海山,右侧大陆架与蜿蜒海岸线。
 */

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

/**
 * 生成海床高程数组,单位:米(负值 = 海平面以下)。
 * 数组按行主序存储,index = y * size + x,与纹理 uv 对应。
 */
export function generateBathymetry(size: number = SIM_SIZE): Float32Array {
  const out = new Float32Array(size * size);

  for (let j = 0; j < size; j++) {
    const v = j / (size - 1);
    for (let i = 0; i < size; i++) {
      const u = i / (size - 1);

      // 蜿蜒的海岸线位置(u 坐标,左侧为海洋)
      const coast =
        0.68 +
        0.045 * Math.sin(v * Math.PI * 2.6) +
        0.02 * Math.sin(v * Math.PI * 7.0 + 1.3);

      // s: 1 = 深海,0 = 陆地
      const s = 1 - smoothstep(coast - 0.16, coast + 0.03, u);

      // 基础剖面:深海 -3600 m,向岸逐渐抬升为 +90 m 左右的陆地
      let bed =
        -3600 * Math.pow(s, 2.2) +
        180 * Math.pow(1 - s, 2.2) -
        90 * (1 - s);

      // 俯冲带海沟(典型震源区)
      bed -= 500 * Math.exp(-Math.pow((u - 0.3) / 0.035, 2));

      // 海山
      bed +=
        2200 *
        Math.exp(
          -(
            Math.pow((u - 0.16) / 0.045, 2) +
            Math.pow((v - 0.74) / 0.045, 2)
          )
        );

      // 轻微起伏,避免地形过于平滑
      bed += 25 * Math.sin(u * 53.0) * Math.sin(v * 61.0);

      out[j * size + i] = bed;
    }
  }
  return out;
}
