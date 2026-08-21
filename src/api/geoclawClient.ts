/**
 * geoclawClient.ts —— GeoClaw 科研后端(server/app.py)的浏览器客户端。
 *
 * 流程:checkHealth → submitJob → 轮询 jobStatus → fetchFrames/fetchFrame。
 */

export const DEFAULT_GEOCLAW_URL = 'http://localhost:8100';

export interface GeoClawEnv {
  geoclaw: boolean;
  gfortran: boolean;
  clawpack_version: string | null;
  note: string;
}

export interface FaultParams {
  longitude?: number;
  latitude?: number;
  depth?: number;
  length?: number;
  width?: number;
  strike?: number;
  dip?: number;
  rake?: number;
  slip?: number;
}

export interface JobRequest {
  sim_time: number;
  nframes: number;
  fault?: FaultParams;
  /** base64 编码的 .tsunami 地形;缺省时后端用随包 data/tohoku.tsunami */
  bathy_b64?: string;
}

export interface JobRegion {
  west: number;
  east: number;
  south: number;
  north: number;
  nx: number;
  ny: number;
  sim_time: number;
  nframes: number;
}

export interface JobStatus {
  id: string;
  status: 'queued' | 'running' | 'done' | 'error';
  region: JobRegion;
  error: string | null;
  nframes: number;
}

export interface FrameMeta {
  index: number;
  time: number;
  nx: number;
  ny: number;
  file: string;
}

async function json(url: string, init?: RequestInit): Promise<unknown> {
  const resp = await fetch(url, init);
  const body = await resp.json().catch(() => null);
  if (!resp.ok) {
    const detail = (body as { detail?: string } | null)?.detail;
    throw new Error(detail || `HTTP ${resp.status}`);
  }
  return body;
}

/** 探测后端与 GeoClaw 环境 */
export async function checkHealth(baseUrl: string): Promise<GeoClawEnv> {
  const body = await json(`${baseUrl}/health`);
  return (body as { geoclaw_env: GeoClawEnv }).geoclaw_env;
}

/** 提交模拟任务,返回任务 id 与区域元信息 */
export async function submitJob(
  baseUrl: string,
  req: JobRequest
): Promise<{ job_id: string; region: JobRegion }> {
  const body = await json(`${baseUrl}/jobs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  return body as { job_id: string; region: JobRegion };
}

/** 查询任务状态 */
export async function jobStatus(baseUrl: string, jobId: string): Promise<JobStatus> {
  return (await json(`${baseUrl}/jobs/${jobId}`)) as JobStatus;
}

/** 获取帧索引(任务完成后) */
export async function fetchFrames(baseUrl: string, jobId: string): Promise<FrameMeta[]> {
  const resp = await fetch(`${baseUrl}/jobs/${jobId}/frames`);
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  return (await resp.json()) as FrameMeta[];
}

/** 下载第 k 帧 η 场(float32,自南向北行主序) */
export async function fetchFrame(
  baseUrl: string,
  jobId: string,
  meta: FrameMeta
): Promise<Float32Array> {
  const resp = await fetch(`${baseUrl}/jobs/${jobId}/frames/${meta.index}`);
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  return new Float32Array(await resp.arrayBuffer());
}

/** 震级 → 断层滑动量的教学级换算(M0=10^(1.5Mw+9.1),slip=M0/(μLW)) */
export function magnitudeToSlip(mw: number, lengthM = 200e3, widthM = 80e3): number {
  const m0 = Math.pow(10, 1.5 * mw + 9.1);
  return m0 / (3e10 * lengthM * widthM);
}
