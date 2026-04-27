import { apiFetch } from './client'

export type RunStatus = 'draft' | 'running' | 'completed' | 'failed' | 'cancelled'

export type Run = {
  id: number
  project_id: number
  federation: string
  run_config: Record<string, unknown>
  status: RunStatus
  pid: number | null
  log_path: string | null
  exp_dir: string | null
  started_at: string | null
  finished_at: string | null
  exit_code: number | null
  error_message: string | null
  created_at: string
}

export type RunCreatePayload = {
  federation?: string
  run_config: Record<string, unknown>
}

export function createRun(projectId: number, payload: RunCreatePayload): Promise<Run> {
  return apiFetch<Run>(`/projects/${projectId}/runs`, {
    method: 'POST',
    body: payload,
    auth: true,
  })
}

export function listRuns(projectId: number): Promise<Run[]> {
  return apiFetch<Run[]>(`/projects/${projectId}/runs`, { auth: true })
}

export function getRun(projectId: number, runId: number): Promise<Run> {
  return apiFetch<Run>(`/projects/${projectId}/runs/${runId}`, { auth: true })
}

export function startRun(projectId: number, runId: number): Promise<Run> {
  return apiFetch<Run>(`/projects/${projectId}/runs/${runId}/start`, {
    method: 'POST',
    auth: true,
  })
}

export function cancelRun(projectId: number, runId: number): Promise<Run> {
  return apiFetch<Run>(`/projects/${projectId}/runs/${runId}/cancel`, {
    method: 'POST',
    auth: true,
  })
}

export async function getRunLog(projectId: number, runId: number): Promise<string> {
  const response = await fetch(`/api/projects/${projectId}/runs/${runId}/log`, {
    headers: {
      Authorization: `Bearer ${localStorage.getItem('fl_web_jwt') ?? ''}`,
    },
  })
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`)
  }
  return response.text()
}
