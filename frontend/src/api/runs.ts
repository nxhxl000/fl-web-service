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

export function deleteRun(projectId: number, runId: number): Promise<void> {
  return apiFetch<void>(`/projects/${projectId}/runs/${runId}`, {
    method: 'DELETE',
    auth: true,
  })
}

export type RunEvent = Record<string, unknown> & { type: string }

export function getRunEvents(projectId: number, runId: number): Promise<RunEvent[]> {
  return apiFetch<RunEvent[]>(`/projects/${projectId}/runs/${runId}/events`, { auth: true })
}

/**
 * Returns the run-config that was actually passed to `flwr run --run-config`,
 * including backend-injected keys (partition-name, output-dir). Falls back to
 * the saved draft config if the run never started.
 */
export function getRunEffectiveConfig(
  projectId: number,
  runId: number,
): Promise<Record<string, unknown>> {
  return apiFetch<Record<string, unknown>>(
    `/projects/${projectId}/runs/${runId}/config`,
    { auth: true },
  )
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
