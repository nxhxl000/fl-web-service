import { apiFetch } from './client'
import type { Project } from './types'

export type TrainedModel = {
  id: number
  project_id: number
  run_id: number | null
  display_name: string
  model_name: string
  dataset: string
  accuracy: number | null
  f1_score: number | null
  num_rounds: number | null
  created_at: string
}

export type TrainedModelCreate = {
  display_name: string
  model_name: string
  dataset: string
  weights_path: string
  accuracy: number | null
  f1_score: number | null
  num_rounds: number | null
}

export function listTrainedModels(projectId: number): Promise<TrainedModel[]> {
  return apiFetch<TrainedModel[]>(`/projects/${projectId}/models`, { auth: true })
}

export function createTrainedModel(
  projectId: number,
  payload: TrainedModelCreate,
): Promise<TrainedModel> {
  return apiFetch<TrainedModel>(`/projects/${projectId}/models`, {
    method: 'POST',
    body: payload,
    auth: true,
  })
}

export function deleteTrainedModel(projectId: number, modelId: number): Promise<Project> {
  return apiFetch<Project>(`/projects/${projectId}/models/${modelId}`, {
    method: 'DELETE',
    auth: true,
  })
}

export function promoteTrainedModel(projectId: number, modelId: number): Promise<Project> {
  return apiFetch<Project>(`/projects/${projectId}/models/${modelId}/promote`, {
    method: 'POST',
    auth: true,
  })
}
