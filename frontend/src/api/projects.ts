import { apiFetch } from './client'
import type {
  ClientTokenWithOwner,
  Project,
  ProjectCreate,
  ProjectUpdate,
} from './types'

export function listProjects(): Promise<Project[]> {
  return apiFetch<Project[]>('/projects', { auth: true })
}

export function getProject(id: number): Promise<Project> {
  return apiFetch<Project>(`/projects/${id}`, { auth: true })
}

export function createProject(payload: ProjectCreate): Promise<Project> {
  return apiFetch<Project>('/projects', {
    method: 'POST',
    body: payload,
    auth: true,
  })
}

export function updateProject(id: number, payload: ProjectUpdate): Promise<Project> {
  return apiFetch<Project>(`/projects/${id}`, {
    method: 'PATCH',
    body: payload,
    auth: true,
  })
}

export function deleteProject(id: number): Promise<void> {
  return apiFetch<void>(`/projects/${id}`, {
    method: 'DELETE',
    auth: true,
  })
}

export function listProjectClients(id: number): Promise<ClientTokenWithOwner[]> {
  return apiFetch<ClientTokenWithOwner[]>(`/projects/${id}/clients`, { auth: true })
}
