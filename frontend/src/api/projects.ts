import { apiFetch } from './client'
import type {
  ClientTokenWithOwner,
  Project,
  ProjectAdmin,
  ProjectCreate,
  ProjectUpdate,
} from './types'

export function listProjects(): Promise<Project[]> {
  return apiFetch<Project[]>('/projects', { auth: true })
}

export function listJoinedProjectIds(): Promise<number[]> {
  return apiFetch<number[]>('/projects/joined', { auth: true })
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

export function getProjectAdmin(id: number): Promise<ProjectAdmin> {
  return apiFetch<ProjectAdmin>(`/projects/${id}/admin`, { auth: true })
}

export function analyzeDataset(id: number, path: string): Promise<ProjectAdmin> {
  return apiFetch<ProjectAdmin>(`/projects/${id}/dataset/analyze`, {
    method: 'POST',
    body: { path },
    auth: true,
  })
}

export type DirListing = {
  path: string
  parent: string | null
  subdirs: string[]
}

export function browseDirectory(path: string): Promise<DirListing> {
  const q = path ? `?path=${encodeURIComponent(path)}` : ''
  return apiFetch<DirListing>(`/projects/dataset/browse${q}`, { auth: true })
}
