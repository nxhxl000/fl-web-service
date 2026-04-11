import { apiFetch } from './client'
import type { ClientToken, ClientTokenCreated } from './types'

export function listClientTokens(projectId: number): Promise<ClientToken[]> {
  return apiFetch<ClientToken[]>(`/projects/${projectId}/tokens`, { auth: true })
}

export function createClientToken(
  projectId: number,
  name: string,
): Promise<ClientTokenCreated> {
  return apiFetch<ClientTokenCreated>(`/projects/${projectId}/tokens`, {
    method: 'POST',
    body: { name },
    auth: true,
  })
}

export function deleteClientToken(projectId: number, tokenId: number): Promise<void> {
  return apiFetch<void>(`/projects/${projectId}/tokens/${tokenId}`, {
    method: 'DELETE',
    auth: true,
  })
}
