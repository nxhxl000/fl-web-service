import { apiFetch } from './client'
import type { ClientToken, ClientTokenCreated } from './types'

export function listClientTokens(): Promise<ClientToken[]> {
  return apiFetch<ClientToken[]>('/clients/tokens', { auth: true })
}

export function createClientToken(name: string): Promise<ClientTokenCreated> {
  return apiFetch<ClientTokenCreated>('/clients/tokens', {
    method: 'POST',
    body: { name },
    auth: true,
  })
}

export function deleteClientToken(id: number): Promise<void> {
  return apiFetch<void>(`/clients/tokens/${id}`, {
    method: 'DELETE',
    auth: true,
  })
}
