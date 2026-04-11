import { apiFetch } from './client'
import type { TokenResponse, User } from './types'

export function register(email: string, password: string): Promise<User> {
  return apiFetch<User>('/auth/register', {
    method: 'POST',
    body: { email, password },
  })
}

export function login(email: string, password: string): Promise<TokenResponse> {
  return apiFetch<TokenResponse>('/auth/login', {
    method: 'POST',
    body: { email, password },
  })
}

export function fetchMe(): Promise<User> {
  return apiFetch<User>('/auth/me', { auth: true })
}
