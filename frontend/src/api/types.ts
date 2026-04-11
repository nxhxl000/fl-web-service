export type User = {
  id: number
  email: string
  is_admin: boolean
  created_at: string
}

export type TokenResponse = {
  access_token: string
  token_type: string
}

export type Project = {
  id: number
  name: string
  summary: string
  description: string
  requirements: string
  created_at: string
}

export type ProjectCreate = {
  name: string
  summary: string
  description: string
  requirements: string
}

export type ProjectUpdate = Partial<ProjectCreate>

export type ClientToken = {
  id: number
  name: string
  created_at: string
  last_seen_at: string | null
}

export type ClientTokenCreated = ClientToken & {
  token: string
}

export type ClientTokenWithOwner = {
  id: number
  name: string
  user_email: string
  created_at: string
  last_seen_at: string | null
}
