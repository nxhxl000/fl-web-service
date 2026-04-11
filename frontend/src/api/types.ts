export type User = {
  id: number
  email: string
  created_at: string
}

export type TokenResponse = {
  access_token: string
  token_type: string
}

export type ClientToken = {
  id: number
  name: string
  created_at: string
  last_seen_at: string | null
}

export type ClientTokenCreated = ClientToken & {
  token: string
}
