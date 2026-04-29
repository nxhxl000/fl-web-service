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

export type TrainedModelInline = {
  id: number
  display_name: string
  model_name: string
  dataset: string
  accuracy: number | null
  f1_score: number | null
}

export type DatasetInfo = {
  name: string
  format: string
  num_samples: number
  num_classes: number
  class_names: string[]
  label_column: string | null
  image_column: string | null
  image_size: [number, number] | null
  image_mode: string | null
}

export type Project = {
  id: number
  name: string
  summary: string
  description: string
  requirements: string
  created_at: string
  inference_target_id: number | null
  inference_target: TrainedModelInline | null
  test_dataset_info: DatasetInfo | null
}

export type ProjectAdmin = Project & {
  test_dataset_path: string | null
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
  docker_command: string
}

export type ClientTokenWithOwner = {
  id: number
  name: string
  user_email: string
  created_at: string
  last_seen_at: string | null
}
