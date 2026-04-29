import { ApiError } from './client'

export type Prediction = {
  class_id: number
  class_name: string
  confidence: number
}

export type PredictResponse = {
  model_name: string
  dataset: string
  predictions: Prediction[]
}

export async function predict(projectId: number, file: File): Promise<PredictResponse> {
  const form = new FormData()
  form.append('image', file)
  const response = await fetch(`/api/projects/${projectId}/inference/predict`, {
    method: 'POST',
    body: form,
  })
  const text = await response.text()
  const data = text ? JSON.parse(text) : null
  if (!response.ok) {
    const detail =
      data && typeof data === 'object' && 'detail' in data
        ? String((data as { detail: unknown }).detail)
        : response.statusText
    throw new ApiError(response.status, detail)
  }
  return data as PredictResponse
}
