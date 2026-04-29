import { useRef, useState } from 'react'
import { ApiError } from '../api/client'
import { predict, type Prediction } from '../api/inference'

type Props = {
  projectId: number
  dataset: string
}

const SUPPORTED_FORMATS = 'PNG, JPEG, WebP, BMP, GIF'

export function InferencePlayground({ projectId, dataset: _dataset }: Props) {
  const fileInput = useRef<HTMLInputElement>(null)
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [predictions, setPredictions] = useState<Prediction[] | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [dragOver, setDragOver] = useState(false)

  const handleFile = async (file: File) => {
    if (!file.type.startsWith('image/')) {
      setError('Please drop an image file.')
      return
    }
    setError(null)
    setPredictions(null)
    setImageUrl(URL.createObjectURL(file))
    setLoading(true)
    try {
      const res = await predict(projectId, file)
      setPredictions(res.predictions)
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : 'Inference failed')
    } finally {
      setLoading(false)
    }
  }

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files?.[0]
    if (file) void handleFile(file)
  }

  return (
    <div>
      <div
        onDragOver={(e) => {
          e.preventDefault()
          setDragOver(true)
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => fileInput.current?.click()}
        className={
          'cursor-pointer rounded border-2 border-dashed p-6 text-center text-sm transition-colors ' +
          (dragOver
            ? 'border-neutral-900 bg-neutral-50'
            : 'border-neutral-300 bg-white hover:bg-neutral-50')
        }
      >
        {imageUrl ? (
          <div className="flex flex-col items-center gap-2">
            <img src={imageUrl} alt="uploaded" className="max-h-40 rounded border border-neutral-200" />
            <span className="text-xs text-neutral-500">Click or drop another image</span>
          </div>
        ) : (
          <>
            <p className="text-neutral-700">Drop an image here, or click to choose a file.</p>
            <p className="mt-1 text-xs text-neutral-500">
              Supported formats: {SUPPORTED_FORMATS}.
            </p>
          </>
        )}
        <input
          ref={fileInput}
          type="file"
          accept="image/*"
          style={{ display: 'none' }}
          onChange={(e) => {
            const f = e.target.files?.[0]
            if (f) void handleFile(f)
          }}
        />
      </div>

      {loading && <p className="mt-3 text-sm text-neutral-500">Predicting…</p>}
      {error && <p className="mt-3 text-sm text-red-600">{error}</p>}
      {predictions && (
        <div className="mt-4">
          <h4 className="text-xs font-semibold uppercase tracking-wide text-neutral-500">
            Top {predictions.length} predictions
          </h4>
          <ul className="mt-2 space-y-1.5">
            {predictions.map((p, i) => (
              <li key={p.class_id} className="text-sm">
                <div className="flex items-baseline justify-between gap-2">
                  <span className={i === 0 ? 'font-semibold text-neutral-900' : 'text-neutral-700'}>
                    {p.class_name}
                  </span>
                  <span className="font-mono text-xs text-neutral-500">
                    {(p.confidence * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="mt-0.5 h-1.5 overflow-hidden rounded-full bg-neutral-100">
                  <div
                    className={i === 0 ? 'h-full bg-neutral-900' : 'h-full bg-neutral-400'}
                    style={{ width: `${Math.max(2, p.confidence * 100)}%` }}
                  />
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
