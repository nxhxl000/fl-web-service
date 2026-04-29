import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { listJoinedProjectIds, listProjects } from '../api/projects'
import type { Project } from '../api/types'
import { useAuth } from '../auth/useAuth'
import { InferencePlayground } from '../components/InferencePlayground'

export function LandingPage() {
  const { user } = useAuth()
  const [projects, setProjects] = useState<Project[]>([])
  const [joinedIds, setJoinedIds] = useState<Set<number>>(new Set())
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    listProjects()
      .then(setProjects)
      .catch((err: Error) => setError(err.message))
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    if (!user) {
      setJoinedIds(new Set())
      return
    }
    listJoinedProjectIds()
      .then((ids) => setJoinedIds(new Set(ids)))
      .catch(() => undefined)
  }, [user])

  const projectsWithModels = projects.filter((p) => p.inference_target !== null)

  return (
    <main className="px-8 py-12">
      <section className="mx-auto max-w-3xl text-center">
        <h1 className="text-4xl font-semibold tracking-tight text-neutral-900">
          Try our federated-trained models
        </h1>
        <p className="mt-4 text-neutral-600">
          Each model below was trained collectively across many machines without sharing raw data.
          Drop an image to see what it predicts — or sign up and contribute compute to the next run.
        </p>
        <div className="mt-8 flex justify-center gap-3">
          {user ? (
            <Link
              to="/projects"
              className="rounded bg-neutral-900 px-5 py-2.5 text-sm text-white hover:bg-neutral-700"
            >
              Browse projects
            </Link>
          ) : (
            <>
              <Link
                to="/login"
                className="rounded border border-neutral-300 px-5 py-2.5 text-sm text-neutral-800 hover:bg-neutral-100"
              >
                Login
              </Link>
              <Link
                to="/register"
                className="rounded bg-neutral-900 px-5 py-2.5 text-sm text-white hover:bg-neutral-700"
              >
                Become a contributor
              </Link>
            </>
          )}
        </div>
      </section>

      <section className="mx-auto mt-16 max-w-6xl">
        <h2 className="mb-6 text-xl font-semibold text-neutral-900">Featured models</h2>
        {loading ? (
          <p className="text-sm text-neutral-500">Loading projects…</p>
        ) : error ? (
          <p className="text-sm text-red-600">{error}</p>
        ) : projectsWithModels.length === 0 ? (
          <div className="rounded border border-dashed border-neutral-300 bg-white p-8 text-center text-sm text-neutral-500">
            No trained models published yet.
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            {projectsWithModels.map((p) => (
              <ProjectShowcaseCard
                key={p.id}
                project={p}
                joined={joinedIds.has(p.id)}
              />
            ))}
          </div>
        )}
      </section>
    </main>
  )
}

function ProjectShowcaseCard({ project, joined }: { project: Project; joined: boolean }) {
  const target = project.inference_target
  const dataset = project.test_dataset_info
  const [showClasses, setShowClasses] = useState(false)
  if (!target) return null
  return (
    <article className="relative flex h-full flex-col rounded border border-neutral-200 bg-white p-6">
      <header className="flex items-start justify-between gap-4">
        <div>
          <Link
            to={`/projects/${project.id}`}
            className="text-lg font-semibold text-neutral-900 hover:underline"
          >
            {project.name}
          </Link>
          <p className="mt-1 text-sm text-neutral-600">{project.summary}</p>
        </div>
        {target.accuracy !== null && (
          <span className="shrink-0 rounded-full bg-green-100 px-3 py-1 text-xs font-medium text-green-800">
            {(target.accuracy * 100).toFixed(1)}% acc
          </span>
        )}
      </header>
      <div className="mt-3 flex flex-wrap items-center gap-2 text-xs">
        <span className="rounded-full border border-neutral-200 bg-neutral-50 px-3 py-1">
          <span className="text-neutral-500">Model:</span>{' '}
          <span className="font-medium text-neutral-900">{target.model_name}</span>
        </span>
        {dataset ? (
          <>
            <span className="rounded-full border border-neutral-200 bg-neutral-50 px-3 py-1">
              <span className="text-neutral-500">Classes:</span>{' '}
              <span className="font-medium text-neutral-900">{dataset.num_classes}</span>
            </span>
            {dataset.image_size && (
              <span className="rounded-full border border-neutral-200 bg-neutral-50 px-3 py-1">
                <span className="text-neutral-500">Image:</span>{' '}
                <span className="font-medium text-neutral-900">
                  {dataset.image_size[0]}×{dataset.image_size[1]}
                  {dataset.image_mode ? ` ${dataset.image_mode}` : ''}
                </span>
              </span>
            )}
            <button
              type="button"
              onClick={() => setShowClasses((v) => !v)}
              className="rounded-full border border-neutral-300 px-3 py-1 text-neutral-700 hover:bg-neutral-100"
            >
              {showClasses ? 'Hide classes' : 'Show classes'}
            </button>
          </>
        ) : (
          <span className="rounded-full border border-neutral-200 bg-neutral-50 px-3 py-1">
            <span className="text-neutral-500">Dataset:</span>{' '}
            <span className="font-medium text-neutral-900">{target.dataset}</span>
          </span>
        )}
      </div>
      {showClasses && dataset && dataset.class_names.length > 0 && (
        <div className="mt-3 max-h-40 overflow-y-auto rounded border border-neutral-200 bg-neutral-50 p-3 font-mono text-xs leading-relaxed text-neutral-800">
          {dataset.class_names.join(', ')}
        </div>
      )}
      <div className="mt-5">
        <InferencePlayground projectId={project.id} dataset={target.dataset} />
      </div>
      <div className="mt-auto flex items-end justify-between gap-3 pt-5">
        {joined ? (
          <Link
            to={`/projects/${project.id}#join`}
            className="inline-block rounded border border-blue-900 px-5 py-2.5 text-sm font-medium text-blue-900 hover:bg-blue-50"
          >
            Open project
          </Link>
        ) : (
          <Link
            to={`/projects/${project.id}#join`}
            className="inline-block rounded bg-blue-900 px-5 py-2.5 text-sm font-medium text-white hover:bg-blue-800"
          >
            Join To Project
          </Link>
        )}
        {joined && (
          <span className="rounded-full bg-green-100 px-3 py-1 text-xs font-medium text-green-800">
            Already joined
          </span>
        )}
      </div>
    </article>
  )
}
