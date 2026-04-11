import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { ApiError } from '../api/client'
import { createProject, listProjects } from '../api/projects'
import type { Project } from '../api/types'
import { useAuth } from '../auth/useAuth'
import { ProjectFormModal } from '../components/ProjectFormModal'

export function ProjectsPage() {
  const { user } = useAuth()
  const [projects, setProjects] = useState<Project[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [showCreate, setShowCreate] = useState(false)

  const load = () => {
    setLoading(true)
    listProjects()
      .then(setProjects)
      .catch((err) => {
        setError(err instanceof ApiError ? err.detail : 'Failed to load projects')
      })
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    load()
  }, [])

  return (
    <main className="px-8 py-10">
      <div className="flex items-start justify-between gap-6">
        <div className="max-w-2xl">
          <h1 className="text-2xl font-semibold tracking-tight text-neutral-900">Projects</h1>
          <p className="mt-2 text-sm text-neutral-600">
            Choose a project to contribute your compute to. Each project trains a shared model
            using federated learning; you run a Docker container on your machine and it joins the
            training.
          </p>
        </div>
        {user?.is_admin && (
          <button
            type="button"
            onClick={() => setShowCreate(true)}
            className="shrink-0 rounded bg-neutral-900 px-4 py-2 text-sm text-white hover:bg-neutral-700"
          >
            Create project
          </button>
        )}
      </div>

      {loading && <p className="mt-8 text-sm text-neutral-500">Loading…</p>}
      {error && <p className="mt-8 text-sm text-red-600">{error}</p>}

      {!loading && !error && projects.length === 0 && (
        <p className="mt-8 text-sm text-neutral-500">
          No projects yet.
          {user?.is_admin ? ' Click “Create project” to add one.' : ''}
        </p>
      )}

      {!loading && projects.length > 0 && (
        <ul className="mt-8 grid gap-4 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
          {projects.map((p) => (
            <li
              key={p.id}
              className="flex flex-col rounded border border-neutral-200 bg-white p-5 shadow-sm"
            >
              <h2 className="text-lg font-semibold text-neutral-900">{p.name}</h2>
              <p className="mt-3 flex-1 text-sm text-neutral-700">{p.summary}</p>
              <div className="mt-4 border-t border-neutral-100 pt-3">
                <div className="text-xs font-medium text-neutral-500">Requirements</div>
                <div className="mt-1 text-sm text-neutral-700">{p.requirements}</div>
              </div>
              <Link
                to={`/projects/${p.id}`}
                className="mt-4 inline-block rounded bg-neutral-900 px-4 py-2 text-center text-sm text-white hover:bg-neutral-700"
              >
                {user?.is_admin ? 'Open' : 'Join'}
              </Link>
            </li>
          ))}
        </ul>
      )}

      {showCreate && (
        <ProjectFormModal
          title="Create project"
          submitLabel="Create"
          onClose={() => setShowCreate(false)}
          onSubmit={async (payload) => {
            await createProject(payload)
            load()
          }}
        />
      )}
    </main>
  )
}
