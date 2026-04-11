import { useEffect, useState, type FormEvent } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { ApiError } from '../api/client'
import {
  createClientToken,
  deleteClientToken,
  listClientTokens,
} from '../api/clientTokens'
import {
  deleteProject,
  getProject,
  listProjectClients,
  updateProject,
} from '../api/projects'
import type { ClientToken, ClientTokenWithOwner, Project } from '../api/types'
import { useAuth } from '../auth/useAuth'
import { ProjectFormModal } from '../components/ProjectFormModal'

type TabId = 'general' | 'participants' | 'model' | 'fl' | 'metrics'

type Tab = {
  id: TabId
  label: string
  enabled: boolean
}

const ADMIN_TABS: Tab[] = [
  { id: 'general', label: 'General', enabled: true },
  { id: 'participants', label: 'Participants', enabled: true },
  { id: 'model', label: 'Model settings', enabled: false },
  { id: 'fl', label: 'FL settings', enabled: false },
  { id: 'metrics', label: 'Metrics', enabled: false },
]

export function ProjectDetailPage() {
  const { projectId } = useParams<{ projectId: string }>()
  const pid = Number(projectId)
  const navigate = useNavigate()
  const { user } = useAuth()
  const isAdmin = user?.is_admin === true

  const [project, setProject] = useState<Project | null>(null)
  const [tokens, setTokens] = useState<ClientToken[]>([])
  const [allClients, setAllClients] = useState<ClientTokenWithOwner[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const [newName, setNewName] = useState('')
  const [creating, setCreating] = useState(false)
  const [justCreated, setJustCreated] = useState<{ id: number; token: string } | null>(null)

  const [showEdit, setShowEdit] = useState(false)
  const [activeTab, setActiveTab] = useState<TabId>('general')

  const loadAll = async () => {
    setLoading(true)
    setError(null)
    try {
      const p = await getProject(pid)
      setProject(p)
      if (isAdmin) {
        try {
          setAllClients(await listProjectClients(pid))
        } catch {
          // ignore; admin section just won't render
        }
      } else {
        setTokens(await listClientTokens(pid))
      }
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : 'Failed to load project')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (!Number.isFinite(pid)) {
      setError('Invalid project id')
      setLoading(false)
      return
    }
    void loadAll()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pid, isAdmin])

  const handleCreate = async (e: FormEvent) => {
    e.preventDefault()
    if (!newName.trim()) return
    setCreating(true)
    setError(null)
    try {
      const created = await createClientToken(pid, newName.trim())
      setJustCreated({ id: created.id, token: created.token })
      setNewName('')
      setTokens(await listClientTokens(pid))
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : 'Failed to create token')
    } finally {
      setCreating(false)
    }
  }

  const handleDelete = async (id: number) => {
    if (!confirm('Delete this token? Clients using it will stop being recognized.')) return
    try {
      await deleteClientToken(pid, id)
      if (justCreated?.id === id) setJustCreated(null)
      setTokens(await listClientTokens(pid))
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : 'Failed to delete token')
    }
  }

  const handleDeleteProject = async () => {
    if (!project) return
    if (
      !confirm(
        `Delete project "${project.name}"? All client tokens in this project will also be removed.`,
      )
    )
      return
    try {
      await deleteProject(pid)
      navigate('/projects')
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : 'Failed to delete project')
    }
  }

  if (loading) {
    return (
      <main className="mx-auto max-w-3xl px-6 py-10">
        <p className="text-sm text-neutral-500">Loading…</p>
      </main>
    )
  }

  if (error && !project) {
    return (
      <main className="mx-auto max-w-3xl px-6 py-10">
        <p className="text-sm text-red-600">{error}</p>
        <Link to="/projects" className="mt-4 inline-block text-sm underline">
          Back to projects
        </Link>
      </main>
    )
  }

  if (!project) return null

  // ---------- admin view ----------
  if (isAdmin) {
    return (
      <main className="flex min-h-[calc(100vh-57px)]">
        <aside className="w-64 shrink-0 border-r border-neutral-200 bg-white px-5 py-8">
          <Link
            to="/projects"
            className="text-xs text-neutral-500 hover:text-neutral-900"
          >
            ← All projects
          </Link>
          <div className="mt-4 break-words text-sm font-semibold text-neutral-900">
            {project.name}
          </div>
          <nav className="mt-6 flex flex-col gap-1">
            {ADMIN_TABS.map((tab) => {
              const isActive = tab.id === activeTab
              const base =
                'rounded px-3 py-2 text-left text-sm transition-colors'
              if (!tab.enabled) {
                return (
                  <span
                    key={tab.id}
                    className={`${base} cursor-not-allowed text-neutral-400`}
                    title="Coming soon"
                  >
                    {tab.label}
                  </span>
                )
              }
              return (
                <button
                  key={tab.id}
                  type="button"
                  onClick={() => setActiveTab(tab.id)}
                  className={
                    isActive
                      ? `${base} bg-neutral-900 font-medium text-white`
                      : `${base} text-neutral-700 hover:bg-neutral-100`
                  }
                >
                  {tab.label}
                </button>
              )
            })}
          </nav>
        </aside>

        <div className="min-w-0 flex-1 px-8 py-10">
          <div className="max-w-5xl">
          {activeTab === 'general' && (
            <section>
              <div className="flex items-start justify-between gap-4">
                <div>
                  <h1 className="text-2xl font-semibold tracking-tight text-neutral-900">
                    {project.name}
                  </h1>
                  <p className="mt-1 text-sm text-neutral-500">{project.summary}</p>
                </div>
                <div className="flex shrink-0 gap-2">
                  <button
                    type="button"
                    onClick={() => setShowEdit(true)}
                    className="rounded border border-neutral-300 px-3 py-1.5 text-xs text-neutral-700 hover:bg-neutral-100"
                  >
                    Edit
                  </button>
                  <button
                    type="button"
                    onClick={handleDeleteProject}
                    className="rounded border border-red-300 px-3 py-1.5 text-xs text-red-700 hover:bg-red-50"
                  >
                    Delete
                  </button>
                </div>
              </div>

              <div className="mt-6 rounded border border-neutral-200 bg-white p-6">
                <h2 className="text-sm font-semibold text-neutral-900">Description</h2>
                <p className="mt-3 whitespace-pre-wrap text-sm leading-relaxed text-neutral-700">
                  {project.description}
                </p>
              </div>

              <div className="mt-4 rounded border border-neutral-200 bg-white p-6">
                <h2 className="text-sm font-semibold text-neutral-900">Node requirements</h2>
                <p className="mt-3 whitespace-pre-wrap text-sm leading-relaxed text-neutral-700">
                  {project.requirements}
                </p>
              </div>
            </section>
          )}

          {activeTab === 'participants' && (
            <section>
              <h1 className="text-2xl font-semibold tracking-tight text-neutral-900">
                Participants
              </h1>
              <p className="mt-1 text-sm text-neutral-600">
                Every client token registered for this project, across all users.
              </p>
              {allClients.length === 0 ? (
                <p className="mt-6 text-sm text-neutral-500">No clients yet.</p>
              ) : (
                <div className="mt-6 overflow-x-auto rounded border border-neutral-200 bg-white">
                  <table className="min-w-full text-left text-sm">
                    <thead className="bg-neutral-50 text-xs uppercase text-neutral-500">
                      <tr>
                        <th className="px-4 py-2 font-medium">User</th>
                        <th className="px-4 py-2 font-medium">Node</th>
                        <th className="px-4 py-2 font-medium">Created</th>
                        <th className="px-4 py-2 font-medium">Last seen</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-neutral-200">
                      {allClients.map((c) => (
                        <tr key={c.id}>
                          <td className="px-4 py-2 text-neutral-900">{c.user_email}</td>
                          <td className="px-4 py-2 text-neutral-900">{c.name}</td>
                          <td className="px-4 py-2 text-neutral-600">
                            {new Date(c.created_at).toLocaleString()}
                          </td>
                          <td className="px-4 py-2 text-neutral-600">
                            {c.last_seen_at
                              ? new Date(c.last_seen_at).toLocaleString()
                              : 'never'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </section>
          )}
          </div>
        </div>

        {showEdit && project && (
          <ProjectFormModal
            title="Edit project"
            submitLabel="Save"
            initial={{
              name: project.name,
              summary: project.summary,
              description: project.description,
              requirements: project.requirements,
            }}
            onClose={() => setShowEdit(false)}
            onSubmit={async (payload) => {
              const updated = await updateProject(pid, payload)
              setProject(updated)
            }}
          />
        )}
      </main>
    )
  }

  // ---------- non-admin view ----------
  return (
    <main className="px-8 py-10">
      <div className="max-w-3xl">
      <Link to="/projects" className="text-sm text-neutral-500 hover:text-neutral-900">
        ← All projects
      </Link>

      <h1 className="mt-4 text-2xl font-semibold tracking-tight text-neutral-900">
        {project.name}
      </h1>

      <section className="mt-6 rounded border border-neutral-200 bg-white p-5">
        <h2 className="text-sm font-semibold text-neutral-900">Description</h2>
        <p className="mt-2 whitespace-pre-wrap text-sm leading-relaxed text-neutral-700">
          {project.description}
        </p>
      </section>

      <section className="mt-4 rounded border border-neutral-200 bg-white p-5">
        <h2 className="text-sm font-semibold text-neutral-900">Node requirements</h2>
        <p className="mt-2 whitespace-pre-wrap text-sm leading-relaxed text-neutral-700">
          {project.requirements}
        </p>
      </section>

      <section className="mt-8">
        <h2 className="text-lg font-semibold text-neutral-900">Join this project</h2>
        <p className="mt-1 text-sm text-neutral-600">
          Create a token for each machine you want to run a client on. Paste it into{' '}
          <code className="rounded bg-neutral-100 px-1 py-0.5 text-xs">FL_TOKEN</code> when you
          start the Docker container.
        </p>

        <form onSubmit={handleCreate} className="mt-4 flex gap-2">
          <input
            type="text"
            placeholder="Node name, e.g. 'laptop' or 'vm-client1'"
            required
            maxLength={100}
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            className="flex-1 rounded border border-neutral-300 px-3 py-2 text-sm focus:border-neutral-500 focus:outline-none"
          />
          <button
            type="submit"
            disabled={creating}
            className="rounded bg-neutral-900 px-4 py-2 text-sm text-white hover:bg-neutral-700 disabled:bg-neutral-400"
          >
            {creating ? 'Creating…' : 'Create token'}
          </button>
        </form>

        {justCreated && (
          <div className="mt-4 rounded border border-amber-300 bg-amber-50 p-4">
            <p className="text-sm font-medium text-amber-900">
              This is the only time the token is shown. Copy it now.
            </p>
            <div className="mt-2 flex items-center gap-2">
              <code className="flex-1 break-all rounded bg-white px-3 py-2 font-mono text-xs text-neutral-900">
                {justCreated.token}
              </code>
              <button
                type="button"
                onClick={() => navigator.clipboard.writeText(justCreated.token)}
                className="rounded border border-neutral-300 px-3 py-2 text-xs text-neutral-700 hover:bg-neutral-100"
              >
                Copy
              </button>
            </div>
          </div>
        )}

        {error && <p className="mt-4 text-sm text-red-600">{error}</p>}

        <h3 className="mt-8 text-sm font-semibold text-neutral-900">
          Your tokens in this project
        </h3>
        {tokens.length === 0 ? (
          <p className="mt-2 text-sm text-neutral-500">No tokens yet.</p>
        ) : (
          <ul className="mt-2 divide-y divide-neutral-200 rounded border border-neutral-200 bg-white">
            {tokens.map((t) => (
              <li key={t.id} className="flex items-center justify-between px-4 py-3">
                <div>
                  <div className="text-sm font-medium text-neutral-900">{t.name}</div>
                  <div className="text-xs text-neutral-500">
                    Created {new Date(t.created_at).toLocaleString()}
                    {t.last_seen_at
                      ? ` · last seen ${new Date(t.last_seen_at).toLocaleString()}`
                      : ' · never seen'}
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => handleDelete(t.id)}
                  className="rounded border border-red-300 px-3 py-1 text-xs text-red-700 hover:bg-red-50"
                >
                  Delete
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>
      </div>
    </main>
  )
}
