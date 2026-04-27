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
import {
  FL_MODELS,
  MODEL_HPARAM_LABELS,
  MODEL_HPARAM_RANGES,
  type ModelHParamNumericKey,
  type ModelHParams,
} from '../api/flModels'
import {
  COMMON_FL_KEYS,
  FL_PARAM_DEFAULTS,
  FL_PARAM_LABELS,
  FL_PARAM_RANGES,
  FL_STRATEGIES,
  STRATEGY_SPECIFIC_KEYS,
  type FLParams,
  type StrategyId,
} from '../api/flStrategies'
import {
  cancelRun,
  createRun,
  getRun,
  startRun,
  type Run,
} from '../api/runs'
import {
  MODE_NUMERIC_KEYS,
  STRAGGLER_MODES,
  STRAGGLER_MODE_DESCRIPTION,
  STRAGGLER_PARAM_DEFAULTS,
  STRAGGLER_PARAM_LABELS,
  STRAGGLER_PARAM_RANGES,
  type StragglerMode,
  type StragglerNumericKey,
  type StragglerParams,
  type StragglerTarget,
} from '../api/stragglerConfig'
import type { ClientToken, ClientTokenWithOwner, Project } from '../api/types'
import { useAuth } from '../auth/useAuth'
import { ProjectFormModal } from '../components/ProjectFormModal'
import { SliderInput } from '../components/SliderInput'
import { TrainingDashboard } from '../components/training/TrainingDashboard'

const ADMIN_SECTIONS = [
  { id: 'general', label: 'General' },
  { id: 'participants', label: 'Participants' },
  { id: 'model', label: 'Model settings' },
  { id: 'fl', label: 'FL settings' },
  { id: 'training', label: 'Training' },
  { id: 'models', label: 'Models' },
] as const

type SectionId = typeof ADMIN_SECTIONS[number]['id']

function scrollToSection(id: SectionId) {
  document
    .getElementById(`section-${id}`)
    ?.scrollIntoView({ behavior: 'smooth', block: 'start' })
}

const READY_THRESHOLD_MS = 90 * 1000

function isReady(lastSeenAt: string | null): boolean {
  if (!lastSeenAt) return false
  return Date.now() - new Date(lastSeenAt).getTime() < READY_THRESHOLD_MS
}

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
  const [checking, setChecking] = useState(false)

  const handleCheck = async () => {
    setChecking(true)
    try {
      setAllClients(await listProjectClients(pid))
    } catch {
      // ignore — section just won't refresh
    } finally {
      setChecking(false)
    }
  }

  const [selectedModel, setSelectedModel] = useState('')
  const [hparams, setHparams] = useState<ModelHParams | null>(null)

  const handleModelChange = (id: string) => {
    setSelectedModel(id)
    const def = FL_MODELS.find((m) => m.id === id)
    setHparams(def ? { ...def.defaults } : null)
  }

  const updateHparam = <K extends keyof ModelHParams>(key: K, value: ModelHParams[K]) => {
    setHparams((prev) => (prev ? { ...prev, [key]: value } : prev))
  }

  const [strategy, setStrategy] = useState<StrategyId | ''>('')
  const [flParams, setFlParams] = useState<FLParams>({ ...FL_PARAM_DEFAULTS })

  const [stragglerParams, setStragglerParams] = useState<StragglerParams>({
    ...STRAGGLER_PARAM_DEFAULTS,
  })

  const updateStragglerNumericParam = (key: StragglerNumericKey, value: number) => {
    setStragglerParams((prev) => ({ ...prev, [key]: value }))
  }

  const [errors, setErrors] = useState<Record<string, string>>({})
  const [currentRun, setCurrentRun] = useState<Run | null>(null)
  const [runError, setRunError] = useState<string | null>(null)
  const [busy, setBusy] = useState<'save' | 'start' | 'cancel' | null>(null)

  const validateConfig = (): Record<string, string> => {
    const errs: Record<string, string> = {}

    if (!selectedModel) {
      errs.model = 'Pick a model.'
    }
    if (selectedModel === 'effnet_b0' && hparams?.optimizer === 'sgd') {
      errs.optimizer = 'EfficientNet-B0 does not support SGD. Use AdamW.'
    }

    if (!strategy) {
      errs.strategy = 'Pick an aggregation strategy.'
    } else {
      const onlineCount = allClients.filter((c) => isReady(c.last_seen_at)).length
      if (flParams['min-available-nodes'] > onlineCount) {
        errs['min-available-nodes'] =
          `Only ${onlineCount} participant(s) online — cannot exceed that.`
      }
    }

    return errs
  }

  const buildRunConfig = (): Record<string, unknown> => ({
    model: selectedModel,
    model_hparams: hparams,
    strategy,
    fl_params: flParams,
    straggler: stragglerParams,
  })

  const handleSave = async () => {
    const errs = validateConfig()
    setErrors(errs)
    setRunError(null)
    if (Object.keys(errs).length > 0) {
      setCurrentRun(null)
      return
    }
    setBusy('save')
    try {
      const run = await createRun(pid, {
        federation: 'dummy',
        run_config: buildRunConfig(),
      })
      setCurrentRun(run)
    } catch (err) {
      setRunError(err instanceof ApiError ? err.detail : 'Failed to save config')
    } finally {
      setBusy(null)
    }
  }

  const handleStart = async () => {
    if (!currentRun) return
    setBusy('start')
    setRunError(null)
    try {
      const run = await startRun(pid, currentRun.id)
      setCurrentRun(run)
    } catch (err) {
      setRunError(err instanceof ApiError ? err.detail : 'Failed to start run')
    } finally {
      setBusy(null)
    }
  }

  const handleCancel = async () => {
    if (!currentRun) return
    setBusy('cancel')
    setRunError(null)
    try {
      const run = await cancelRun(pid, currentRun.id)
      setCurrentRun(run)
    } catch (err) {
      setRunError(err instanceof ApiError ? err.detail : 'Failed to cancel run')
    } finally {
      setBusy(null)
    }
  }

  useEffect(() => {
    if (!currentRun || currentRun.status !== 'running') return
    const runId = currentRun.id
    const interval = setInterval(async () => {
      try {
        const updated = await getRun(pid, runId)
        setCurrentRun(updated)
        if (updated.status !== 'running') {
          clearInterval(interval)
        }
      } catch {
        // ignore transient errors; next tick will retry
      }
    }, 2000)
    return () => clearInterval(interval)
  }, [currentRun?.id, currentRun?.status, pid])

  const updateFlParam = <K extends keyof FLParams>(key: K, value: FLParams[K]) => {
    setFlParams((prev) => ({ ...prev, [key]: value }))
  }

  const renderFlField = (key: keyof FLParams) => {
    const range = FL_PARAM_RANGES[key]
    const errMsg = errors[key]
    return (
      <label key={key} className="block">
        <span className="mb-1 block text-xs font-medium text-neutral-700">
          {FL_PARAM_LABELS[key]}
        </span>
        <SliderInput
          value={flParams[key]}
          onChange={(v) => updateFlParam(key, v)}
          min={range.min}
          max={range.max}
          step={range.step}
        />
        {errMsg && <p className="mt-1 text-xs text-red-600">{errMsg}</p>}
      </label>
    )
  }

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
      <>
        <aside className="fixed bottom-0 left-0 top-[57px] w-64 overflow-y-auto border-r border-neutral-200 bg-white px-5 py-8">
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
            {ADMIN_SECTIONS.map((s) => (
              <button
                key={s.id}
                type="button"
                onClick={() => scrollToSection(s.id)}
                className="rounded px-3 py-2 text-left text-sm text-neutral-700 transition-colors hover:bg-neutral-100"
              >
                {s.label}
              </button>
            ))}
          </nav>
        </aside>

        <main className="ml-64 px-8 py-10">
          <div className="max-w-5xl space-y-12">
            <section id="section-general" className="scroll-mt-20">
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

            <section id="section-participants" className="scroll-mt-20">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <h2 className="text-2xl font-semibold tracking-tight text-neutral-900">
                    Participants
                  </h2>
                  <p className="mt-1 text-sm text-neutral-600">
                    Every client token registered for this project. Click “Check” to refresh
                    readiness — a client is Ready if its last heartbeat was within 90 seconds.
                  </p>
                </div>
                <button
                  type="button"
                  onClick={handleCheck}
                  disabled={checking}
                  className="shrink-0 rounded border border-neutral-300 px-3 py-1.5 text-sm text-neutral-700 hover:bg-neutral-100 disabled:opacity-50"
                >
                  {checking ? 'Checking…' : 'Check'}
                </button>
              </div>
              {allClients.length === 0 ? (
                <p className="mt-6 text-sm text-neutral-500">No clients yet.</p>
              ) : (
                <div className="mt-6 max-h-96 overflow-auto rounded border border-neutral-200 bg-white">
                  <table className="min-w-full text-left text-sm">
                    <thead className="sticky top-0 bg-neutral-50 text-xs uppercase text-neutral-500 shadow-[inset_0_-1px_0_0_rgb(229,229,229)]">
                      <tr>
                        <th className="px-4 py-2 font-medium">User</th>
                        <th className="px-4 py-2 font-medium">Node</th>
                        <th className="px-4 py-2 font-medium">Status</th>
                        <th className="px-4 py-2 font-medium">Created</th>
                        <th className="px-4 py-2 font-medium">Last seen</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-neutral-200">
                      {allClients.map((c) => (
                        <tr key={c.id}>
                          <td className="px-4 py-2 text-neutral-900">{c.user_email}</td>
                          <td className="px-4 py-2 text-neutral-900">{c.name}</td>
                          <td className="px-4 py-2">
                            {isReady(c.last_seen_at) ? (
                              <span className="inline-block rounded-full bg-green-100 px-2 py-0.5 text-xs font-medium text-green-800">
                                Ready
                              </span>
                            ) : (
                              <span className="inline-block rounded-full bg-neutral-100 px-2 py-0.5 text-xs font-medium text-neutral-600">
                                Offline
                              </span>
                            )}
                          </td>
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

            <section id="section-model" className="scroll-mt-20">
              <h2 className="text-2xl font-semibold tracking-tight text-neutral-900">
                Model settings
              </h2>
              <p className="mt-1 text-sm text-neutral-600">
                Pick a model architecture and tune its training hyperparameters.
              </p>

              <div className="mt-6 space-y-6 rounded border border-neutral-200 bg-white p-6">
                <label className="block">
                  <span className="mb-1 block text-xs font-medium text-neutral-700">Model</span>
                  <select
                    value={selectedModel}
                    onChange={(e) => handleModelChange(e.target.value)}
                    className="w-full max-w-md rounded border border-neutral-300 bg-white px-3 py-2 text-sm focus:border-neutral-500 focus:outline-none"
                  >
                    <option value="">Select a model…</option>
                    {FL_MODELS.map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.label} — {m.description}
                      </option>
                    ))}
                  </select>
                  {errors.model && (
                    <p className="mt-1 text-xs text-red-600">{errors.model}</p>
                  )}
                </label>

                {hparams && (
                  <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                    <label className="block">
                      <span className="mb-1 block text-xs font-medium text-neutral-700">Optimizer</span>
                      <select
                        value={hparams.optimizer}
                        onChange={(e) => updateHparam('optimizer', e.target.value as 'sgd' | 'adamw')}
                        className="w-full rounded border border-neutral-300 bg-white px-3 py-1.5 text-sm focus:border-neutral-500 focus:outline-none"
                      >
                        <option value="sgd">SGD</option>
                        <option value="adamw">AdamW</option>
                      </select>
                      {errors.optimizer && (
                        <p className="mt-1 text-xs text-red-600">{errors.optimizer}</p>
                      )}
                    </label>
                    {(Object.keys(MODEL_HPARAM_RANGES) as ModelHParamNumericKey[]).map((key) => {
                      const range = MODEL_HPARAM_RANGES[key]
                      return (
                        <label key={key} className="block">
                          <span className="mb-1 block text-xs font-medium text-neutral-700">
                            {MODEL_HPARAM_LABELS[key]}
                          </span>
                          <SliderInput
                            value={hparams[key]}
                            onChange={(v) => updateHparam(key, v)}
                            min={range.min}
                            max={range.max}
                            step={range.step}
                          />
                        </label>
                      )
                    })}
                  </div>
                )}
              </div>
            </section>

            <section id="section-fl" className="scroll-mt-20">
              <h2 className="text-2xl font-semibold tracking-tight text-neutral-900">
                FL settings
              </h2>
              <p className="mt-1 text-sm text-neutral-600">
                Pick an aggregation strategy and tune its parameters.
              </p>

              <div className="mt-6 space-y-6 rounded border border-neutral-200 bg-white p-6">
                <div>
                  <div className="flex flex-wrap gap-2">
                    {FL_STRATEGIES.map((s) => {
                      const isActive = strategy === s.id
                      return (
                        <button
                          key={s.id}
                          type="button"
                          onClick={() => setStrategy(s.id)}
                          className={
                            isActive
                              ? 'rounded bg-neutral-900 px-4 py-2 text-sm font-medium text-white'
                              : 'rounded border border-neutral-300 px-4 py-2 text-sm text-neutral-700 hover:bg-neutral-100'
                          }
                        >
                          {s.label}
                        </button>
                      )
                    })}
                  </div>
                  {errors.strategy && (
                    <p className="mt-2 text-xs text-red-600">{errors.strategy}</p>
                  )}
                </div>

                {strategy && (
                  <div className="space-y-6">
                    <div>
                      <h3 className="text-sm font-semibold text-neutral-900">
                        Common parameters
                      </h3>
                      <div className="mt-3 grid grid-cols-1 gap-4 sm:grid-cols-2">
                        {COMMON_FL_KEYS.map(renderFlField)}
                      </div>
                    </div>

                    {STRATEGY_SPECIFIC_KEYS[strategy].length > 0 && (
                      <div>
                        <h3 className="text-sm font-semibold text-neutral-900">
                          Strategy parameters
                        </h3>
                        <div className="mt-3 grid grid-cols-1 gap-4 sm:grid-cols-2">
                          {STRATEGY_SPECIFIC_KEYS[strategy].map(renderFlField)}
                        </div>
                      </div>
                    )}

                    <div>
                      <h3 className="text-sm font-semibold text-neutral-900">
                        Straggler mitigation
                      </h3>
                      <p className="mt-1 text-xs text-neutral-500">
                        How slow clients are handled in heterogeneous environments.
                      </p>
                      <div className="mt-3 flex flex-wrap gap-2">
                        {STRAGGLER_MODES.map((m) => {
                          const isActive = stragglerParams.mode === m.id
                          return (
                            <button
                              key={m.id}
                              type="button"
                              onClick={() =>
                                setStragglerParams((prev) => ({ ...prev, mode: m.id }))
                              }
                              className={
                                isActive
                                  ? 'rounded bg-neutral-900 px-4 py-2 text-sm font-medium text-white'
                                  : 'rounded border border-neutral-300 px-4 py-2 text-sm text-neutral-700 hover:bg-neutral-100'
                              }
                            >
                              {m.label}
                            </button>
                          )
                        })}
                      </div>
                      <p className="mt-2 text-xs text-neutral-500">
                        {STRAGGLER_MODE_DESCRIPTION[stragglerParams.mode]}
                      </p>

                      {stragglerParams.mode !== 'none' && (
                        <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
                          <label className="block">
                            <span className="mb-1 block text-xs font-medium text-neutral-700">
                              Target
                            </span>
                            <select
                              value={stragglerParams.target}
                              onChange={(e) =>
                                setStragglerParams((prev) => ({
                                  ...prev,
                                  target: e.target.value as StragglerTarget,
                                }))
                              }
                              className="w-full rounded border border-neutral-300 bg-white px-3 py-1.5 text-sm focus:border-neutral-500 focus:outline-none"
                            >
                              <option value="min">min (fastest client)</option>
                              <option value="median">median</option>
                            </select>
                          </label>
                          {MODE_NUMERIC_KEYS[stragglerParams.mode].map((key) => {
                            const range = STRAGGLER_PARAM_RANGES[key]
                            return (
                              <label key={key} className="block">
                                <span className="mb-1 block text-xs font-medium text-neutral-700">
                                  {STRAGGLER_PARAM_LABELS[key]}
                                </span>
                                <SliderInput
                                  value={stragglerParams[key]}
                                  onChange={(v) => updateStragglerNumericParam(key, v)}
                                  min={range.min}
                                  max={range.max}
                                  step={range.step}
                                />
                              </label>
                            )
                          })}
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>

              <div className="mt-6 flex flex-wrap items-center justify-between gap-4 rounded border border-neutral-200 bg-white p-6">
                <div className="text-sm">
                  {Object.keys(errors).length > 0 ? (
                    <span className="text-red-600">
                      Fix {Object.keys(errors).length} error
                      {Object.keys(errors).length > 1 ? 's' : ''} above before saving.
                    </span>
                  ) : runError ? (
                    <span className="text-red-600">{runError}</span>
                  ) : currentRun?.status === 'running' ? (
                    <span className="text-blue-700">
                      Run #{currentRun.id} is running (pid {currentRun.pid}).
                    </span>
                  ) : currentRun?.status === 'completed' ? (
                    <span className="text-green-700">
                      Run #{currentRun.id} completed
                      {currentRun.finished_at
                        ? ` at ${new Date(currentRun.finished_at).toLocaleTimeString()}`
                        : ''}
                      .
                    </span>
                  ) : currentRun?.status === 'failed' ? (
                    <span className="text-red-600">
                      Run #{currentRun.id} failed (exit {currentRun.exit_code}).
                      {currentRun.error_message ? ` ${currentRun.error_message}` : ''}
                    </span>
                  ) : currentRun?.status === 'cancelled' ? (
                    <span className="text-neutral-600">
                      Run #{currentRun.id} cancelled.
                    </span>
                  ) : currentRun?.status === 'draft' ? (
                    <span className="text-green-700">
                      Saved as run #{currentRun.id}. Click Start Training to launch.
                    </span>
                  ) : (
                    <span className="text-neutral-600">
                      Review the configuration and save it before starting a run.
                    </span>
                  )}
                </div>
                <div className="flex gap-3">
                  <button
                    type="button"
                    onClick={handleSave}
                    disabled={busy !== null || currentRun?.status === 'running'}
                    className="rounded bg-orange-500 px-6 py-3 text-base font-medium text-white hover:bg-orange-600 disabled:opacity-50"
                  >
                    {busy === 'save' ? 'Saving…' : 'Save Config'}
                  </button>
                  {currentRun?.status === 'running' ? (
                    <button
                      type="button"
                      onClick={handleCancel}
                      disabled={busy !== null}
                      className="rounded bg-red-600 px-6 py-3 text-base font-medium text-white hover:bg-red-700 disabled:opacity-50"
                    >
                      {busy === 'cancel' ? 'Cancelling…' : 'Cancel'}
                    </button>
                  ) : (
                    <button
                      type="button"
                      onClick={handleStart}
                      disabled={busy !== null || currentRun?.status !== 'draft'}
                      className="rounded bg-green-600 px-6 py-3 text-base font-medium text-white hover:bg-green-700 disabled:opacity-50"
                      title={
                        currentRun?.status === 'draft'
                          ? 'Launch the run'
                          : 'Save the config first'
                      }
                    >
                      {busy === 'start' ? 'Starting…' : 'Start Training'}
                    </button>
                  )}
                </div>
              </div>
            </section>

            <section id="section-training" className="scroll-mt-20">
              <h2 className="text-2xl font-semibold tracking-tight text-neutral-900">
                Training
              </h2>
              <p className="mt-1 text-sm text-neutral-600">
                Live metrics and graphs from the active training run.
              </p>
              <div className="mt-6">
                <TrainingDashboard />
              </div>
            </section>

            <section id="section-models" className="scroll-mt-20">
              <h2 className="text-2xl font-semibold tracking-tight text-neutral-900">
                Models
              </h2>
              <p className="mt-1 text-sm text-neutral-600">
                Models trained in this project. Pick one to serve for inference requests.
              </p>
              <div className="mt-6 rounded border border-dashed border-neutral-300 bg-white p-8 text-center text-sm text-neutral-500">
                No models yet. Models will appear here after a training run completes.
              </div>
            </section>
          </div>
        </main>

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
      </>
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
