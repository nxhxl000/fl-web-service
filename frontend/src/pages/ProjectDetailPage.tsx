import { useEffect, useState, type FormEvent } from 'react'
import { Link, useLocation, useNavigate, useParams } from 'react-router-dom'
import { ApiError } from '../api/client'
import {
  createClientToken,
  deleteClientToken,
  listClientTokens,
} from '../api/clientTokens'
import {
  analyzeDataset,
  browseDirectory,
  deleteProject,
  getProject,
  getProjectAdmin,
  listProjectClients,
  updateProject,
  type DirListing,
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
  deleteRun,
  getRun,
  getRunEvents,
  listRuns,
  startRun,
  type Run,
} from '../api/runs'
import { eventsToExperiment, type SampleExperiment } from '../api/sampleExperiment'
import {
  createTrainedModel,
  deleteTrainedModel,
  listTrainedModels,
  promoteTrainedModel,
  type TrainedModel,
} from '../api/trainedModels'
import {
  MODE_NUMERIC_KEYS,
  STRAGGLER_MODES,
  STRAGGLER_MODE_DESCRIPTION,
  STRAGGLER_PARAM_DEFAULTS,
  STRAGGLER_PARAM_LABELS,
  STRAGGLER_PARAM_RANGES,
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
  { id: 'dataset', label: 'Dataset' },
  { id: 'participants', label: 'Participants' },
  { id: 'model', label: 'Model settings' },
  { id: 'fl', label: 'FL settings' },
  { id: 'training', label: 'Training' },
  { id: 'models', label: 'Models' },
  { id: 'runs', label: 'Runs' },
] as const

type SectionId = typeof ADMIN_SECTIONS[number]['id']

function scrollToSection(id: SectionId) {
  document
    .getElementById(`section-${id}`)
    ?.scrollIntoView({ behavior: 'smooth', block: 'start' })
}

function DatasetSpecs({ info }: { info: NonNullable<Project['test_dataset_info']> }) {
  const sizeLabel = info.image_size ? `${info.image_size[0]}×${info.image_size[1]}` : '—'
  return (
    <div className="mt-4 rounded border border-neutral-200 bg-white p-5">
      <h3 className="text-sm font-semibold text-neutral-900">Dataset specs</h3>
      <p className="mt-1 text-xs text-neutral-500">
        Format: <code className="font-mono">{info.format}</code>
      </p>
      <div className="mt-3 flex flex-wrap gap-3 text-xs">
        <div className="rounded-full border border-neutral-200 bg-neutral-50 px-3 py-1">
          <span className="text-neutral-500">Samples:</span>{' '}
          <span className="font-medium text-neutral-900">{info.num_samples}</span>
        </div>
        <div className="rounded-full border border-neutral-200 bg-neutral-50 px-3 py-1">
          <span className="text-neutral-500">Classes:</span>{' '}
          <span className="font-medium text-neutral-900">{info.num_classes}</span>
        </div>
        <div className="rounded-full border border-neutral-200 bg-neutral-50 px-3 py-1">
          <span className="text-neutral-500">Image size:</span>{' '}
          <span className="font-medium text-neutral-900">{sizeLabel}</span>
        </div>
        {info.image_mode && (
          <div className="rounded-full border border-neutral-200 bg-neutral-50 px-3 py-1">
            <span className="text-neutral-500">Image mode:</span>{' '}
            <span className="font-medium text-neutral-900">{info.image_mode}</span>
          </div>
        )}
      </div>
      <div className="mt-4">
        <p className="text-xs font-medium text-neutral-700">Class labels</p>
        <div className="mt-2 max-h-40 overflow-y-auto rounded border border-neutral-200 bg-neutral-50 p-3 font-mono text-xs leading-relaxed text-neutral-800">
          {info.class_names.length > 0 ? info.class_names.join(', ') : '—'}
        </div>
      </div>
    </div>
  )
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
  const location = useLocation()
  const { user } = useAuth()
  const isAdmin = user?.is_admin === true

  const [project, setProject] = useState<Project | null>(null)
  const [tokens, setTokens] = useState<ClientToken[]>([])
  const [allClients, setAllClients] = useState<ClientTokenWithOwner[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!location.hash || !project) return
    const id = location.hash.slice(1)
    const t = setTimeout(() => {
      document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }, 100)
    return () => clearTimeout(t)
  }, [location.hash, project])

  const [newName, setNewName] = useState('')
  const [creating, setCreating] = useState(false)
  const [justCreated, setJustCreated] = useState<{
    id: number
    token: string
    docker_command: string
  } | null>(null)

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
  const [federation, setFederation] = useState<'local-sim' | 'remote'>('local-sim')

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
    }

    // Real-distributed only: SuperLink waits forever if min-nodes > online clients,
    // and 0 makes the run start without anyone to do the training. Sim federations
    // spin up virtual nodes from --num-supernodes, so this guard doesn't apply.
    if (federation === 'remote') {
      const onlineCount = allClients.filter((c) => isReady(c.last_seen_at)).length
      const mt = flParams['min-train-nodes']
      const ma = flParams['min-available-nodes']
      if (mt < 1) errs['min-train-nodes'] = 'Must be ≥ 1.'
      else if (mt > onlineCount)
        errs['min-train-nodes'] = `Only ${onlineCount} client(s) online — cannot exceed.`
      if (ma < 1) errs['min-available-nodes'] = 'Must be ≥ 1.'
      else if (ma > onlineCount)
        errs['min-available-nodes'] = `Only ${onlineCount} client(s) online — cannot exceed.`
    }

    return errs
  }

  const buildRunConfig = (): Record<string, unknown> => {
    const cfg: Record<string, unknown> = {
      model: selectedModel,
      aggregation: strategy,
      ...flParams,
      ...(hparams ?? {}),
      'straggler-mode': stragglerParams.mode,
      'straggler-target': stragglerParams.target,
      'straggler-tolerance': stragglerParams.tolerance,
      'straggler-drop-tolerance': stragglerParams['drop-tolerance'],
      'straggler-max-dropped': stragglerParams['max-dropped'],
      'straggler-min-chunk': stragglerParams['min-chunk'],
      'straggler-min-epochs': stragglerParams['min-epochs'],
    }
    return cfg
  }

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
        federation,
        run_config: buildRunConfig(),
      })
      setCurrentRun(run)
      void refreshRuns()
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
      void refreshRuns()
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
      void refreshRuns()
    } catch (err) {
      setRunError(err instanceof ApiError ? err.detail : 'Failed to cancel run')
    } finally {
      setBusy(null)
    }
  }

  const [models, setModels] = useState<TrainedModel[]>([])
  const [selectedModelId, setSelectedModelId] = useState<number | null>(null)
  const [modelsBusy, setModelsBusy] = useState<'add' | 'delete' | 'promote' | null>(null)
  const [modelsError, setModelsError] = useState<string | null>(null)
  const [showAddForm, setShowAddForm] = useState(false)
  const [addForm, setAddForm] = useState({
    display_name: '',
    model_name: 'wrn_16_4',
    dataset: 'cifar100',
    weights_path: '',
    accuracy: '',
    f1_score: '',
    num_rounds: '',
  })

  const [runs, setRuns] = useState<Run[]>([])
  const [runsBusy, setRunsBusy] = useState<number | null>(null)
  const [runsError, setRunsError] = useState<string | null>(null)

  const [runExperiment, setRunExperiment] = useState<SampleExperiment | null>(null)
  const [runExperimentError, setRunExperimentError] = useState<string | null>(null)
  const [runExperimentLoading, setRunExperimentLoading] = useState(false)

  const [datasetPath, setDatasetPath] = useState('')
  const [datasetBusy, setDatasetBusy] = useState(false)
  const [datasetError, setDatasetError] = useState<string | null>(null)
  const [browser, setBrowser] = useState<DirListing | null>(null)
  const [browserError, setBrowserError] = useState<string | null>(null)

  const openBrowser = async (startPath = '') => {
    setBrowserError(null)
    try {
      setBrowser(await browseDirectory(startPath))
    } catch (err) {
      setBrowserError(err instanceof ApiError ? err.detail : 'Failed to browse')
      setBrowser({ path: '', parent: null, subdirs: [] })
    }
  }

  const navigateBrowser = async (path: string) => {
    setBrowserError(null)
    try {
      setBrowser(await browseDirectory(path))
    } catch (err) {
      setBrowserError(err instanceof ApiError ? err.detail : 'Failed to browse')
    }
  }

  useEffect(() => {
    if (!isAdmin || !Number.isFinite(pid)) return
    getProjectAdmin(pid)
      .then((p) => setDatasetPath(p.test_dataset_path ?? ''))
      .catch(() => undefined)
  }, [pid, isAdmin])

  // Refresh participants list periodically so the heartbeat-driven Online/
  // Offline indicator stays accurate without manual Check clicks. Interval
  // must be < READY_THRESHOLD_MS so a live client never appears stale.
  useEffect(() => {
    if (!isAdmin || !Number.isFinite(pid)) return
    const interval = setInterval(() => {
      listProjectClients(pid)
        .then(setAllClients)
        .catch(() => undefined)
    }, 30000)
    return () => clearInterval(interval)
  }, [pid, isAdmin])

  const handleAnalyzeDataset = async () => {
    setDatasetError(null)
    if (!datasetPath.trim()) {
      setDatasetError('Path is required.')
      return
    }
    setDatasetBusy(true)
    try {
      const updated = await analyzeDataset(pid, datasetPath.trim())
      setProject(updated)
      setDatasetPath(updated.test_dataset_path ?? '')
    } catch (err) {
      setDatasetError(err instanceof ApiError ? err.detail : 'Failed to analyze dataset')
    } finally {
      setDatasetBusy(false)
    }
  }

  useEffect(() => {
    if (!isAdmin || !Number.isFinite(pid)) return
    listTrainedModels(pid).then(setModels).catch(() => undefined)
    listRuns(pid)
      .then((rs) => {
        setRuns(rs)
        // Restore the most relevant run on page load so the dashboard survives reloads.
        // Priority: running run → most recent (rs is created_at-desc).
        if (rs.length === 0) return
        const running = rs.find((r) => r.status === 'running')
        setCurrentRun(running ?? rs[0])
      })
      .catch(() => undefined)
  }, [pid, isAdmin])

  const refreshModels = async () => {
    try {
      setModels(await listTrainedModels(pid))
    } catch {
      // ignore
    }
  }

  const refreshRuns = async () => {
    try {
      setRuns(await listRuns(pid))
    } catch {
      // ignore
    }
  }

  const handleDeleteRun = async (runId: number) => {
    if (!confirm(`Delete run #${runId}? This removes the run record and runs_data/run_${runId}/.`)) {
      return
    }
    setRunsError(null)
    setRunsBusy(runId)
    try {
      await deleteRun(pid, runId)
      await refreshRuns()
    } catch (err) {
      setRunsError(err instanceof ApiError ? err.detail : 'Failed to delete run')
    } finally {
      setRunsBusy(null)
    }
  }

  const handleAddModel = async () => {
    setModelsError(null)
    if (!addForm.display_name.trim() || !addForm.weights_path.trim()) {
      setModelsError('Name and weights path are required.')
      return
    }
    setModelsBusy('add')
    try {
      await createTrainedModel(pid, {
        display_name: addForm.display_name.trim(),
        model_name: addForm.model_name,
        dataset: addForm.dataset,
        weights_path: addForm.weights_path.trim(),
        accuracy: addForm.accuracy.trim() ? Number(addForm.accuracy) : null,
        f1_score: addForm.f1_score.trim() ? Number(addForm.f1_score) : null,
        num_rounds: addForm.num_rounds.trim() ? Number(addForm.num_rounds) : null,
      })
      setAddForm({
        display_name: '',
        model_name: 'wrn_16_4',
        dataset: 'cifar100',
        weights_path: '',
        accuracy: '',
        f1_score: '',
        num_rounds: '',
      })
      setShowAddForm(false)
      await refreshModels()
    } catch (err) {
      setModelsError(err instanceof ApiError ? err.detail : 'Failed to add model')
    } finally {
      setModelsBusy(null)
    }
  }

  const handleDeleteModel = async (modelId: number) => {
    if (!confirm('Delete this model from the registry?')) return
    setModelsError(null)
    setModelsBusy('delete')
    try {
      const updatedProject = await deleteTrainedModel(pid, modelId)
      setProject(updatedProject)
      if (selectedModelId === modelId) setSelectedModelId(null)
      await refreshModels()
    } catch (err) {
      setModelsError(err instanceof ApiError ? err.detail : 'Failed to delete model')
    } finally {
      setModelsBusy(null)
    }
  }

  const handlePromote = async () => {
    if (selectedModelId === null) return
    setModelsError(null)
    setModelsBusy('promote')
    try {
      const updatedProject = await promoteTrainedModel(pid, selectedModelId)
      setProject(updatedProject)
    } catch (err) {
      setModelsError(err instanceof ApiError ? err.detail : 'Failed to promote model')
    } finally {
      setModelsBusy(null)
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
          if (updated.status === 'completed') {
            void refreshModels()
          }
          void refreshRuns()
        }
      } catch {
        // ignore transient errors; next tick will retry
      }
    }, 2000)
    return () => clearInterval(interval)
  }, [currentRun?.id, currentRun?.status, pid])

  // Live events polling for the active run — drives TrainingDashboard.
  useEffect(() => {
    if (!currentRun) {
      setRunExperiment(null)
      setRunExperimentError(null)
      return
    }
    const runId = currentRun.id
    let cancelled = false

    const fetchOnce = async (showLoading: boolean) => {
      if (showLoading) setRunExperimentLoading(true)
      try {
        const events = await getRunEvents(pid, runId)
        if (cancelled) return
        setRunExperiment(eventsToExperiment(events))
        setRunExperimentError(null)
      } catch (err) {
        if (cancelled) return
        setRunExperimentError(err instanceof ApiError ? err.detail : 'Failed to load events')
      } finally {
        if (!cancelled) setRunExperimentLoading(false)
      }
    }

    void fetchOnce(true)
    if (currentRun.status !== 'running') return () => { cancelled = true }

    const interval = setInterval(() => void fetchOnce(false), 5000)
    return () => {
      cancelled = true
      clearInterval(interval)
    }
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
      setJustCreated({
        id: created.id,
        token: created.token,
        docker_command: created.docker_command,
      })
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

        <main className="ml-64 px-8 pb-[50vh] pt-10">
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

            <section id="section-dataset" className="scroll-mt-20">
              <h2 className="text-2xl font-semibold tracking-tight text-neutral-900">Dataset</h2>
              <p className="mt-1 text-sm text-neutral-600">
                Point to a folder on the server containing the test set. Analyze inspects the
                directory and publishes the spec (classes, image size, sample count) so participants
                see exactly what their data needs to look like.
              </p>

              <div className="mt-4 space-y-3 rounded border border-neutral-200 bg-white p-5">
                <label className="block">
                  <span className="mb-1 block text-xs font-medium text-neutral-700">
                    Test dataset path (server-local)
                  </span>
                  <input
                    type="text"
                    value={datasetPath}
                    onChange={(e) => setDatasetPath(e.target.value)}
                    placeholder="data/partitions/<partition-name>/test"
                    className="w-full rounded border border-neutral-300 px-3 py-2 font-mono text-sm focus:border-neutral-500 focus:outline-none"
                  />
                </label>
                <p className="text-xs text-neutral-500">
                  Supported: HuggingFace <code>load_from_disk</code> directories or ImageFolder
                  layout (<code>{`<root>/<class_name>/*.png`}</code>).
                </p>
                <div className="flex items-center gap-3">
                  <button
                    type="button"
                    onClick={handleAnalyzeDataset}
                    disabled={datasetBusy}
                    className="rounded bg-neutral-900 px-4 py-2 text-sm font-medium text-white hover:bg-neutral-700 disabled:opacity-50"
                  >
                    {datasetBusy ? 'Analyzing…' : 'Analyze'}
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      if (browser) {
                        setBrowser(null)
                      } else {
                        void openBrowser(datasetPath || '')
                      }
                    }}
                    className="rounded border border-neutral-300 px-4 py-2 text-sm text-neutral-700 hover:bg-neutral-100"
                  >
                    {browser ? 'Close browser' : 'Browse…'}
                  </button>
                  {datasetError && (
                    <span className="text-xs text-red-600">{datasetError}</span>
                  )}
                </div>

                {browser && (
                  <div className="mt-2 rounded border border-neutral-200 bg-neutral-50 p-3">
                    <div className="flex items-center justify-between gap-3 text-xs">
                      <span className="font-mono text-neutral-700">
                        /{browser.path}
                      </span>
                      <div className="flex gap-2">
                        {browser.parent !== null && (
                          <button
                            type="button"
                            onClick={() => void navigateBrowser(browser.parent ?? '')}
                            className="rounded border border-neutral-300 px-2 py-1 text-neutral-700 hover:bg-white"
                          >
                            ↑ Up
                          </button>
                        )}
                        <button
                          type="button"
                          onClick={() => {
                            setDatasetPath(browser.path)
                            setBrowser(null)
                          }}
                          className="rounded bg-green-600 px-2 py-1 text-white hover:bg-green-700"
                        >
                          Use this folder
                        </button>
                      </div>
                    </div>
                    {browserError && (
                      <p className="mt-2 text-xs text-red-600">{browserError}</p>
                    )}
                    {browser.subdirs.length === 0 ? (
                      <p className="mt-2 text-xs text-neutral-500">
                        (no subfolders here — use this one or go up)
                      </p>
                    ) : (
                      <ul className="mt-2 max-h-64 overflow-y-auto divide-y divide-neutral-200 text-sm">
                        {browser.subdirs.map((d) => {
                          const newPath = browser.path ? `${browser.path}/${d}` : d
                          return (
                            <li key={d}>
                              <button
                                type="button"
                                onClick={() => void navigateBrowser(newPath)}
                                className="block w-full px-2 py-1.5 text-left font-mono text-xs text-neutral-800 hover:bg-white"
                              >
                                📁 {d}
                              </button>
                            </li>
                          )
                        })}
                      </ul>
                    )}
                  </div>
                )}
              </div>

              {project.test_dataset_info && (
                <DatasetSpecs info={project.test_dataset_info} />
              )}
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

              {runError && (
                <div className="mt-6 flex items-start gap-3 rounded border border-red-300 bg-red-50 p-4">
                  <span className="mt-0.5 text-xl leading-none text-red-600">⚠</span>
                  <div className="flex-1">
                    <p className="text-sm font-semibold text-red-900">
                      Cannot start the run
                    </p>
                    <p className="mt-1 whitespace-pre-line text-sm text-red-800">
                      {runError}
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() => setRunError(null)}
                    className="text-red-700 hover:text-red-900"
                    aria-label="Dismiss error"
                  >
                    ✕
                  </button>
                </div>
              )}

              <div className="mt-6 flex flex-wrap items-center justify-between gap-4 rounded border border-neutral-200 bg-white p-6">
                <div className="text-sm">
                  {Object.keys(errors).length > 0 ? (
                    <span className="text-red-600">
                      Fix {Object.keys(errors).length} error
                      {Object.keys(errors).length > 1 ? 's' : ''} above before saving.
                    </span>
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
                <div className="flex items-center gap-3">
                  <label className="flex items-center gap-2 text-sm text-neutral-700">
                    <span>Federation:</span>
                    <select
                      value={federation}
                      onChange={(e) => setFederation(e.target.value as 'local-sim' | 'remote')}
                      disabled={busy !== null || currentRun?.status === 'running'}
                      className="rounded border border-neutral-300 bg-white px-2 py-1.5 text-sm disabled:opacity-50"
                    >
                      <option value="local-sim">local-sim</option>
                      <option value="remote">remote</option>
                    </select>
                  </label>
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
                {currentRun
                  ? `Run #${currentRun.id} — ${currentRun.status}${
                      currentRun.status === 'running' ? ' (polling every 5s)' : ''
                    }`
                  : 'No active run yet. Configure the run above and click Start Training.'}
              </p>
              <div className="mt-6">
                {currentRun && (
                  <TrainingDashboard
                    data={runExperiment}
                    loading={runExperimentLoading}
                    error={runExperimentError}
                  />
                )}
              </div>
            </section>

            <section id="section-models" className="scroll-mt-20">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <h2 className="text-2xl font-semibold tracking-tight text-neutral-900">Models</h2>
                  <p className="mt-1 text-sm text-neutral-600">
                    All trained checkpoints registered for this project. Pick one and promote it to
                    serve as the public inference target.
                  </p>
                </div>
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={handlePromote}
                    disabled={
                      selectedModelId === null ||
                      selectedModelId === project.inference_target_id ||
                      modelsBusy !== null
                    }
                    className="rounded bg-green-600 px-4 py-2 text-sm font-medium text-white hover:bg-green-700 disabled:opacity-50"
                    title={
                      selectedModelId === null
                        ? 'Pick a model row first'
                        : selectedModelId === project.inference_target_id
                          ? 'Already the inference target'
                          : 'Promote selected model'
                    }
                  >
                    {modelsBusy === 'promote' ? 'Promoting…' : 'Set as inference target'}
                  </button>
                  <button
                    type="button"
                    onClick={() => setShowAddForm((v) => !v)}
                    className="rounded border border-neutral-300 px-4 py-2 text-sm text-neutral-700 hover:bg-neutral-100"
                  >
                    {showAddForm ? 'Cancel' : '+ Add model'}
                  </button>
                </div>
              </div>

              {modelsError && (
                <p className="mt-3 text-sm text-red-600">{modelsError}</p>
              )}

              {showAddForm && (
                <div className="mt-4 space-y-3 rounded border border-neutral-200 bg-white p-5">
                  <h3 className="text-sm font-semibold text-neutral-900">Register existing checkpoint</h3>
                  <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                    <label className="block sm:col-span-2">
                      <span className="mb-1 block text-xs font-medium text-neutral-700">Display name</span>
                      <input
                        type="text"
                        value={addForm.display_name}
                        onChange={(e) => setAddForm({ ...addForm, display_name: e.target.value })}
                        placeholder="WideResNet baseline (round 80)"
                        className="w-full rounded border border-neutral-300 px-3 py-1.5 text-sm focus:border-neutral-500 focus:outline-none"
                      />
                    </label>
                    <label className="block">
                      <span className="mb-1 block text-xs font-medium text-neutral-700">Model architecture</span>
                      <select
                        value={addForm.model_name}
                        onChange={(e) => setAddForm({ ...addForm, model_name: e.target.value })}
                        className="w-full rounded border border-neutral-300 bg-white px-3 py-1.5 text-sm"
                      >
                        <option value="wrn_16_4">WideResNet 16-4</option>
                        <option value="se_resnet">SE-ResNet</option>
                        <option value="effnet_b0">EfficientNet-B0</option>
                      </select>
                    </label>
                    <label className="block">
                      <span className="mb-1 block text-xs font-medium text-neutral-700">Dataset</span>
                      <select
                        value={addForm.dataset}
                        onChange={(e) => setAddForm({ ...addForm, dataset: e.target.value })}
                        className="w-full rounded border border-neutral-300 bg-white px-3 py-1.5 text-sm"
                      >
                        <option value="cifar100">CIFAR-100</option>
                        <option value="plantvillage">PlantVillage</option>
                      </select>
                    </label>
                    <label className="block sm:col-span-2">
                      <span className="mb-1 block text-xs font-medium text-neutral-700">
                        Weights path (relative to repo root, or absolute)
                      </span>
                      <input
                        type="text"
                        value={addForm.weights_path}
                        onChange={(e) => setAddForm({ ...addForm, weights_path: e.target.value })}
                        placeholder="inference_models/wrn_16_4.pt"
                        className="w-full rounded border border-neutral-300 px-3 py-1.5 font-mono text-sm"
                      />
                    </label>
                    <label className="block">
                      <span className="mb-1 block text-xs font-medium text-neutral-700">Accuracy (0–1)</span>
                      <input
                        type="number" step="0.0001" min="0" max="1"
                        value={addForm.accuracy}
                        onChange={(e) => setAddForm({ ...addForm, accuracy: e.target.value })}
                        className="w-full rounded border border-neutral-300 px-3 py-1.5 text-sm"
                      />
                    </label>
                    <label className="block">
                      <span className="mb-1 block text-xs font-medium text-neutral-700">F1 score (0–1)</span>
                      <input
                        type="number" step="0.0001" min="0" max="1"
                        value={addForm.f1_score}
                        onChange={(e) => setAddForm({ ...addForm, f1_score: e.target.value })}
                        className="w-full rounded border border-neutral-300 px-3 py-1.5 text-sm"
                      />
                    </label>
                    <label className="block">
                      <span className="mb-1 block text-xs font-medium text-neutral-700">Rounds completed</span>
                      <input
                        type="number" step="1" min="0"
                        value={addForm.num_rounds}
                        onChange={(e) => setAddForm({ ...addForm, num_rounds: e.target.value })}
                        className="w-full rounded border border-neutral-300 px-3 py-1.5 text-sm"
                      />
                    </label>
                  </div>
                  <div>
                    <button
                      type="button"
                      onClick={handleAddModel}
                      disabled={modelsBusy !== null}
                      className="rounded bg-neutral-900 px-4 py-2 text-sm font-medium text-white hover:bg-neutral-700 disabled:opacity-50"
                    >
                      {modelsBusy === 'add' ? 'Adding…' : 'Register model'}
                    </button>
                  </div>
                </div>
              )}

              {models.length === 0 ? (
                <div className="mt-6 rounded border border-dashed border-neutral-300 bg-white p-8 text-center text-sm text-neutral-500">
                  No models registered yet. Click “+ Add model” to register an existing checkpoint.
                </div>
              ) : (
                <div className="mt-6 overflow-x-auto rounded border border-neutral-200 bg-white">
                  <table className="min-w-full text-left text-sm">
                    <thead className="bg-neutral-50 text-xs uppercase text-neutral-500">
                      <tr>
                        <th className="px-4 py-2 font-medium" />
                        <th className="px-4 py-2 font-medium">Name</th>
                        <th className="px-4 py-2 font-medium">Model</th>
                        <th className="px-4 py-2 font-medium">Dataset</th>
                        <th className="px-4 py-2 font-medium">Accuracy</th>
                        <th className="px-4 py-2 font-medium">F1</th>
                        <th className="px-4 py-2 font-medium">Created</th>
                        <th className="px-4 py-2 font-medium" />
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-neutral-200">
                      {models.map((m) => {
                        const isCurrent = m.id === project.inference_target_id
                        return (
                          <tr
                            key={m.id}
                            className={selectedModelId === m.id ? 'bg-neutral-50' : ''}
                          >
                            <td className="px-4 py-2">
                              <input
                                type="radio"
                                name="model-select"
                                checked={selectedModelId === m.id}
                                onChange={() => setSelectedModelId(m.id)}
                              />
                            </td>
                            <td className="px-4 py-2">
                              <div className="font-medium text-neutral-900">{m.display_name}</div>
                              {isCurrent && (
                                <span className="mt-0.5 inline-block rounded-full bg-green-100 px-2 py-0.5 text-xs font-medium text-green-800">
                                  Currently serving
                                </span>
                              )}
                            </td>
                            <td className="px-4 py-2 font-mono text-xs text-neutral-700">{m.model_name}</td>
                            <td className="px-4 py-2 font-mono text-xs text-neutral-700">
                              {project.test_dataset_info?.name ?? m.dataset}
                            </td>
                            <td className="px-4 py-2 text-neutral-700">
                              {m.accuracy != null ? `${(m.accuracy * 100).toFixed(1)}%` : '—'}
                            </td>
                            <td className="px-4 py-2 text-neutral-700">
                              {m.f1_score != null ? `${(m.f1_score * 100).toFixed(1)}%` : '—'}
                            </td>
                            <td className="px-4 py-2 text-xs text-neutral-500">
                              {new Date(m.created_at).toLocaleDateString()}
                            </td>
                            <td className="px-4 py-2">
                              <div className="flex gap-2">
                                <button
                                  type="button"
                                  disabled={m.run_id == null}
                                  title={
                                    m.run_id == null
                                      ? 'Manually imported — no run data to show'
                                      : 'Show this run on the Training dashboard'
                                  }
                                  onClick={() => {
                                    if (m.run_id == null) return
                                    const linkedRun = runs.find((r) => r.id === m.run_id)
                                    if (linkedRun) {
                                      setCurrentRun(linkedRun)
                                      scrollToSection('training')
                                    }
                                  }}
                                  className="rounded border border-neutral-300 px-2 py-1 text-xs text-neutral-700 hover:bg-neutral-100 disabled:opacity-50"
                                >
                                  Details
                                </button>
                                <button
                                  type="button"
                                  onClick={() => handleDeleteModel(m.id)}
                                  disabled={modelsBusy !== null}
                                  className="rounded border border-red-300 px-2 py-1 text-xs text-red-700 hover:bg-red-50 disabled:opacity-50"
                                >
                                  Delete
                                </button>
                              </div>
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              )}

            </section>

            <section id="section-runs" className="scroll-mt-20">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <h2 className="text-2xl font-semibold tracking-tight text-neutral-900">Runs</h2>
                  <p className="mt-1 text-sm text-neutral-600">
                    Every training run for this project. Completed runs that produced a model are
                    cleaned up via the Models section (deleting the model also removes its run).
                    Use this list to wipe failed/cancelled/draft runs.
                  </p>
                </div>
              </div>

              {runsError && <p className="mt-3 text-sm text-red-600">{runsError}</p>}

              {runs.length === 0 ? (
                <p className="mt-4 text-sm text-neutral-500">No runs yet.</p>
              ) : (
                <div className="mt-4 overflow-x-auto rounded border border-neutral-200 bg-white">
                  <table className="min-w-full divide-y divide-neutral-200 text-sm">
                    <thead className="bg-neutral-50 text-left text-xs uppercase tracking-wide text-neutral-500">
                      <tr>
                        <th className="px-4 py-2">ID</th>
                        <th className="px-4 py-2">Status</th>
                        <th className="px-4 py-2">Federation</th>
                        <th className="px-4 py-2">Model / Aggregation</th>
                        <th className="px-4 py-2">Started</th>
                        <th className="px-4 py-2">Finished</th>
                        <th className="px-4 py-2"></th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-neutral-100">
                      {runs.map((r) => {
                        const linkedModel = models.find((m) => m.run_id === r.id)
                        return (
                          <tr key={r.id}>
                            <td className="px-4 py-2 font-mono text-neutral-900">#{r.id}</td>
                            <td className="px-4 py-2">
                              <span
                                className={
                                  r.status === 'running'
                                    ? 'rounded-full bg-blue-100 px-2 py-0.5 text-xs text-blue-700'
                                    : r.status === 'completed'
                                      ? 'rounded-full bg-green-100 px-2 py-0.5 text-xs text-green-700'
                                      : r.status === 'failed'
                                        ? 'rounded-full bg-red-100 px-2 py-0.5 text-xs text-red-700'
                                        : 'rounded-full bg-neutral-100 px-2 py-0.5 text-xs text-neutral-600'
                                }
                              >
                                {r.status}
                              </span>
                            </td>
                            <td className="px-4 py-2 font-mono text-xs text-neutral-700">{r.federation}</td>
                            <td className="px-4 py-2 font-mono text-xs text-neutral-700">
                              {String(r.run_config.model ?? '—')}/
                              {String(r.run_config.aggregation ?? '—')}
                            </td>
                            <td className="px-4 py-2 text-xs text-neutral-600">
                              {r.started_at ? new Date(r.started_at).toLocaleString() : '—'}
                            </td>
                            <td className="px-4 py-2 text-xs text-neutral-600">
                              {r.finished_at ? new Date(r.finished_at).toLocaleString() : '—'}
                            </td>
                            <td className="px-4 py-2 text-right">
                              {linkedModel ? (
                                <button
                                  type="button"
                                  onClick={() => scrollToSection('models')}
                                  title="This run produced a model — delete it via Models, the run will be removed too."
                                  className="text-xs text-neutral-500 hover:text-neutral-900 hover:underline"
                                >
                                  → delete via Models
                                </button>
                              ) : (
                                <button
                                  type="button"
                                  onClick={() => handleDeleteRun(r.id)}
                                  disabled={runsBusy === r.id || r.status === 'running'}
                                  title={
                                    r.status === 'running'
                                      ? 'Cancel the run first'
                                      : 'Delete run record + run dir'
                                  }
                                  className="rounded border border-red-300 px-2 py-1 text-xs text-red-700 hover:bg-red-50 disabled:opacity-50"
                                >
                                  {runsBusy === r.id ? '…' : 'Delete'}
                                </button>
                              )}
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              )}
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

      {project.test_dataset_info && (
        <section className="mt-4">
          <h2 className="text-sm font-semibold text-neutral-900">
            Your dataset must match these specs
          </h2>
          <DatasetSpecs info={project.test_dataset_info} />
        </section>
      )}

      <section id="join" className="mt-8 scroll-mt-20">
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
          <div className="mt-4 space-y-3 rounded border border-amber-300 bg-amber-50 p-4">
            <div>
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
            <div>
              <p className="text-sm font-medium text-amber-900">
                Run this on the machine that holds your dataset. Replace{' '}
                <code className="font-mono text-xs">/PATH/TO/YOUR/DATA</code> with the
                folder containing your class subdirectories.
              </p>
              <div className="mt-2 flex items-start gap-2">
                <pre className="flex-1 overflow-x-auto rounded bg-neutral-900 px-3 py-2 font-mono text-xs leading-relaxed text-neutral-100">
                  {justCreated.docker_command}
                </pre>
                <button
                  type="button"
                  onClick={() =>
                    navigator.clipboard.writeText(justCreated.docker_command)
                  }
                  className="shrink-0 rounded border border-neutral-300 bg-white px-3 py-2 text-xs text-neutral-700 hover:bg-neutral-100"
                >
                  Copy
                </button>
              </div>
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
