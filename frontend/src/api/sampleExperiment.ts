export type RoundRow = {
  round: number
  test_loss: number
  test_acc: number
  test_f1: number
  train_loss_first_mean: number
  train_loss_last_mean: number
  t_compute_mean: number
  drift_mean: number
  update_norm_rel_mean: number
  grad_norm_last_mean: number
  delta_norm: number
  momentum_norm: number
  c_server_norm: number
  comm_mb: number
  SR: number
  IF: number
  I_s: number
  T_min: number
  T_max: number
  W_total: number
  W_imbalance: number
  n_dropped: number
}

export type ClientRow = {
  round: number
  partition_id: number
  node_name?: string
  num_examples: number
  chunk_fraction: number
  local_epochs: number
  w_client: number
  train_loss_first: number
  train_loss_last: number
  t_compute: number
  t_serialize: number
  t_local: number
  created_at: number
  t_aggr_start: number
  t_lifecycle: number
  w_drift: number
  update_norm_rel: number
  grad_norm_last: number
}

export type ScheduleData = {
  mode: 'none' | 'chunk' | 'epochs' | 'drop'
  target: string
  T_target: number
  T_upper: number
  tolerance: number
  base_epochs: number
  chunks: Record<string, number>
  epochs: Record<string, number>
  excluded?: number[]
}

export type SummaryData = {
  config: Record<string, unknown>
  best_acc: number
  best_round: number
  rounds_completed: number
  num_rounds: number
  data_heterogeneity: {
    MPJS: number
    Gini_quantity: number
    num_classes: number
  }
  system_heterogeneity_mean: {
    SR: number
    IF: number
    I_s: number
    W_total_sum: number
  }
  // Wall-clock duration (seconds). `finished_at_ts` is null while the run is
  // still in progress; the dashboard computes elapsed against now in that case.
  started_at_ts: number | null
  finished_at_ts: number | null
  // Filled from run_done. Empty array while the run is still in progress.
  per_class_accuracy: { class_id: number; name: string; accuracy: number }[]
}

export type SampleExperiment = {
  rounds: RoundRow[]
  clients: ClientRow[]
  schedule: ScheduleData
  summary: SummaryData
}

function parseCsv<T extends Record<string, number | string>>(text: string): T[] {
  const lines = text.trim().split(/\r?\n/)
  if (lines.length < 2) return []
  const headers = lines[0].split(',').map((h) => h.trim())
  return lines.slice(1).map((line) => {
    const values = line.split(',').map((v) => v.trim())
    const row: Record<string, number | string> = {}
    headers.forEach((h, i) => {
      const raw = values[i]
      const num = Number(raw)
      row[h] = raw === '' || isNaN(num) ? raw : num
    })
    return row as T
  })
}

export async function loadSampleExperiment(): Promise<SampleExperiment> {
  const base = '/sample-experiment'
  const [roundsText, clientsText, scheduleText, summaryText] = await Promise.all([
    fetch(`${base}/rounds.csv`).then((r) => r.text()),
    fetch(`${base}/clients.csv`).then((r) => r.text()),
    fetch(`${base}/schedule.json`).then((r) => r.text()),
    fetch(`${base}/summary.json`).then((r) => r.text()),
  ])

  return {
    rounds: parseCsv<RoundRow>(roundsText),
    clients: parseCsv<ClientRow>(clientsText),
    schedule: JSON.parse(scheduleText),
    summary: JSON.parse(summaryText),
  }
}

type RawEvent = Record<string, unknown> & { type: string }

const num = (v: unknown, fallback = 0): number => (typeof v === 'number' ? v : fallback)

export function eventsToExperiment(events: RawEvent[]): SampleExperiment {
  const rounds: RoundRow[] = []
  const clients: ClientRow[] = []
  let schedule: ScheduleData = {
    mode: 'none',
    target: 'min',
    T_target: 0,
    T_upper: 0,
    tolerance: 0,
    base_epochs: 0,
    chunks: {},
    epochs: {},
  }
  let dataHet = { MPJS: 0, Gini_quantity: 0, num_classes: 100 }
  let runStartedConfig: Record<string, unknown> = {}
  let runDone: RawEvent | null = null
  let startedAtTs: number | null = null
  let lastRoundTs: number | null = null

  for (const e of events) {
    if (e.type === 'run_started') {
      runStartedConfig = (e.config as Record<string, unknown>) ?? {}
      startedAtTs = num(e.ts) || null
    } else if (e.type === 'data_heterogeneity') {
      dataHet = {
        MPJS: num(e.mpjs),
        Gini_quantity: num(e.gini_quantity),
        num_classes: num(e.num_classes, 100),
      }
    } else if (e.type === 'schedule') {
      schedule = {
        mode: (e.mode as ScheduleData['mode']) ?? 'none',
        target: String(e.target ?? 'min'),
        T_target: num(e.T_target),
        T_upper: num(e.T_upper),
        tolerance: num(e.tolerance),
        base_epochs: num(e.base_epochs),
        chunks: (e.chunks as Record<string, number>) ?? {},
        epochs: (e.epochs as Record<string, number>) ?? {},
        excluded: e.excluded as number[] | undefined,
      }
    } else if (e.type === 'round') {
      lastRoundTs = num(e.ts) || lastRoundTs
      const sh = (e.system_het as Record<string, unknown>) ?? {}
      const st = (e.strategy as Record<string, unknown>) ?? {}
      rounds.push({
        round: num(e.round),
        test_loss: num(e.test_loss),
        test_acc: num(e.test_acc),
        test_f1: num(e.test_f1),
        train_loss_first_mean: num(e.train_loss_first_mean),
        train_loss_last_mean: num(e.train_loss_last_mean),
        t_compute_mean: num(e.t_compute_mean),
        drift_mean: num(e.drift_mean),
        update_norm_rel_mean: num(e.update_norm_rel_mean),
        grad_norm_last_mean: num(e.grad_norm_last_mean),
        delta_norm: num(st.delta_norm),
        momentum_norm: num(st.momentum_norm),
        c_server_norm: num(st.c_server_norm),
        comm_mb: num(e.comm_mb),
        SR: num(sh.SR),
        IF: num(sh.IF),
        I_s: num(sh.I_s),
        T_min: num(sh.T_min),
        T_max: num(sh.T_max),
        W_total: num(sh.W_total),
        W_imbalance: num(sh.W_imbalance),
        n_dropped: num(sh.n_dropped),
      })
      const cs = (e.clients as RawEvent[] | undefined) ?? []
      for (const c of cs) {
        clients.push({
          round: num(e.round),
          partition_id: num(c.partition_id),
          node_name: typeof c.node_name === 'string' ? c.node_name : undefined,
          num_examples: num(c.num_examples),
          chunk_fraction: num(c.chunk_fraction),
          local_epochs: num(c.local_epochs),
          w_client: num(c.w_client),
          train_loss_first: num(c.train_loss_first),
          train_loss_last: num(c.train_loss_last),
          t_compute: num(c.t_compute),
          t_serialize: num(c.t_serialize),
          t_local: num(c.t_local),
          created_at: num(c.created_at),
          t_aggr_start: num(c.t_aggr_start),
          t_lifecycle: num(c.t_lifecycle),
          w_drift: num(c.w_drift),
          update_norm_rel: num(c.update_norm_rel),
          grad_norm_last: num(c.grad_norm_last),
        })
      }
    } else if (e.type === 'run_done') {
      runDone = e
    }
  }

  const bestRow = rounds.reduce<RoundRow | null>(
    (acc, r) => (acc === null || r.test_acc > acc.test_acc ? r : acc),
    null,
  )
  const summary: SummaryData = {
    config: runStartedConfig,
    best_acc: num(runDone?.best_acc, bestRow?.test_acc ?? 0),
    best_round: num(runDone?.best_round, bestRow?.round ?? 0),
    rounds_completed: num(runDone?.rounds_completed, rounds.length),
    num_rounds: num(runDone?.num_rounds, num(runStartedConfig['num-server-rounds'], rounds.length)),
    data_heterogeneity: dataHet,
    system_heterogeneity_mean: {
      SR: rounds.length ? rounds.reduce((s, r) => s + r.SR, 0) / rounds.length : 0,
      IF: rounds.length ? rounds.reduce((s, r) => s + r.IF, 0) / rounds.length : 0,
      I_s: rounds.length ? rounds.reduce((s, r) => s + r.I_s, 0) / rounds.length : 0,
      W_total_sum: rounds.reduce((s, r) => s + r.W_total, 0),
    },
    started_at_ts: startedAtTs,
    finished_at_ts: runDone ? num(runDone.ts) || lastRoundTs : null,
    per_class_accuracy: Array.isArray(runDone?.per_class_accuracy)
      ? (runDone.per_class_accuracy as { class_id: number; name: string; accuracy: number }[])
      : [],
  }

  return { rounds, clients, schedule, summary }
}
