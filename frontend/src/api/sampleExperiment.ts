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
