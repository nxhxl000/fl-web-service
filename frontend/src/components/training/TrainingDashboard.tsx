import { useEffect, useMemo, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  LineChart,
  ReferenceArea,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from 'recharts'
import {
  loadSampleExperiment,
  type ClientRow,
  type RoundRow,
  type SampleExperiment,
} from '../../api/sampleExperiment'

type Props = {
  data?: SampleExperiment | null
  loading?: boolean
  error?: string | null
}

export function TrainingDashboard({ data: externalData, loading, error: externalError }: Props = {}) {
  const [internalData, setInternalData] = useState<SampleExperiment | null>(null)
  const [internalError, setInternalError] = useState<string | null>(null)

  // Controlled mode: parent passes `data`/`loading`/`error`. Uncontrolled: load sample.
  const controlled =
    externalData !== undefined || loading !== undefined || externalError !== undefined

  useEffect(() => {
    if (controlled) return
    loadSampleExperiment()
      .then(setInternalData)
      .catch((err) => setInternalError(String(err)))
  }, [controlled])

  const data = controlled ? externalData ?? null : internalData
  const error = controlled ? externalError ?? null : internalError

  if (error) {
    return <p className="text-sm text-red-600">Failed to load run data: {error}</p>
  }
  if (!data) {
    return <p className="text-sm text-neutral-500">{loading ? 'Loading run data…' : 'No data yet.'}</p>
  }
  if (data.rounds.length === 0) {
    return <p className="text-sm text-neutral-500">Waiting for the first round to complete…</p>
  }

  return (
    <div className="space-y-6">
      <RunHeader data={data} />
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        <ChartCard title="Test accuracy">
          <AccuracyChart rounds={data.rounds} />
        </ChartCard>
        <ChartCard title="Macro F1">
          <F1Chart rounds={data.rounds} />
        </ChartCard>
        <ChartCard title="Train loss per round (per-client points + mean)">
          <TrainLossChart clients={data.clients} rounds={data.rounds} />
        </ChartCard>
        <ChartCard title="Data heterogeneity space">
          <HeterogeneityScatter
            mpjs={data.summary.data_heterogeneity.MPJS}
            gini={data.summary.data_heterogeneity.Gini_quantity}
            partition={String(data.summary.config['partition-name'] ?? 'unknown')}
          />
        </ChartCard>
      </div>
      <ChartCard title="Straggler mitigation: round 1 vs rounds 2+">
        <StragglerComparison clients={data.clients} rounds={data.rounds} />
      </ChartCard>
      {(data.summary.per_class_accuracy?.length ?? 0) > 0 && (
        <ChartCard title="Per-class accuracy (best model on test set)">
          <PerClassAccuracyHeatmap rows={data.summary.per_class_accuracy} />
        </ChartCard>
      )}
    </div>
  )
}

function ChartCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded border border-neutral-200 bg-white p-5">
      <h3 className="text-sm font-semibold text-neutral-900">{title}</h3>
      <div className="mt-4">{children}</div>
    </div>
  )
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(1)}s`
  const m = Math.floor(seconds / 60)
  const s = Math.round(seconds - m * 60)
  if (m < 60) return `${m}m ${s}s`
  const h = Math.floor(m / 60)
  return `${h}h ${m - h * 60}m ${s}s`
}

function RunHeader({ data }: { data: SampleExperiment }) {
  const cfg = data.summary.config
  const sr1 = data.rounds[0]?.SR ?? 0
  const srMean = data.summary.system_heterogeneity_mean.SR
  const partition = String(cfg['partition-name'] ?? 'unknown')
  const dataset = partition.split('__')[0] || 'unknown'
  const startTs = data.summary.started_at_ts
  const endTs = data.summary.finished_at_ts ?? Date.now() / 1000
  const elapsed = startTs ? Math.max(0, endTs - startTs) : 0
  const completed = data.summary.rounds_completed
  const avgRound = completed > 0 ? elapsed / completed : 0
  return (
    <div className="rounded border border-neutral-200 bg-white p-5">
      <div className="flex flex-wrap items-baseline justify-between gap-4">
        <div>
          <h3 className="text-sm font-semibold text-neutral-900">
            {dataset} · {String(cfg.model ?? '—')} / {String(cfg.aggregation ?? '—')}
          </h3>
          <p className="mt-1 text-xs text-neutral-500 font-mono">{partition}</p>
        </div>
        <div className="flex flex-wrap gap-3 text-xs">
          <Pill label="Rounds" value={`${data.summary.rounds_completed} / ${data.summary.num_rounds}`} />
          <Pill label="Best acc" value={`${(data.summary.best_acc * 100).toFixed(1)}% @ r${data.summary.best_round}`} />
          {startTs != null && (
            <>
              <Pill label="Elapsed" value={formatDuration(elapsed)} />
              <Pill label="Avg round" value={completed > 0 ? formatDuration(avgRound) : '—'} />
            </>
          )}
          <Pill label="Speedup ratio" value={`${sr1.toFixed(2)} → ${srMean.toFixed(2)}`} />
        </div>
      </div>
    </div>
  )
}

function Pill({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-full border border-neutral-200 bg-neutral-50 px-3 py-1">
      <span className="text-neutral-500">{label}:</span>{' '}
      <span className="font-medium text-neutral-900">{value}</span>
    </div>
  )
}

function AccuracyChart({ rounds }: { rounds: RoundRow[] }) {
  const data = rounds.map((r) => ({
    round: r.round,
    acc: +(r.test_acc * 100).toFixed(2),
  }))
  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={data} margin={{ top: 10, right: 20, left: 20, bottom: 28 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e5e5" />
        <XAxis
          dataKey="round"
          tick={{ fontSize: 12 }}
          label={{ value: 'Round', position: 'insideBottom', offset: -10, fontSize: 12 }}
        />
        <YAxis
          tick={{ fontSize: 12 }}
          domain={[0, 'auto']}
          label={{
            value: 'Test accuracy (%)',
            angle: -90,
            position: 'insideLeft',
            offset: 0,
            style: { textAnchor: 'middle', fontSize: 12 },
          }}
        />
        <Tooltip />
        <Line type="monotone" dataKey="acc" name="Test accuracy (%)" stroke="#2563eb" strokeWidth={2} dot={{ r: 3 }} />
      </LineChart>
    </ResponsiveContainer>
  )
}

function F1Chart({ rounds }: { rounds: RoundRow[] }) {
  const data = rounds.map((r) => ({
    round: r.round,
    f1: +(r.test_f1 * 100).toFixed(2),
  }))
  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={data} margin={{ top: 10, right: 20, left: 20, bottom: 28 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e5e5" />
        <XAxis
          dataKey="round"
          tick={{ fontSize: 12 }}
          label={{ value: 'Round', position: 'insideBottom', offset: -10, fontSize: 12 }}
        />
        <YAxis
          tick={{ fontSize: 12 }}
          domain={[0, 'auto']}
          label={{
            value: 'Macro F1 (%)',
            angle: -90,
            position: 'insideLeft',
            offset: 0,
            style: { textAnchor: 'middle', fontSize: 12 },
          }}
        />
        <Tooltip />
        <Line type="monotone" dataKey="f1" name="Macro F1 (%)" stroke="#16a34a" strokeWidth={2} dot={{ r: 3 }} />
      </LineChart>
    </ResponsiveContainer>
  )
}

type HoverInfo = {
  x: number
  y: number
  kind: 'client' | 'mean'
  round: number
  label?: string
  loss: number
}

function TrainLossChart({ clients, rounds }: { clients: ClientRow[]; rounds: RoundRow[] }) {
  const scatter = clients.map((c) => ({
    round: c.round,
    pid: c.partition_id,
    label: c.node_name || `pid ${c.partition_id}`,
    train_loss_last: +c.train_loss_last.toFixed(4),
  }))
  const mean = rounds.map((r) => ({
    round: r.round,
    mean: +r.train_loss_last_mean.toFixed(4),
  }))

  const [hover, setHover] = useState<HoverInfo | null>(null)

  // Custom shape: each scatter point is its own SVG circle with mouse handlers,
  // bypassing Recharts' axis-shared Tooltip. ComposedChart's tooltip would otherwise
  // collapse all points at the same X to one entry.
  const ClientDot = (props: {
    cx?: number
    cy?: number
    payload?: { round: number; pid: number; label: string; train_loss_last: number }
  }) => {
    const { cx, cy, payload } = props
    if (cx === undefined || cy === undefined || !payload) return null
    return (
      <circle
        cx={cx}
        cy={cy}
        r={4}
        fill="#3b82f6"
        fillOpacity={0.55}
        stroke="#1e3a8a"
        strokeWidth={0.5}
        style={{ cursor: 'pointer' }}
        onMouseEnter={() =>
          setHover({
            x: cx,
            y: cy,
            kind: 'client',
            round: payload.round,
            label: payload.label,
            loss: payload.train_loss_last,
          })
        }
        onMouseLeave={() => setHover(null)}
      />
    )
  }

  const MeanDot = (props: {
    cx?: number
    cy?: number
    payload?: { round: number; mean: number }
  }) => {
    const { cx, cy, payload } = props
    if (cx === undefined || cy === undefined || !payload) return null
    return (
      <circle
        cx={cx}
        cy={cy}
        r={5}
        fill="#dc2626"
        style={{ cursor: 'pointer' }}
        onMouseEnter={() =>
          setHover({ x: cx, y: cy, kind: 'mean', round: payload.round, loss: payload.mean })
        }
        onMouseLeave={() => setHover(null)}
      />
    )
  }

  return (
    <div className="relative">
      <ResponsiveContainer width="100%" height={500}>
        <ComposedChart margin={{ top: 10, right: 20, left: 20, bottom: 28 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e5e5" />
          <XAxis
            type="number"
            dataKey="round"
            domain={[
              Math.min(...rounds.map((r) => r.round)) - 0.5,
              Math.max(...rounds.map((r) => r.round)) + 0.5,
            ]}
            tickCount={rounds.length}
            tick={{ fontSize: 12 }}
            allowDecimals={false}
            label={{ value: 'Round', position: 'insideBottom', offset: -10, fontSize: 12 }}
          />
          <YAxis
            type="number"
            dataKey="train_loss_last"
            tick={{ fontSize: 12 }}
            domain={['auto', 'auto']}
            tickCount={10}
            label={{
              value: 'Train loss',
              angle: -90,
              position: 'insideLeft',
              offset: 0,
              style: { textAnchor: 'middle', fontSize: 12 },
            }}
          />
          <ZAxis range={[40, 40]} />
          <Legend verticalAlign="top" align="right" wrapperStyle={{ fontSize: 12, paddingBottom: 8 }} />
          <Scatter name="Per-client" data={scatter} shape={ClientDot} />
          <Line
            type="monotone"
            dataKey="mean"
            data={mean}
            stroke="#dc2626"
            strokeWidth={2}
            dot={MeanDot}
            activeDot={false}
            name="Mean"
            isAnimationActive={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
      {hover && (
        <div
          className="pointer-events-none absolute z-10 rounded border border-neutral-200 bg-white p-2 text-xs shadow-sm"
          style={{
            left: hover.x + 10,
            top: hover.y + 10,
          }}
        >
          {hover.kind === 'client' ? (
            <>
              <div className="font-medium text-neutral-900">
                Round {hover.round} · {hover.label}
              </div>
              <div className="mt-1 text-blue-700">
                train loss: <span className="font-mono">{hover.loss.toFixed(4)}</span>
              </div>
            </>
          ) : (
            <>
              <div className="font-medium text-neutral-900">Round {hover.round} · mean</div>
              <div className="mt-1 text-red-600">
                <span className="font-mono">{hover.loss.toFixed(4)}</span>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  )
}

function HeterogeneityScatter({
  mpjs,
  gini,
  partition,
}: {
  mpjs: number
  gini: number
  partition: string
}) {
  const point = [{ mpjs, gini, name: partition }]
  const MPJS_SPLIT = 0.4
  const GINI_SPLIT = 0.2

  const zones = [
    {
      id: 'iid',
      label: 'IID-like',
      strategies: ['FedAvg'],
      active: mpjs < MPJS_SPLIT && gini < GINI_SPLIT,
      labelColor: 'text-green-700',
    },
    {
      id: 'qty',
      label: 'Quantity skew',
      strategies: ['FedAvg', 'FedProx'],
      active: mpjs < MPJS_SPLIT && gini >= GINI_SPLIT,
      labelColor: 'text-orange-700',
    },
    {
      id: 'lbl',
      label: 'Label skew',
      strategies: ['FedAvgM', 'FedNovaM'],
      active: mpjs >= MPJS_SPLIT && gini < GINI_SPLIT,
      labelColor: 'text-blue-700',
    },
    {
      id: 'combo',
      label: 'Combined skew',
      strategies: ['FedNovaM'],
      active: mpjs >= MPJS_SPLIT && gini >= GINI_SPLIT,
      labelColor: 'text-red-700',
    },
  ]

  return (
    <>
    <ResponsiveContainer width="100%" height={300}>
      <ScatterChart margin={{ top: 10, right: 20, left: 20, bottom: 28 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e5e5" />
        <XAxis
          type="number"
          dataKey="mpjs"
          name="MPJS"
          domain={[0, 1]}
          tick={{ fontSize: 12 }}
          label={{
            value: 'MPJS — label skew',
            position: 'insideBottom',
            offset: -10,
            fontSize: 12,
          }}
        />
        <YAxis
          type="number"
          dataKey="gini"
          name="Gini"
          domain={[0, 1]}
          tick={{ fontSize: 12 }}
          label={{
            value: 'Gini — quantity skew',
            angle: -90,
            position: 'insideLeft',
            offset: 0,
            style: { textAnchor: 'middle', fontSize: 12 },
          }}
        />
        <ReferenceArea
          x1={0}
          x2={MPJS_SPLIT}
          y1={0}
          y2={GINI_SPLIT}
          fill="#86efac"
          fillOpacity={0.18}
          label={{
            value: 'IID-like',
            position: 'center',
            fontSize: 12,
            fill: '#15803d',
          }}
        />
        <ReferenceArea
          x1={MPJS_SPLIT}
          x2={1}
          y1={0}
          y2={GINI_SPLIT}
          fill="#93c5fd"
          fillOpacity={0.18}
          label={{
            value: 'Label skew',
            position: 'center',
            fontSize: 12,
            fill: '#1d4ed8',
          }}
        />
        <ReferenceArea
          x1={0}
          x2={MPJS_SPLIT}
          y1={GINI_SPLIT}
          y2={1}
          fill="#fdba74"
          fillOpacity={0.18}
          label={{
            value: 'Quantity skew',
            position: 'center',
            fontSize: 12,
            fill: '#c2410c',
          }}
        />
        <ReferenceArea
          x1={MPJS_SPLIT}
          x2={1}
          y1={GINI_SPLIT}
          y2={1}
          fill="#fca5a5"
          fillOpacity={0.18}
          label={{
            value: 'Combined skew',
            position: 'center',
            fontSize: 12,
            fill: '#b91c1c',
          }}
        />
        <Tooltip cursor={{ strokeDasharray: '3 3' }} />
        <ZAxis range={[720, 720]} />
        <Scatter
          data={point}
          fill="#db2777"
          stroke="#ffffff"
          strokeWidth={2}
          shape="circle"
        />
      </ScatterChart>
    </ResponsiveContainer>
    <div className="mt-4 space-y-2">
      <p className="text-xs font-medium text-neutral-700">Recommended strategies by zone</p>
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
        {zones.map((z) => (
          <div
            key={z.id}
            className={
              z.active
                ? 'rounded border border-neutral-400 bg-neutral-50 px-3 py-2 shadow-sm'
                : 'rounded border border-neutral-200 px-3 py-2'
            }
          >
            <div className="flex items-center justify-between gap-3 text-xs">
              <span className={`font-medium ${z.labelColor}`}>
                {z.label}
                {z.active && <span className="ml-2 text-neutral-500">(current partition)</span>}
              </span>
              <span className="font-mono text-neutral-900">{z.strategies.join(', ')}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
    </>
  )
}

function StragglerComparison({ clients, rounds }: { clients: ClientRow[]; rounds: RoundRow[] }) {
  const data = useMemo(() => {
    const pids = Array.from(new Set(clients.map((c) => c.partition_id))).sort((a, b) => a - b)
    return pids.map((pid) => {
      const r1 = clients.find((c) => c.partition_id === pid && c.round === 1)
      const rest = clients.filter((c) => c.partition_id === pid && c.round > 1)
      const restAvg =
        rest.length === 0 ? 0 : rest.reduce((acc, c) => acc + c.t_compute, 0) / rest.length
      const restChunk =
        rest.length === 0
          ? 1
          : rest.reduce((acc, c) => acc + c.chunk_fraction, 0) / rest.length
      const label = (r1 ?? rest[0])?.node_name || `pid ${pid}`
      return {
        pid: label,
        'Round 1 t_compute (s)': r1 ? +r1.t_compute.toFixed(1) : 0,
        'Rounds 2+ avg t_compute (s)': +restAvg.toFixed(1),
        chunk: +restChunk.toFixed(2),
      }
    })
  }, [clients])

  const restSlice = rounds.slice(1)
  const hasRest = restSlice.length > 0
  const meanRest = (key: 'SR' | 'IF' | 'I_s') =>
    hasRest ? restSlice.reduce((acc, r) => acc + r[key], 0) / restSlice.length : 0
  const r1 = rounds[0]
  const fmt = (v1: number, v2: number, digits = 2) =>
    hasRest ? `${v1.toFixed(digits)} → ${v2.toFixed(digits)}` : v1.toFixed(digits)

  return (
    <div>
      <p className="mb-3 text-xs text-neutral-600">
        Bars show per-client compute time before mitigation (round 1) and after (rounds 2+ avg).
        Chunk-fraction column reflects what each client was assigned by the round-1 schedule.
      </p>
      <div className="mb-4 flex flex-wrap gap-2 text-xs">
        <Pill
          label="Speedup ratio (T_max / T_min)"
          value={fmt(r1?.SR ?? 0, meanRest('SR'))}
        />
        <Pill label="Idle fraction" value={fmt(r1?.IF ?? 0, meanRest('IF'))} />
        <Pill
          label="Heterogeneity index Iₛ"
          value={fmt(r1?.I_s ?? 0, meanRest('I_s'), 3)}
        />
      </div>
      <ResponsiveContainer width="100%" height={Math.max(280, data.length * 40)}>
        <BarChart
          data={data}
          layout="vertical"
          barGap={4}
          margin={{ top: 10, right: 20, left: 30, bottom: 28 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e5e5" />
          <XAxis
            type="number"
            tick={{ fontSize: 12 }}
            label={{
              value: 't_compute (seconds)',
              position: 'insideBottom',
              offset: -10,
              fontSize: 12,
            }}
          />
          <YAxis type="category" dataKey="pid" tick={{ fontSize: 12 }} width={70} />
          <Tooltip />
          <Legend verticalAlign="top" align="right" wrapperStyle={{ fontSize: 12, paddingBottom: 8 }} />
          <Bar dataKey="Round 1 t_compute (s)" fill="#9ca3af" />
          <Bar dataKey="Rounds 2+ avg t_compute (s)" fill="#2563eb" />
        </BarChart>
      </ResponsiveContainer>
      <div className="mt-3 overflow-x-auto rounded border border-neutral-200">
        <table className="min-w-full text-left text-xs">
          <thead className="bg-neutral-50 text-neutral-500">
            <tr>
              <th className="px-3 py-2 font-medium">Node</th>
              <th className="px-3 py-2 font-medium">Round 1 t_compute</th>
              <th className="px-3 py-2 font-medium">Rounds 2+ avg</th>
              <th className="px-3 py-2 font-medium">Assigned chunk</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-neutral-200">
            {data.map((d) => (
              <tr key={d.pid}>
                <td className="px-3 py-2 text-neutral-900">{d.pid}</td>
                <td className="px-3 py-2 text-neutral-700">{d['Round 1 t_compute (s)']} s</td>
                <td className="px-3 py-2 text-neutral-700">{d['Rounds 2+ avg t_compute (s)']} s</td>
                <td className="px-3 py-2 text-neutral-700">{d.chunk}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function PerClassAccuracyHeatmap({
  rows,
}: {
  rows: { class_id: number; name: string; accuracy: number }[]
}) {
  // Bucket accuracy → red/orange/yellow/green tail. Pure CSS, no extra deps.
  const bg = (acc: number): string => {
    // Stops: 0 → #fee2e2, 0.25 → #fed7aa, 0.5 → #fef08a, 0.75 → #bbf7d0, 1.0 → #16a34a
    if (acc < 0.25) return '#fee2e2'
    if (acc < 0.5) return '#fed7aa'
    if (acc < 0.75) return '#fef08a'
    if (acc < 0.9) return '#bbf7d0'
    return '#86efac'
  }
  const fg = (acc: number): string => (acc >= 0.75 ? '#14532d' : '#7f1d1d')

  return (
    <div>
      <p className="mb-3 text-xs text-neutral-600">
        Accuracy of the best-checkpoint model on each class of the centralized
        test set. Cells are colored from red (low) through yellow to green (high).
      </p>
      <div
        className="grid gap-1"
        style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(110px, 1fr))' }}
      >
        {rows.map((r) => (
          <div
            key={r.class_id}
            className="rounded px-2 py-1.5 text-xs"
            style={{ backgroundColor: bg(r.accuracy), color: fg(r.accuracy) }}
            title={`${r.name}: ${(r.accuracy * 100).toFixed(1)}%`}
          >
            <div className="truncate font-medium">{r.name}</div>
            <div className="font-mono">{(r.accuracy * 100).toFixed(1)}%</div>
          </div>
        ))}
      </div>
    </div>
  )
}
