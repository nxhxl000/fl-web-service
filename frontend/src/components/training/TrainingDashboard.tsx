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

export function TrainingDashboard() {
  const [data, setData] = useState<SampleExperiment | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    loadSampleExperiment()
      .then(setData)
      .catch((err) => setError(String(err)))
  }, [])

  if (error) {
    return <p className="text-sm text-red-600">Failed to load sample experiment: {error}</p>
  }
  if (!data) {
    return <p className="text-sm text-neutral-500">Loading sample experiment…</p>
  }

  return (
    <div className="space-y-6">
      <RunHeader data={data} />
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        <ChartCard title="Test accuracy & loss">
          <AccuracyChart rounds={data.rounds} />
        </ChartCard>
        <ChartCard title="Macro F1 & loss">
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

function RunHeader({ data }: { data: SampleExperiment }) {
  const cfg = data.summary.config
  const sr1 = data.rounds[0]?.SR ?? 0
  const srMean = data.summary.system_heterogeneity_mean.SR
  return (
    <div className="rounded border border-neutral-200 bg-white p-5">
      <div className="flex flex-wrap items-baseline justify-between gap-4">
        <div>
          <h3 className="text-sm font-semibold text-neutral-900">Sample run (CIFAR-100, IID)</h3>
          <p className="mt-1 text-xs text-neutral-500">
            Static placeholder from <code>exp/iid/</code> — will be replaced by live data once the
            orchestrator is wired up.
          </p>
        </div>
        <div className="flex flex-wrap gap-3 text-xs">
          <Pill label="Model" value={String(cfg.model)} />
          <Pill label="Aggregation" value={String(cfg.aggregation)} />
          <Pill label="Rounds" value={`${data.summary.rounds_completed} / ${data.summary.num_rounds}`} />
          <Pill label="Best acc" value={`${(data.summary.best_acc * 100).toFixed(1)}% @ r${data.summary.best_round}`} />
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

function TrainLossChart({ clients, rounds }: { clients: ClientRow[]; rounds: RoundRow[] }) {
  const scatter = clients.map((c) => ({
    round: c.round,
    train_loss_last: +c.train_loss_last.toFixed(4),
  }))
  const mean = rounds.map((r) => ({
    round: r.round,
    mean: +r.train_loss_last_mean.toFixed(4),
  }))
  return (
    <ResponsiveContainer width="100%" height={300}>
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
          label={{
            value: 'Train loss',
            angle: -90,
            position: 'insideLeft',
            offset: 0,
            style: { textAnchor: 'middle', fontSize: 12 },
          }}
        />
        <ZAxis range={[40, 40]} />
        <Tooltip cursor={{ strokeDasharray: '3 3' }} />
        <Legend verticalAlign="top" align="right" wrapperStyle={{ fontSize: 12, paddingBottom: 8 }} />
        <Scatter name="Per-client" data={scatter} fill="#3b82f6" fillOpacity={0.45} />
        <Line
          type="monotone"
          dataKey="mean"
          data={mean}
          stroke="#dc2626"
          strokeWidth={2}
          dot={{ r: 4, fill: '#dc2626' }}
          name="Mean"
          isAnimationActive={false}
        />
      </ComposedChart>
    </ResponsiveContainer>
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
  return (
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
      return {
        pid: `pid ${pid}`,
        'Round 1 t_compute (s)': r1 ? +r1.t_compute.toFixed(1) : 0,
        'Rounds 2+ avg t_compute (s)': +restAvg.toFixed(1),
        chunk: +restChunk.toFixed(2),
      }
    })
  }, [clients])

  const r1SR = rounds[0]?.SR ?? 0
  const restSR =
    rounds.length > 1
      ? rounds.slice(1).reduce((acc, r) => acc + r.SR, 0) / (rounds.length - 1)
      : 0

  return (
    <div>
      <p className="mb-3 text-xs text-neutral-600">
        Bars show per-client compute time before mitigation (round 1) and after (rounds 2+ avg).
        Chunk-fraction column reflects what each client was assigned by the round-1 schedule.
        Speedup ratio (T_max / T_min): <span className="font-medium">{r1SR.toFixed(2)}</span> →{' '}
        <span className="font-medium">{restSR.toFixed(2)}</span>.
      </p>
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
              <th className="px-3 py-2 font-medium">pid</th>
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
