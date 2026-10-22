import { useState } from 'react'
import Header from '../Layout/Header'
import LoadingSpinner, { EmptyState } from '../Common/LoadingSpinner'
import { useApi } from '../../hooks/useApi'
import { api } from '../../services/api'
import { useDataset } from '../../hooks/DatasetContext'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
  LineChart, Line, Legend, Cell,
} from 'recharts'

const OURS = '#388bfd'
const OTHER = '#6e7681'

export default function Benchmarks() {
  const { dataset: selected } = useDataset()
  const dsLoading = false
  const [running, setRunning] = useState(false)

  const { data, loading, error, refetch } = useApi(
    () => (selected ? api.getBenchmark(selected) : Promise.resolve(null)),
    [selected],
  )

  const run = async () => {
    setRunning(true)
    try {
      await api.runBenchmarks(selected)
      for (;;) {
        await new Promise((r) => setTimeout(r, 3000))
        const st = await api.getBenchmarkStatus()
        if (st.status !== 'running') break
      }
      await refetch()
    } finally {
      setRunning(false)
    }
  }

  const actions = (
    <button className="btn btn-secondary btn-sm" onClick={run} disabled={running}>
      {running ? 'Running...' : 'Re-run'}
    </button>
  )

  return (
    <>
      <Header
        title="Benchmarks"
        subtitle="Our models against standard baselines on synthetic and public datasets"
        actions={actions}
      />
      <div className="page-body">
        {(dsLoading || loading) && <LoadingSpinner message="Loading benchmark results..." />}
        {error && !loading && (
          <EmptyState
            title="No results for this dataset yet"
            message={error}
            action={
              <button className="btn btn-primary" onClick={run} disabled={running}>
                {running ? 'Running...' : 'Run benchmark'}
              </button>
            }
          />
        )}
        {data && !loading && (
          <>
            <DatasetCard data={data} />
            {Object.entries(data.tasks).map(([key, task]) =>
              task.kind === 'segmentation'
                ? <SegmentationCard key={key} task={task} />
                : <ModelTaskCard key={key} task={task} />
            )}
          </>
        )}
      </div>
    </>
  )
}

function DatasetCard({ data }) {
  return (
    <div className="card mb-24">
      <div className="card-header">
        <div className="card-title">{data.label}</div>
        <span className="badge badge-muted">{data.license}</span>
      </div>
      <p style={{ fontSize: 13, color: 'var(--text-secondary)', margin: '0 0 12px' }}>{data.description}</p>
      <div style={{ display: 'flex', gap: 24, fontSize: 12, color: 'var(--text-muted)', flexWrap: 'wrap' }}>
        <span>Rows: <b style={{ color: 'var(--text-primary)' }}>{data.rows.toLocaleString()}</b></span>
        <span>Columns: <b style={{ color: 'var(--text-primary)' }}>{data.columns}</b></span>
        <span>Source: {data.source}</span>
        <span>Run: {new Date(data.generated_at).toLocaleString()}</span>
      </div>
    </div>
  )
}

function bestOf(models, key, lowerIsBetter = false) {
  const vals = models.map((m) => m[key]).filter((v) => v !== undefined)
  return lowerIsBetter ? Math.min(...vals) : Math.max(...vals)
}

function MetricCell({ value, isBest, fmt }) {
  return (
    <td style={{
      fontFamily: 'var(--font-mono)', fontSize: 12, fontWeight: isBest ? 700 : 400,
      color: isBest ? 'var(--accent-green)' : undefined,
    }}>
      {fmt(value)}
    </td>
  )
}

function formatMetric(key, v) {
  if (key === 'mae') return v.toLocaleString(undefined, { maximumFractionDigits: 0 })
  if (key === 'lift_top_decile') return v.toFixed(2) + '×'
  return v.toFixed(3)
}

function ModelTaskCard({ task }) {
  const isClf = task.kind === 'classification'
  const cols = isClf
    ? [['auc', 'AUC'], ['pr_auc', 'PR-AUC'], ['f1', 'F1'], ['accuracy', 'Accuracy'], ['lift_top_decile', 'Lift @10%']]
    : [['r2_log', 'R² (log)'], ['mae', 'MAE']]
  const metric = isClf ? 'auc' : 'r2_log'
  const chartData = task.models.map((m) => ({
    name: m.name.replace(' (ours)', ''), value: m[metric], ours: m.is_ours,
  }))
  const fi = Object.entries(task.feature_importance ?? {}).map(([k, v]) => ({
    feature: k.replace(/_/g, ' '), value: +(v * 100).toFixed(1),
  }))

  return (
    <div className="card mb-24">
      <div className="card-header">
        <div>
          <div className="card-title">{task.label}</div>
          <div className="card-subtitle">
            Target: {task.target}
            {isClf
              ? ` · positive rate ${(task.positive_rate * 100).toFixed(1)}% · majority-class accuracy ${(task.baseline_accuracy * 100).toFixed(1)}%`
              : ` · mean ${task.target_mean.toLocaleString()} · ${(task.zero_share * 100).toFixed(0)}% zeros`}
            {` · ${task.n_train.toLocaleString()} train / ${task.n_test.toLocaleString()} test`}
          </div>
        </div>
      </div>

      <div className="data-table-wrap mb-16">
        <table className="data-table">
          <thead>
            <tr>
              <th>Model</th>
              {cols.map(([, label]) => <th key={label}>{label}</th>)}
              {isClf && <th>AUC 95% CI</th>}
            </tr>
          </thead>
          <tbody>
            {task.models.map((m) => (
              <tr key={m.name}>
                <td>{m.name} {m.is_ours && <span className="badge badge-blue">ours</span>}</td>
                {cols.map(([key]) => (
                  <MetricCell
                    key={key}
                    value={m[key]}
                    isBest={m[key] === bestOf(task.models, key, key === 'mae')}
                    fmt={(v) => formatMetric(key, v)}
                  />
                ))}
                {isClf && (
                  <td style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)' }}>
                    {m.auc_ci[0].toFixed(3)} – {m.auc_ci[1].toFixed(3)}
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="grid-2">
        <div>
          <div className="card-subtitle mb-16">{isClf ? 'AUC by model' : 'R² (log scale) by model'}</div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={chartData} margin={{ top: 4, right: 8, left: -16, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="name" tick={{ fontSize: 9 }} interval={0} />
              <YAxis tick={{ fontSize: 9 }} domain={isClf ? [0.4, 'auto'] : ['auto', 'auto']} />
              <Tooltip />
              <Bar dataKey="value" radius={[2, 2, 0, 0]}>
                {chartData.map((d, i) => <Cell key={i} fill={d.ours ? OURS : OTHER} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div>
          <div className="card-subtitle mb-16">Top features (our model, % of importance)</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={fi} layout="vertical" margin={{ top: 4, right: 12, left: 70, bottom: 0 }}>
              <XAxis type="number" tick={{ fontSize: 9 }} />
              <YAxis dataKey="feature" type="category" tick={{ fontSize: 9 }} width={90} interval={0} />
              <Tooltip />
              <Bar dataKey="value" fill={OURS} radius={[0, 2, 2, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {task.leakage_check && (
        <div className="alert alert-warning" style={{ marginTop: 16 }}>
          <b>Leakage check:</b> {task.leakage_check.label} reaches AUC {task.leakage_check.auc.toFixed(3)}. {task.leakage_check.note}
        </div>
      )}
      <Notes notes={task.notes} />
    </div>
  )
}

function SegmentationCard({ task }) {
  const bestMethod = (key, lower) => {
    const vals = task.methods.map((m) => m[key])
    return lower ? Math.min(...vals) : Math.max(...vals)
  }
  return (
    <div className="card mb-24">
      <div className="card-header">
        <div>
          <div className="card-title">{task.label}</div>
          <div className="card-subtitle">
            {task.n_samples.toLocaleString()} customers · {task.n_features} features · k = {task.k_ours} · best k by silhouette = {task.best_k_by_silhouette}
          </div>
        </div>
      </div>

      <div className="data-table-wrap mb-16">
        <table className="data-table">
          <thead>
            <tr><th>Method</th><th>Silhouette ↑</th><th>Davies-Bouldin ↓</th><th>Stability (ARI) ↑</th></tr>
          </thead>
          <tbody>
            {task.methods.map((m) => (
              <tr key={m.name}>
                <td>{m.name} {m.is_ours && <span className="badge badge-blue">ours</span>}</td>
                <MetricCell value={m.silhouette} isBest={m.silhouette === bestMethod('silhouette')} fmt={(v) => v.toFixed(3)} />
                <MetricCell value={m.davies_bouldin} isBest={m.davies_bouldin === bestMethod('davies_bouldin', true)} fmt={(v) => v.toFixed(2)} />
                <td style={{ fontFamily: 'var(--font-mono)', fontSize: 12 }}>
                  {m.stability_ari !== undefined ? m.stability_ari.toFixed(3) : '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="card-subtitle mb-16">KMeans quality by number of clusters</div>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={task.k_sweep} margin={{ top: 4, right: 12, left: -16, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis dataKey="k" tick={{ fontSize: 10 }} />
          <YAxis yAxisId="s" tick={{ fontSize: 9 }} />
          <YAxis yAxisId="d" orientation="right" tick={{ fontSize: 9 }} />
          <Tooltip />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Line yAxisId="s" type="monotone" dataKey="silhouette" stroke={OURS} name="Silhouette ↑" />
          <Line yAxisId="d" type="monotone" dataKey="davies_bouldin" stroke="#d29922" name="Davies-Bouldin ↓" />
        </LineChart>
      </ResponsiveContainer>
      <Notes notes={task.notes} />
    </div>
  )
}

function Notes({ notes }) {
  if (!notes?.length) return null
  return (
    <ul style={{ margin: '16px 0 0', paddingLeft: 18, fontSize: 12, color: 'var(--text-muted)' }}>
      {notes.map((n, i) => <li key={i}>{n}</li>)}
    </ul>
  )
}
