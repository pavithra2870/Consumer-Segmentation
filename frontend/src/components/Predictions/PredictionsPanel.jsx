import Header from '../Layout/Header'
import LoadingSpinner, { EmptyState } from '../Common/LoadingSpinner'
import { useApi } from '../../hooks/useApi'
import { api } from '../../services/api'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from 'recharts'

const TASK_UI = {
  churn: { title: 'Churn Prediction', color: '#f85149', badge: 'Classifier', metric: 'auc', metricLabel: 'AUC', distTitle: 'Churn Score Distribution' },
  ltv: { title: 'Value Prediction', color: '#3fb950', badge: 'Regressor', metric: 'r2_score', metricLabel: 'R² (log scale)', distTitle: 'Predicted Value Distribution' },
  conversion: { title: 'Conversion Scoring', color: '#388bfd', badge: 'Classifier', metric: 'auc', metricLabel: 'AUC', distTitle: 'Conversion Score Distribution' },
}
const TASK_ORDER = ['churn', 'ltv', 'conversion']

export default function PredictionsPanel() {
  const { data, loading, error } = useApi(api.getPredictions)
  const tasks = data ? TASK_ORDER.filter((t) => data[t]) : []
  const cols = { display: 'grid', gridTemplateColumns: `repeat(${Math.max(tasks.length, 1)}, 1fr)`, gap: 16 }
  const meta = data?.meta

  return (
    <>
      <Header
        title="Predictions Panel"
        subtitle={tasks.map((t) => TASK_UI[t].title).join(' · ') || 'Model results'}
      />
      <div className="page-body">
        {loading && <LoadingSpinner message="Loading predictions..." />}
        {error && <div className="alert alert-error">{error}</div>}
        {data && (
          <>
            <div style={cols} className="mb-24">
              {tasks.map((t) => (
                <ModelSummaryCard key={t} ui={TASK_UI[t]} model={data[t]} />
              ))}
            </div>

            <div style={cols} className="mb-24">
              {tasks.map((t) => (
                <DistributionChart
                  key={t}
                  title={TASK_UI[t].distTitle}
                  data={t === 'ltv' ? data[t].distribution?.slice(0, 15) : data[t].distribution}
                  color={TASK_UI[t].color}
                />
              ))}
            </div>

            <div style={cols} className="mb-24">
              {tasks.map((t) => (
                <FeatureImportanceChart
                  key={t}
                  title={`${TASK_UI[t].title.replace(' Prediction', '').replace(' Scoring', '')} — Feature Importance`}
                  importance={data[t].feature_importance}
                  color={TASK_UI[t].color}
                />
              ))}
            </div>

            <div className="card">
              <div className="card-header">
                <div className="card-title">Sample Predictions</div>
                <div className="card-subtitle">{data.sample_predictions?.length ?? 0} sampled customers with model outputs</div>
              </div>
              <div className="data-table-wrap">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Customer ID</th>
                      <th>Segment</th>
                      {data.churn && <th>Churn Prob</th>}
                      {data.ltv && <th>{meta?.value_label ?? 'Value'} Estimate</th>}
                      {data.conversion && <th>Conversion Score</th>}
                    </tr>
                  </thead>
                  <tbody>
                    {data.sample_predictions?.map((p, i) => (
                      <tr key={i}>
                        <td className="mono">{p.customer_id}</td>
                        <td><span className="badge badge-muted">Seg {p.segment}</span></td>
                        {data.churn && (
                          <td>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                              <div className="progress-bar-wrap" style={{ width: 60, display: 'inline-block' }}>
                                <div
                                  className="progress-bar-fill"
                                  style={{
                                    width: `${p.churn_probability * 100}%`,
                                    background: p.churn_probability > 0.5 ? '#f85149' : p.churn_probability > 0.3 ? '#d29922' : '#3fb950'
                                  }}
                                />
                              </div>
                              <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)' }}>
                                {(p.churn_probability * 100).toFixed(1)}%
                              </span>
                            </div>
                          </td>
                        )}
                        {data.ltv && (
                          <td style={{ fontFamily: 'var(--font-mono)', fontSize: 12 }}>
                            {meta?.value_prefix ?? '$'}{p.ltv_estimate?.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                          </td>
                        )}
                        {data.conversion && (
                          <td>
                            <span className={`badge ${p.conversion_score > 0.5 ? 'badge-green' : p.conversion_score > 0.3 ? 'badge-yellow' : 'badge-muted'}`}>
                              {(p.conversion_score * 100).toFixed(1)}%
                            </span>
                          </td>
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}
      </div>
    </>
  )
}

function ModelSummaryCard({ ui, model }) {
  const value = model?.[ui.metric]
  return (
    <div className="card">
      <div className="flex items-center justify-between mb-16">
        <div className="card-title">{ui.title}</div>
        <span className="badge badge-muted">{ui.badge}</span>
      </div>
      <div style={{ fontSize: 32, fontWeight: 700, color: ui.color, marginBottom: 4, fontVariantNumeric: 'tabular-nums' }}>
        {value !== undefined ? value.toFixed(3) : '--'}
      </div>
      <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
        {ui.metricLabel} · {model?.model ?? 'N/A'}
      </div>
      {model?.baseline_accuracy !== undefined && (
        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
          Accuracy {(model.accuracy * 100).toFixed(1)}% vs {(model.baseline_accuracy * 100).toFixed(1)}% baseline · PR-AUC {model.avg_precision?.toFixed(3)}
        </div>
      )}
    </div>
  )
}

function DistributionChart({ title, data, color }) {
  return (
    <div className="card">
      <div className="card-header">
        <div className="card-title">{title}</div>
      </div>
      {data?.length > 0 ? (
        <ResponsiveContainer width="100%" height={180}>
          <BarChart data={data} margin={{ top: 4, right: 4, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis dataKey="range" tick={{ fontSize: 8 }} angle={-30} textAnchor="end" height={32} />
            <YAxis tick={{ fontSize: 9 }} width={44} />
            <Tooltip />
            <Bar dataKey="count" fill={color} radius={[2,2,0,0]} />
          </BarChart>
        </ResponsiveContainer>
      ) : <EmptyState title="No data" />}
    </div>
  )
}

function FeatureImportanceChart({ title, importance, color }) {
  const data = importance
    ? Object.entries(importance).map(([k, v]) => ({ feature: k.replace(/_/g,' '), value: +(v * 100).toFixed(3) }))
    : []

  return (
    <div className="card">
      <div className="card-header">
        <div className="card-title">{title}</div>
      </div>
      {data.length > 0 ? (
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={data} layout="vertical" margin={{ top: 4, right: 12, left: 80, bottom: 0 }}>
            <XAxis type="number" tick={{ fontSize: 9 }} />
            <YAxis dataKey="feature" type="category" tick={{ fontSize: 9 }} width={80} interval={0} />
            <Tooltip />
            <Bar dataKey="value" fill={color} radius={[0,2,2,0]} />
          </BarChart>
        </ResponsiveContainer>
      ) : <EmptyState title="No data" />}
    </div>
  )
}
