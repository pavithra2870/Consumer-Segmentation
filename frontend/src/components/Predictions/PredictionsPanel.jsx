import Header from '../Layout/Header'
import LoadingSpinner, { EmptyState } from '../Common/LoadingSpinner'
import { useApi } from '../../hooks/useApi'
import { api } from '../../services/api'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell,
} from 'recharts'

export default function PredictionsPanel() {
  const { data, loading, error } = useApi(api.getPredictions)

  return (
    <>
      <Header
        title="Predictions Panel"
        subtitle="Churn, LTV, and conversion model results"
      />
      <div className="page-body">
        {loading && <LoadingSpinner message="Loading predictions..." />}
        {error && <div className="alert alert-error">{error}</div>}
        {data && (
          <>
            {/* Model summary cards */}
            <div className="grid-3 mb-24">
              <ModelSummaryCard
                title="Churn Prediction"
                model={data.churn}
                metric="accuracy"
                color="#f85149"
                badge="Classifier"
              />
              <ModelSummaryCard
                title="LTV Prediction"
                model={data.ltv}
                metric="r2_score"
                color="#3fb950"
                badge="Regressor"
                metricLabel="R²"
              />
              <ModelSummaryCard
                title="Conversion Scoring"
                model={data.conversion}
                metric="accuracy"
                color="#388bfd"
                badge="Classifier"
              />
            </div>

            {/* Distributions */}
            <div className="grid-3 mb-24">
              <DistributionChart
                title="Churn Score Distribution"
                data={data.churn?.distribution}
                color="#f85149"
                xlabel="Churn Probability"
              />
              <DistributionChart
                title="LTV Distribution"
                data={data.ltv?.distribution?.slice(0, 15)}
                color="#3fb950"
                xlabel="LTV ($)"
              />
              <DistributionChart
                title="Conversion Score Distribution"
                data={data.conversion?.distribution}
                color="#388bfd"
                xlabel="Conversion Probability"
              />
            </div>

            {/* Feature importance */}
            <div className="grid-3 mb-24">
              <FeatureImportanceChart title="Churn — Feature Importance" importance={data.churn?.feature_importance} color="#f85149" />
              <FeatureImportanceChart title="LTV — Feature Importance" importance={data.ltv?.feature_importance} color="#3fb950" />
              <FeatureImportanceChart title="Conversion — Feature Importance" importance={data.conversion?.feature_importance} color="#388bfd" />
            </div>

            {/* Sample predictions table */}
            <div className="card">
              <div className="card-header">
                <div className="card-title">Sample Predictions</div>
                <div className="card-subtitle">50 sampled customers with model outputs</div>
              </div>
              <div className="data-table-wrap">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Customer ID</th>
                      <th>Segment</th>
                      <th>Churn Prob</th>
                      <th>LTV Estimate</th>
                      <th>Conversion Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.sample_predictions?.map((p, i) => (
                      <tr key={i}>
                        <td className="mono">{p.customer_id}</td>
                        <td><span className="badge badge-muted">Seg {p.segment}</span></td>
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
                        <td style={{ fontFamily: 'var(--font-mono)', fontSize: 12 }}>
                          ${p.ltv_estimate?.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                        </td>
                        <td>
                          <span className={`badge ${p.conversion_score > 0.5 ? 'badge-green' : p.conversion_score > 0.3 ? 'badge-yellow' : 'badge-muted'}`}>
                            {(p.conversion_score * 100).toFixed(1)}%
                          </span>
                        </td>
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

function ModelSummaryCard({ title, model, metric, color, badge, metricLabel }) {
  const value = model?.[metric]
  return (
    <div className="card">
      <div className="flex items-center justify-between mb-16">
        <div className="card-title">{title}</div>
        <span className="badge badge-muted">{badge}</span>
      </div>
      <div style={{ fontSize: 32, fontWeight: 700, color, marginBottom: 4, fontVariantNumeric: 'tabular-nums' }}>
        {value !== undefined ? (value * 100).toFixed(1) + '%' : '--'}
      </div>
      <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
        {metricLabel ?? 'Accuracy'} · {model?.model ?? 'N/A'}
      </div>
    </div>
  )
}

function DistributionChart({ title, data, color, xlabel }) {
  return (
    <div className="card">
      <div className="card-header">
        <div className="card-title">{title}</div>
      </div>
      {data?.length > 0 ? (
        <ResponsiveContainer width="100%" height={180}>
          <BarChart data={data} margin={{ top: 4, right: 4, left: -24, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis dataKey="range" tick={{ fontSize: 8 }} angle={-30} textAnchor="end" height={32} />
            <YAxis tick={{ fontSize: 9 }} />
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
        <ResponsiveContainer width="100%" height={180}>
          <BarChart data={data} layout="vertical" margin={{ top: 4, right: 12, left: 80, bottom: 0 }}>
            <XAxis type="number" tick={{ fontSize: 9 }} />
            <YAxis dataKey="feature" type="category" tick={{ fontSize: 9 }} width={80} />
            <Tooltip />
            <Bar dataKey="value" fill={color} radius={[0,2,2,0]} />
          </BarChart>
        </ResponsiveContainer>
      ) : <EmptyState title="No data" />}
    </div>
  )
}
