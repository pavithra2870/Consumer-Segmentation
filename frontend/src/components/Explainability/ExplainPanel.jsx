import { useState } from 'react'
import Header from '../Layout/Header'
import LoadingSpinner, { EmptyState } from '../Common/LoadingSpinner'
import { useApi } from '../../hooks/useApi'
import { api } from '../../services/api'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from 'recharts'

const MODELS = ['churn', 'ltv', 'conversion']
const MODEL_COLORS = { churn: '#f85149', ltv: '#3fb950', conversion: '#388bfd' }
const MODEL_LABELS = { churn: 'Churn Prediction', ltv: 'LTV Regression', conversion: 'Conversion Scoring' }

export default function ExplainPanel() {
  const { data, loading, error } = useApi(api.getExplanations)
  const [activeModel, setActiveModel] = useState('churn')

  const current = data?.[activeModel]

  return (
    <>
      <Header
        title="Explainability"
        subtitle="SHAP-based feature importance and per-customer attribution"
      />
      <div className="page-body">
        {loading && <LoadingSpinner message="Computing explanations..." />}
        {error && <div className="alert alert-error">{error}</div>}
        {data && (
          <>
            <div className="tab-bar mb-24" style={{ marginBottom: 0 }}>
              {MODELS.map(m => (
                <button key={m} className={`tab-item${activeModel === m ? ' active' : ''}`} onClick={() => setActiveModel(m)}>
                  {MODEL_LABELS[m]}
                </button>
              ))}
            </div>

            {current && (
              <div style={{ marginTop: 20 }}>
                {/* Global importance chart */}
                <div className="card mb-24">
                  <div className="card-header">
                    <div>
                      <div className="card-title">Global Feature Importance — {MODEL_LABELS[activeModel]}</div>
                      <div className="card-subtitle">Mean absolute SHAP values across all customers</div>
                    </div>
                  </div>
                  {current.global_importance?.length > 0 ? (
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart
                        data={current.global_importance.slice(0, 12).map(f => ({
                          feature: f.feature.replace(/_/g, ' '),
                          value: +(f.importance * 1000).toFixed(2),
                        }))}
                        layout="vertical"
                        margin={{ top: 4, right: 20, left: 130, bottom: 4 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" horizontal={false} />
                        <XAxis type="number" tick={{ fontSize: 10 }} />
                        <YAxis dataKey="feature" type="category" tick={{ fontSize: 10 }} width={130} />
                        <Tooltip formatter={(v) => (v / 1000).toFixed(5)} />
                        <Bar dataKey="value" fill={MODEL_COLORS[activeModel]} radius={[0,2,2,0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  ) : <EmptyState title="No importance data" />}
                </div>

                {/* Top features table */}
                <div className="grid-2 mb-24">
                  <div className="card">
                    <div className="card-title mb-16">Top Predictors</div>
                    <div className="data-table-wrap">
                      <table className="data-table">
                        <thead>
                          <tr>
                            <th>#</th>
                            <th>Feature</th>
                            <th>Mean |SHAP|</th>
                            <th>Direction</th>
                          </tr>
                        </thead>
                        <tbody>
                          {current.top_features?.slice(0, 10).map((f, i) => (
                            <tr key={i}>
                              <td className="mono" style={{ color: 'var(--text-muted)' }}>{i + 1}</td>
                              <td>{f.feature.replace(/_/g, ' ')}</td>
                              <td className="mono">{f.mean_abs_shap?.toFixed(5)}</td>
                              <td>
                                <span className={`badge ${f.direction === 'positive' ? 'badge-green' : 'badge-red'}`}>
                                  {f.direction}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  <div className="card">
                    <div className="card-title mb-16">Sample Customer Explanations</div>
                    {current.segment_reasoning?.length > 0 ? (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                        {current.segment_reasoning.slice(0, 3).map((row, i) => (
                          <div key={i} style={{ background: 'var(--bg-tertiary)', borderRadius: 6, padding: 12 }}>
                            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }}>
                              Customer index: {row.customer_idx}
                            </div>
                            {row.top_contributors?.map((c, j) => (
                              <div key={j} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 4 }}>
                                <span style={{ color: 'var(--text-secondary)' }}>{c.feature.replace(/_/g, ' ')}</span>
                                <span style={{ fontFamily: 'var(--font-mono)', color: c.shap_value >= 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                                  {c.shap_value >= 0 ? '+' : ''}{c.shap_value?.toFixed(4)}
                                </span>
                              </div>
                            ))}
                          </div>
                        ))}
                      </div>
                    ) : (
                      <EmptyState title="SHAP sample explanations not available" message="Requires SHAP library installed" />
                    )}
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </>
  )
}
