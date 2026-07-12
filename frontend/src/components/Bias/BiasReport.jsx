import Header from '../Layout/Header'
import LoadingSpinner, { EmptyState } from '../Common/LoadingSpinner'
import { useApi } from '../../hooks/useApi'
import { api } from '../../services/api'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from 'recharts'

export default function BiasReport() {
  const { data, loading, error } = useApi(api.getBiasReport)

  const genderDist = data?.distribution_by_group?.gender
    ? Object.entries(data.distribution_by_group.gender).map(([k, v]) => ({ name: k, value: v }))
    : []
  const incomeDist = data?.distribution_by_group?.income_bracket
    ? Object.entries(data.distribution_by_group.income_bracket).map(([k, v]) => ({ name: k, value: v }))
    : []

  return (
    <>
      <Header
        title="Fairness Audit"
        subtitle="Bias detection across demographic attributes — 80% adverse impact rule"
      />
      <div className="page-body">
        {loading && <LoadingSpinner message="Analysing fairness metrics..." />}
        {error && <div className="alert alert-error">{error}</div>}
        {data && (
          <>
            {/* Summary alert */}
            <div className={`alert ${data.overall_flag ? 'alert-warning' : 'alert-success'} mb-24`}>
              <span style={{ fontSize: 16 }}>{data.overall_flag ? '⚠' : '✓'}</span>
              <div>
                <div style={{ fontWeight: 600, marginBottom: 4 }}>
                  {data.overall_flag ? 'Fairness Issues Detected' : 'All Groups Pass Fairness Thresholds'}
                </div>
                <div style={{ fontSize: 12, opacity: 0.85 }}>{data.summary}</div>
              </div>
            </div>

            {/* Metrics table */}
            <div className="card mb-24">
              <div className="card-header">
                <div className="card-title">Fairness Metrics by Group</div>
                <div className="card-subtitle">Adverse Impact Ratio · Demographic Parity Gap · 80% rule threshold</div>
              </div>
              <div className="data-table-wrap">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Attribute</th>
                      <th>Group</th>
                      <th>Prediction</th>
                      <th>Positive Rate</th>
                      <th>Adverse Impact Ratio</th>
                      <th>Parity Gap</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.metrics?.map((m, i) => (
                      <tr key={i}>
                        <td style={{ color: 'var(--text-secondary)', fontSize: 11 }}>{m.attribute}</td>
                        <td style={{ fontWeight: 500 }}>{m.group}</td>
                        <td>
                          <span className="badge badge-muted">{m.prediction}</span>
                        </td>
                        <td className="mono">{(m.positive_rate * 100).toFixed(2)}%</td>
                        <td>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            <div className="progress-bar-wrap" style={{ width: 50, display: 'inline-block' }}>
                              <div
                                className="progress-bar-fill"
                                style={{
                                  width: `${Math.min(m.adverse_impact_ratio, 1) * 100}%`,
                                  background: m.adverse_impact_ratio < 0.8 ? '#f85149' : '#3fb950',
                                }}
                              />
                            </div>
                            <span className="mono" style={{ fontSize: 11 }}>{m.adverse_impact_ratio?.toFixed(3)}</span>
                          </div>
                        </td>
                        <td className="mono">{(m.parity_gap * 100).toFixed(2)}pp</td>
                        <td>
                          <span className={`badge ${m.flagged ? 'badge-red' : 'badge-green'}`}>
                            {m.flagged ? 'FLAGGED' : 'PASS'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Distribution charts */}
            <div className="grid-2 mb-24">
              <div className="card">
                <div className="card-header"><div className="card-title">Gender Distribution</div></div>
                <ResponsiveContainer width="100%" height={180}>
                  <BarChart data={genderDist} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                    <YAxis tick={{ fontSize: 10 }} />
                    <Tooltip />
                    <Bar dataKey="value" fill="#388bfd" radius={[2,2,0,0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="card">
                <div className="card-header"><div className="card-title">Income Bracket Distribution</div></div>
                <ResponsiveContainer width="100%" height={180}>
                  <BarChart data={incomeDist} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                    <YAxis tick={{ fontSize: 10 }} />
                    <Tooltip />
                    <Bar dataKey="value" fill="#d29922" radius={[2,2,0,0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Recommendations */}
            <div className="card">
              <div className="card-header"><div className="card-title">Mitigation Recommendations</div></div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {data.recommendations?.map((r, i) => (
                  <div key={i} style={{ display: 'flex', gap: 12, alignItems: 'flex-start', padding: '10px 14px', background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                    <span style={{ color: 'var(--accent-blue)', fontWeight: 700, fontSize: 11, flexShrink: 0 }}>REC {i + 1}</span>
                    <span style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.6 }}>{r}</span>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
      </div>
    </>
  )
}
