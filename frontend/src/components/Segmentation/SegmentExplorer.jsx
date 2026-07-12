import { useState } from 'react'
import Header from '../Layout/Header'
import LoadingSpinner, { EmptyState } from '../Common/LoadingSpinner'
import { useApi } from '../../hooks/useApi'
import { api } from '../../services/api'
import {
  ScatterChart, Scatter, XAxis, YAxis, Tooltip, ResponsiveContainer,
  BarChart, Bar, CartesianGrid, Cell,
} from 'recharts'

const SEGMENT_COLORS = ['#388bfd','#f85149','#3fb950','#d29922','#8957e5','#db6d28','#39c5cf','#e36d9b']

export default function SegmentExplorer() {
  const { data, loading, error } = useApi(api.getSegments)
  const [selected, setSelected] = useState(null)
  const [detailData, setDetailData] = useState(null)
  const [detailLoading, setDetailLoading] = useState(false)

  const loadDetail = async (id) => {
    if (selected === id) { setSelected(null); setDetailData(null); return }
    setSelected(id)
    setDetailLoading(true)
    try {
      const d = await api.getSegmentDetail(id)
      setDetailData(d)
    } catch { setDetailData(null) }
    setDetailLoading(false)
  }

  return (
    <>
      <Header
        title="Segmentation Explorer"
        subtitle={data ? `${data.n_clusters} clusters · ${data.method?.toUpperCase()} · silhouette ${data.silhouette_score}` : 'Cluster visualization and analysis'}
      />
      <div className="page-body">
        {loading && <LoadingSpinner message="Loading segmentation..." />}
        {error && <div className="alert alert-error">{error}</div>}
        {data && (
          <>
            {/* Scatter plot */}
            <div className="card mb-24">
              <div className="card-header">
                <div>
                  <div className="card-title">Cluster Scatter Plot (PCA 2D)</div>
                  <div className="card-subtitle">First two principal components — {data.scatter_data?.length?.toLocaleString()} sampled points</div>
                </div>
                <div className="flex gap-8" style={{ flexWrap: 'wrap' }}>
                  {data.segments?.map(s => (
                    <button
                      key={s.segment_id}
                      className={`badge ${selected === s.segment_id ? 'badge-blue' : 'badge-muted'}`}
                      style={{ cursor: 'pointer', border: 'none' }}
                      onClick={() => loadDetail(s.segment_id)}
                    >
                      <span className="seg-dot" style={{ background: SEGMENT_COLORS[s.segment_id % SEGMENT_COLORS.length], marginRight: 4, width:7,height:7 }} />
                      {s.label}
                    </button>
                  ))}
                </div>
              </div>
              <ResponsiveContainer width="100%" height={400}>
                <ScatterChart margin={{ top: 10, right: 20, left: -10, bottom: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis type="number" dataKey="x" name="PC1" tick={{ fontSize: 10 }} />
                  <YAxis type="number" dataKey="y" name="PC2" tick={{ fontSize: 10 }} />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} content={<ScatterTooltip />} />
                  {data.segments?.map(seg => {
                    const points = data.scatter_data?.filter(p => p.segment === seg.segment_id) ?? []
                    return (
                      <Scatter
                        key={seg.segment_id}
                        name={seg.label}
                        data={points}
                        fill={SEGMENT_COLORS[seg.segment_id % SEGMENT_COLORS.length]}
                        opacity={selected === null || selected === seg.segment_id ? 0.75 : 0.12}
                        r={3}
                      />
                    )
                  })}
                </ScatterChart>
              </ResponsiveContainer>
            </div>

            {/* Segment cards */}
            <div className="grid-auto mb-24">
              {data.segments?.map(s => (
                <SegmentCard
                  key={s.segment_id}
                  segment={s}
                  color={SEGMENT_COLORS[s.segment_id % SEGMENT_COLORS.length]}
                  active={selected === s.segment_id}
                  onClick={() => loadDetail(s.segment_id)}
                />
              ))}
            </div>

            {/* Detail panel */}
            {selected !== null && (
              <div className="card">
                <div className="card-header">
                  <div className="card-title">
                    Segment {selected} — Detailed Statistics
                  </div>
                </div>
                {detailLoading && <LoadingSpinner message="Loading segment detail..." />}
                {detailData && <SegmentDetail detail={detailData} color={SEGMENT_COLORS[selected % SEGMENT_COLORS.length]} />}
              </div>
            )}
          </>
        )}
      </div>
    </>
  )
}

function SegmentCard({ segment, color, active, onClick }) {
  const churnPct = (segment.avg_churn_risk * 100).toFixed(1)
  const churnClass = segment.avg_churn_risk > 0.5 ? 'badge-red' : segment.avg_churn_risk > 0.3 ? 'badge-yellow' : 'badge-green'

  return (
    <div
      className="card"
      style={{ cursor: 'pointer', borderColor: active ? color : undefined, transition: 'border-color 150ms' }}
      onClick={onClick}
    >
      <div className="flex items-center gap-8 mb-16">
        <div style={{ width: 10, height: 10, borderRadius: '50%', background: color, flexShrink: 0 }} />
        <div className="card-title">{segment.label}</div>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
        <Stat label="Customers" value={segment.size.toLocaleString()} />
        <Stat label="Share" value={`${segment.percentage}%`} />
        <Stat label="Avg LTV" value={`$${segment.avg_ltv.toLocaleString()}`} />
        <div>
          <div className="text-xs text-muted" style={{ marginBottom: 4 }}>Churn Risk</div>
          <span className={`badge ${churnClass}`}>{churnPct}%</span>
        </div>
      </div>
    </div>
  )
}

function Stat({ label, value }) {
  return (
    <div>
      <div className="text-xs text-muted" style={{ marginBottom: 2 }}>{label}</div>
      <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-primary)' }}>{value}</div>
    </div>
  )
}

function SegmentDetail({ detail, color }) {
  const subMix = Object.entries(detail.subscription_mix || {}).map(([k, v]) => ({ name: k, value: v }))
  const incomeMix = Object.entries(detail.income_mix || {}).map(([k, v]) => ({ name: k, value: v }))

  return (
    <div>
      <div className="grid-2 mb-16">
        <div>
          <div className="card-subtitle mb-16">Subscription Mix</div>
          <ResponsiveContainer width="100%" height={160}>
            <BarChart data={subMix} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
              <XAxis dataKey="name" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10 }} />
              <Tooltip />
              <Bar dataKey="value" fill={color} radius={[2,2,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div>
          <div className="card-subtitle mb-16">Income Distribution</div>
          <ResponsiveContainer width="100%" height={160}>
            <BarChart data={incomeMix} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
              <XAxis dataKey="name" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10 }} />
              <Tooltip />
              <Bar dataKey="value" fill={color} radius={[2,2,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
      <div className="data-table-wrap">
        <table className="data-table">
          <thead>
            <tr><th>Customer ID</th><th>Age</th><th>Spend</th><th>LTV</th><th>Churn Prob</th><th>NPS</th></tr>
          </thead>
          <tbody>
            {detail.sample_customers?.slice(0, 8).map((c, i) => (
              <tr key={i}>
                <td className="mono">{c.customer_id}</td>
                <td>{c.age}</td>
                <td>${Number(c.total_spend || 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                <td>${Number(c.ltv_actual || 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                <td>
                  <span className={`badge ${c.churn_probability > 0.5 ? 'badge-red' : c.churn_probability > 0.3 ? 'badge-yellow' : 'badge-green'}`}>
                    {(c.churn_probability * 100).toFixed(1)}%
                  </span>
                </td>
                <td>{c.nps_score}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function ScatterTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const d = payload[0]?.payload
  return (
    <div style={{ background: 'var(--bg-tertiary)', border: '1px solid var(--border)', borderRadius: 6, padding: '8px 12px', fontSize: 11 }}>
      <div style={{ color: 'var(--text-secondary)' }}>Segment {d?.segment}</div>
      <div>PC1: {d?.x?.toFixed(3)}</div>
      <div>PC2: {d?.y?.toFixed(3)}</div>
    </div>
  )
}
