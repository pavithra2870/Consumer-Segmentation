import { useState, useEffect } from 'react'
import Header from '../Layout/Header'
import LoadingSpinner, { EmptyState } from '../Common/LoadingSpinner'
import { api } from '../../services/api'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, PieChart, Pie, Cell, Legend,
} from 'recharts'

const SEGMENT_COLORS = ['#388bfd','#f85149','#3fb950','#d29922','#8957e5','#db6d28','#39c5cf','#e36d9b']

export default function Overview() {
  const [health, setHealth] = useState(null)
  const [segments, setSegments] = useState(null)
  const [predictions, setPredictions] = useState(null)
  const [dataInfo, setDataInfo] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.allSettled([
      api.health(),
      api.getSegments(),
      api.getPredictions(),
      api.getDataInfo(),
    ]).then(([h, s, p, d]) => {
      if (h.status === 'fulfilled') setHealth(h.value)
      if (s.status === 'fulfilled') setSegments(s.value)
      if (p.status === 'fulfilled') setPredictions(p.value)
      if (d.status === 'fulfilled') setDataInfo(d.value)
      setLoading(false)
    })
  }, [])

  if (loading) return <><Header title="Executive Overview" subtitle="Platform KPIs and segment summary"/><div className="page-body"><LoadingSpinner message="Loading dashboard..." /></div></>

  const totalCustomers = dataInfo?.rows ?? 0
  const avgChurn = segments?.segments
    ? (segments.segments.reduce((s, x) => s + x.avg_churn_risk * x.size, 0) / (totalCustomers || 1) * 100).toFixed(1)
    : '--'
  const avgLtv = segments?.segments
    ? (segments.segments.reduce((s, x) => s + x.avg_ltv * x.size, 0) / (totalCustomers || 1)).toFixed(0)
    : '--'
  const nSegments = segments?.n_clusters ?? '--'

  const segmentPieData = segments?.segments?.map(s => ({
    name: s.label,
    value: s.size,
  })) ?? []

  const churnDist = predictions?.churn?.distribution ?? []
  const ltvDist = predictions?.ltv?.distribution?.slice(0, 15) ?? []

  const featureImportance = predictions?.churn?.feature_importance
    ? Object.entries(predictions.churn.feature_importance).map(([k, v]) => ({ feature: k.replace(/_/g,' '), importance: +(v*100).toFixed(2) }))
    : []

  return (
    <>
      <Header
        title="Executive Overview"
        subtitle="Consumer Intelligence Platform — key performance indicators"
      />
      <div className="page-body">
        {/* KPIs */}
        <div className="kpi-grid">
          <KPICard label="Total Customers" value={totalCustomers.toLocaleString()} delta={null} color="" />
          <KPICard label="Segments" value={nSegments} delta={null} color="accent-blue" />
          <KPICard label="Avg Churn Risk" value={`${avgChurn}%`} delta="Predicted" color="accent-red" />
          <KPICard label="Avg Customer LTV" value={`$${Number(avgLtv).toLocaleString()}`} delta="Estimated" color="accent-green" />
          <KPICard label="Silhouette Score" value={segments?.silhouette_score?.toFixed(3) ?? '--'} delta="Clustering quality" color="accent-yellow" />
          <KPICard
            label="Churn Model AUC"
            value={predictions?.churn?.auc ? predictions.churn.auc.toFixed(3) : (predictions?.churn?.accuracy ? predictions.churn.accuracy.toFixed(3) : '--')}
            delta="Classification"
            color="accent-purple"
          />
        </div>

        {/* Charts row 1 */}
        <div className="grid-2 mb-24">
          <div className="card">
            <div className="card-header">
              <div>
                <div className="card-title">Segment Distribution</div>
                <div className="card-subtitle">Customers per cluster</div>
              </div>
            </div>
            {segmentPieData.length > 0 ? (
              <ResponsiveContainer width="100%" height={240}>
                <PieChart>
                  <Pie data={segmentPieData} cx="50%" cy="50%" outerRadius={90} dataKey="value" label={({ name, percent }) => `${(percent*100).toFixed(0)}%`} labelLine={false}>
                    {segmentPieData.map((_, i) => <Cell key={i} fill={SEGMENT_COLORS[i % SEGMENT_COLORS.length]} />)}
                  </Pie>
                  <Tooltip formatter={(v) => v.toLocaleString()} />
                  <Legend iconType="circle" iconSize={8} />
                </PieChart>
              </ResponsiveContainer>
            ) : <EmptyState title="No segment data" message="Run the training pipeline first" />}
          </div>

          <div className="card">
            <div className="card-header">
              <div>
                <div className="card-title">Churn Probability Distribution</div>
                <div className="card-subtitle">Predicted churn score histogram</div>
              </div>
            </div>
            {churnDist.length > 0 ? (
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={churnDist} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="range" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#f85149" radius={[2,2,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : <EmptyState title="No prediction data" message="Run the training pipeline first" />}
          </div>
        </div>

        {/* Charts row 2 */}
        <div className="grid-2 mb-24">
          <div className="card">
            <div className="card-header">
              <div>
                <div className="card-title">LTV Distribution</div>
                <div className="card-subtitle">Customer lifetime value histogram</div>
              </div>
            </div>
            {ltvDist.length > 0 ? (
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={ltvDist} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="range" tick={{ fontSize: 9 }} angle={-30} textAnchor="end" height={40} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#3fb950" radius={[2,2,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : <EmptyState title="No LTV data" />}
          </div>

          <div className="card">
            <div className="card-header">
              <div>
                <div className="card-title">Top Churn Predictors</div>
                <div className="card-subtitle">Feature importance for churn model</div>
              </div>
            </div>
            {featureImportance.length > 0 ? (
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={featureImportance.slice(0,8)} layout="vertical" margin={{ top: 4, right: 20, left: 90, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" horizontal={false} />
                  <XAxis type="number" tick={{ fontSize: 10 }} />
                  <YAxis dataKey="feature" type="category" tick={{ fontSize: 10 }} width={90} />
                  <Tooltip />
                  <Bar dataKey="importance" fill="#388bfd" radius={[0,2,2,0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : <EmptyState title="No feature data" />}
          </div>
        </div>

        {/* Segment table */}
        {segments?.segments && (
          <div className="card">
            <div className="card-header">
              <div className="card-title">Segment Summary</div>
            </div>
            <div className="data-table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Segment</th>
                    <th>Label</th>
                    <th>Customers</th>
                    <th>Share</th>
                    <th>Avg LTV</th>
                    <th>Churn Risk</th>
                  </tr>
                </thead>
                <tbody>
                  {segments.segments.map(s => (
                    <tr key={s.segment_id}>
                      <td>
                        <div className="flex items-center gap-8">
                          <span className="seg-dot" style={{ background: SEGMENT_COLORS[s.segment_id % SEGMENT_COLORS.length] }} />
                          <span className="text-mono">{s.segment_id}</span>
                        </div>
                      </td>
                      <td>{s.label}</td>
                      <td>{s.size.toLocaleString()}</td>
                      <td>{s.percentage}%</td>
                      <td>${s.avg_ltv.toLocaleString()}</td>
                      <td>
                        <span className={`badge ${s.avg_churn_risk > 0.5 ? 'badge-red' : s.avg_churn_risk > 0.3 ? 'badge-yellow' : 'badge-green'}`}>
                          {(s.avg_churn_risk * 100).toFixed(1)}%
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </>
  )
}

function KPICard({ label, value, delta, color }) {
  return (
    <div className="kpi-card">
      <div className="kpi-label">{label}</div>
      <div className={`kpi-value ${color}`}>{value}</div>
      {delta && <div className="kpi-delta neutral">{delta}</div>}
    </div>
  )
}
