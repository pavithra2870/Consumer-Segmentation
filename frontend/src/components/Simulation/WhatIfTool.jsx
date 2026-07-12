import { useState } from 'react'
import Header from '../Layout/Header'
import { api } from '../../services/api'

const DEFAULT_FEATURES = {
  rfm_recency: 0.5,
  rfm_frequency: 2.0,
  rfm_monetary: 8.0,
  rfm_score: 0.5,
  engagement_score: 0.4,
  support_burden: 0.05,
  session_quality_score: 0.5,
  last_purchase_days_ago: 60,
  nps_score: 7,
  email_open_rate: 0.25,
  app_opens_30d: 30,
  support_tickets: 2,
  total_spend: 1500,
  avg_order_value: 85,
  purchase_frequency_monthly: 2.5,
  account_age_days: 365,
  session_count_30d: 20,
  digital_activity_score: 5.0,
  subscription_type_enc: 1,
  income_bracket_enc: 1,
  email_click_rate: 0.08,
  pages_per_session: 6.0,
}

const SLIDER_CONFIG = [
  { key: 'rfm_recency', label: 'RFM Recency Score', min: 0, max: 1, step: 0.01 },
  { key: 'rfm_frequency', label: 'RFM Frequency (log)', min: 0, max: 4, step: 0.1 },
  { key: 'engagement_score', label: 'Engagement Score', min: 0, max: 1, step: 0.01 },
  { key: 'email_open_rate', label: 'Email Open Rate', min: 0, max: 1, step: 0.01 },
  { key: 'nps_score', label: 'NPS Score', min: 0, max: 10, step: 1 },
  { key: 'last_purchase_days_ago', label: 'Days Since Last Purchase', min: 1, max: 365, step: 1 },
  { key: 'support_tickets', label: 'Support Tickets', min: 0, max: 20, step: 1 },
  { key: 'session_count_30d', label: 'Sessions (30d)', min: 0, max: 200, step: 1 },
  { key: 'app_opens_30d', label: 'App Opens (30d)', min: 0, max: 300, step: 1 },
  { key: 'total_spend', label: 'Total Spend ($)', min: 10, max: 50000, step: 50 },
]

export default function WhatIfTool() {
  const [features, setFeatures] = useState({ ...DEFAULT_FEATURES })
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleChange = (key, value) => {
    setFeatures(prev => ({ ...prev, [key]: parseFloat(value) }))
  }

  const handlePredict = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await api.whatIf(features)
      setResult(res)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setFeatures({ ...DEFAULT_FEATURES })
    setResult(null)
  }

  return (
    <>
      <Header
        title="What-If Simulator"
        subtitle="Adjust customer feature values and predict outcomes in real-time"
      />
      <div className="page-body">
        <div className="grid-2">
          {/* Controls */}
          <div className="card">
            <div className="card-header">
              <div className="card-title">Customer Feature Configuration</div>
              <button className="btn btn-secondary btn-sm" onClick={handleReset}>Reset</button>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              {SLIDER_CONFIG.map(({ key, label, min, max, step }) => (
                <div key={key}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                    <label className="form-label" style={{ marginBottom: 0 }}>{label}</label>
                    <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--accent-blue)' }}>
                      {features[key]}
                    </span>
                  </div>
                  <input
                    type="range"
                    className="form-range"
                    min={min}
                    max={max}
                    step={step}
                    value={features[key]}
                    onChange={e => handleChange(key, e.target.value)}
                  />
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--text-muted)' }}>
                    <span>{min}</span>
                    <span>{max}</span>
                  </div>
                </div>
              ))}
            </div>
            <button
              className="btn btn-primary btn-lg"
              style={{ marginTop: 24, width: '100%' }}
              onClick={handlePredict}
              disabled={loading}
            >
              {loading ? 'Predicting...' : 'Run Prediction'}
            </button>
            {error && <div className="alert alert-error" style={{ marginTop: 12 }}>{error}</div>}
          </div>

          {/* Results */}
          <div>
            {!result && (
              <div className="card" style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <div style={{ textAlign: 'center', color: 'var(--text-muted)' }}>
                  <div style={{ fontSize: 40, marginBottom: 12 }}>⊙</div>
                  <div style={{ fontSize: 13 }}>Configure features and click "Run Prediction"</div>
                </div>
              </div>
            )}
            {result && (
              <>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12, marginBottom: 16 }}>
                  <ResultMetric
                    label="Churn Probability"
                    value={`${(result.churn_probability * 100).toFixed(1)}%`}
                    color={result.churn_probability > 0.5 ? '#f85149' : result.churn_probability > 0.3 ? '#d29922' : '#3fb950'}
                    description={result.churn_probability > 0.5 ? 'High Risk' : result.churn_probability > 0.3 ? 'Medium Risk' : 'Low Risk'}
                  />
                  <ResultMetric
                    label="LTV Estimate"
                    value={`$${Number(result.ltv_estimate).toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
                    color="#3fb950"
                    description="Lifetime Value"
                  />
                  <ResultMetric
                    label="Conversion Score"
                    value={`${(result.conversion_probability * 100).toFixed(1)}%`}
                    color={result.conversion_probability > 0.5 ? '#3fb950' : result.conversion_probability > 0.3 ? '#d29922' : '#388bfd'}
                    description={result.conversion_probability > 0.5 ? 'High Likelihood' : 'Moderate'}
                  />
                </div>

                {/* Gauge-style visualisation */}
                <div className="card mb-12">
                  <div className="card-title mb-16">Risk & Opportunity Summary</div>
                  <GaugeMeter label="Churn Risk" value={result.churn_probability} color="#f85149" />
                  <GaugeMeter label="Conversion Opportunity" value={result.conversion_probability} color="#3fb950" />
                  <GaugeMeter label="LTV Percentile (est.)" value={Math.min(result.ltv_estimate / 30000, 1)} color="#388bfd" />
                </div>

                <div className="card">
                  <div className="card-title mb-16">Recommended Actions</div>
                  {result.recommendations?.map((r, i) => (
                    <div key={i} style={{ display: 'flex', gap: 10, padding: '10px 0', borderBottom: i < result.recommendations.length - 1 ? '1px solid var(--border)' : 'none' }}>
                      <span style={{ color: 'var(--accent-blue)', fontWeight: 700, fontSize: 12 }}>{i + 1}</span>
                      <span style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.5 }}>{r}</span>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </>
  )
}

function ResultMetric({ label, value, color, description }) {
  return (
    <div className="card" style={{ textAlign: 'center' }}>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 6 }}>{label}</div>
      <div style={{ fontSize: 28, fontWeight: 700, color, fontVariantNumeric: 'tabular-nums' }}>{value}</div>
      <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginTop: 4 }}>{description}</div>
    </div>
  )
}

function GaugeMeter({ label, value, color }) {
  const pct = Math.round(Math.min(Math.max(value, 0), 1) * 100)
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5 }}>
        <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{label}</span>
        <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color }}>{pct}%</span>
      </div>
      <div className="progress-bar-wrap">
        <div className="progress-bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  )
}
