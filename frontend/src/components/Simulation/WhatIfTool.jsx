import { useState, useEffect } from 'react'
import Header from '../Layout/Header'
import LoadingSpinner from '../Common/LoadingSpinner'
import { api } from '../../services/api'

export default function WhatIfTool() {
  const [sliders, setSliders] = useState([])
  const [meta, setMeta] = useState(null)
  const [features, setFeatures] = useState({})
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [schemaLoading, setSchemaLoading] = useState(true)
  const [error, setError] = useState(null)

  // Sliders are the most influential features for this dataset, with ranges taken from its data
  useEffect(() => {
    api.getWhatIfSchema()
      .then((s) => {
        setSliders(s.sliders)
        setMeta(s.meta)
        setFeatures(Object.fromEntries(s.sliders.map((x) => [x.key, x.default])))
      })
      .catch((e) => setError(e.message))
      .finally(() => setSchemaLoading(false))
  }, [])

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
    setFeatures(Object.fromEntries(sliders.map((x) => [x.key, x.default])))
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
              {schemaLoading && <LoadingSpinner message="Loading feature ranges..." />}
              {sliders.map(({ key, label, min, max, step }) => (
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
                <div style={{ display: 'grid', gridTemplateColumns: `repeat(${[result.churn_probability, result.ltv_estimate, result.conversion_probability].filter((v) => v !== undefined).length}, 1fr)`, gap: 12, marginBottom: 16 }}>
                  {result.churn_probability !== undefined && (
                    <ResultMetric
                      label="Churn Probability"
                      value={`${(result.churn_probability * 100).toFixed(1)}%`}
                      color={riskColor(result.churn_probability, result.base_rates?.churn, true)}
                      description={`Average ${(result.base_rates.churn * 100).toFixed(0)}%`}
                    />
                  )}
                  {result.ltv_estimate !== undefined && (
                    <ResultMetric
                      label={`${meta?.value_label ?? 'LTV'} Estimate`}
                      value={`${meta?.value_prefix ?? '$'}${Number(result.ltv_estimate).toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
                      color="#3fb950"
                      description="Model estimate"
                    />
                  )}
                  {result.conversion_probability !== undefined && (
                    <ResultMetric
                      label="Conversion Score"
                      value={`${(result.conversion_probability * 100).toFixed(1)}%`}
                      color={riskColor(result.conversion_probability, result.base_rates?.conversion, false)}
                      description={`Average ${(result.base_rates.conversion * 100).toFixed(0)}%`}
                    />
                  )}
                </div>

                <div className="card mb-12">
                  <div className="card-title mb-16">Risk & Opportunity Summary</div>
                  {result.churn_probability !== undefined && <GaugeMeter label="Churn Risk" value={result.churn_probability} color="#f85149" />}
                  {result.conversion_probability !== undefined && <GaugeMeter label="Conversion Opportunity" value={result.conversion_probability} color="#3fb950" />}
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

// Colour relative to the dataset's own average rate instead of fixed thresholds
function riskColor(value, base, higherIsWorse) {
  const ratio = base ? value / base : 1
  const bad = higherIsWorse ? ratio > 1.5 : ratio < 0.75
  const good = higherIsWorse ? ratio < 0.75 : ratio > 1.5
  return bad ? '#f85149' : good ? '#3fb950' : '#d29922'
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
