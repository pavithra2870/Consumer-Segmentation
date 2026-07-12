import { useState, useEffect, useRef } from 'react'
import Header from '../Layout/Header'
import { api } from '../../services/api'

const STEPS = [
  { id: 'generate', label: 'Generate Dataset', description: '100k+ synthetic consumer records', endpoint: 'data' },
  { id: 'train', label: 'Train Models', description: 'Segmentation + Prediction pipeline', endpoint: 'models' },
]

export default function Pipeline() {
  const [health, setHealth] = useState(null)
  const [dataStatus, setDataStatus] = useState(null)
  const [trainStatus, setTrainStatus] = useState(null)
  const [dataInfo, setDataInfo] = useState(null)
  const [nRows, setNRows] = useState(100000)
  const [nClusters, setNClusters] = useState(6)
  const [method, setMethod] = useState('kmeans')
  const [log, setLog] = useState([])
  const logRef = useRef(null)

  const addLog = (msg, type = 'info') => {
    const ts = new Date().toLocaleTimeString()
    setLog(prev => [...prev.slice(-199), { ts, msg, type }])
  }

  // Poll status while running
  useEffect(() => {
    let active = true
    const poll = async () => {
      try {
        const [h, ds, ts] = await Promise.all([
          api.health(),
          api.getDataStatus(),
          api.getTrainStatus(),
        ])
        if (!active) return
        setHealth(h)
        setDataStatus(ds)
        setTrainStatus(ts)

        if (h.data_ready) {
          try { const di = await api.getDataInfo(); setDataInfo(di) } catch {}
        }
      } catch {}
      if (active) setTimeout(poll, 3000)
    }
    poll()
    return () => { active = false }
  }, [])

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [log])

  const handleGenerate = async () => {
    addLog(`Triggering data generation (${nRows.toLocaleString()} rows)...`, 'info')
    try {
      const res = await api.generateData(nRows)
      addLog(res.message, 'success')
    } catch (e) {
      addLog(`Error: ${e.message}`, 'error')
    }
  }

  const handleTrain = async () => {
    addLog(`Starting training pipeline (${method}, ${nClusters} clusters)...`, 'info')
    try {
      const res = await api.trainModels(nClusters, method)
      addLog(res.message, 'success')
    } catch (e) {
      addLog(`Error: ${e.message}`, 'error')
    }
  }

  const dataReady = health?.data_ready
  const modelsReady = health?.models_ready

  const dataRunning = dataStatus?.status === 'running'
  const trainRunning = trainStatus?.status === 'running'

  return (
    <>
      <Header
        title="Data Pipeline"
        subtitle="Generate data and train all models end-to-end"
      />
      <div className="page-body">
        {/* System status */}
        <div className="grid-3 mb-24">
          <StatusCard
            label="Ollama LLM"
            status={health?.ollama_available ? 'ok' : 'warning'}
            value={health?.ollama_available ? 'Connected' : 'Offline'}
            detail="Local model server"
          />
          <StatusCard
            label="Dataset"
            status={dataReady ? 'ok' : 'pending'}
            value={dataReady ? `${dataInfo?.rows?.toLocaleString() ?? '?'} rows` : 'Not generated'}
            detail={dataReady ? `${dataInfo?.columns ?? '?'} features · ${dataInfo?.size_mb ?? '?'} MB` : 'Run Step 1'}
          />
          <StatusCard
            label="Models"
            status={modelsReady ? 'ok' : 'pending'}
            value={modelsReady ? 'All trained' : 'Not trained'}
            detail={modelsReady ? 'Segmentation + Predictions' : 'Run Step 2'}
          />
        </div>

        <div className="grid-2">
          {/* Step controls */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {/* Step 1 */}
            <div className="card">
              <div className="flex items-center gap-12 mb-16">
                <PipelineStepBadge num={1} done={dataReady} running={dataRunning} />
                <div>
                  <div className="card-title">Generate Synthetic Dataset</div>
                  <div className="card-subtitle">Creates consumer profiles with demographics, behavioural signals, and reviews</div>
                </div>
              </div>

              <div className="form-group">
                <label className="form-label">Number of rows</label>
                <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <input
                    type="range"
                    className="form-range"
                    min={10000}
                    max={500000}
                    step={10000}
                    value={nRows}
                    onChange={e => setNRows(+e.target.value)}
                  />
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--accent-blue)', minWidth: 70 }}>
                    {nRows.toLocaleString()}
                  </span>
                </div>
              </div>

              {dataStatus && dataStatus.status !== 'idle' && (
                <div className={`alert ${dataStatus.status === 'failed' ? 'alert-error' : dataStatus.status === 'complete' ? 'alert-success' : 'alert-info'} mb-16`} style={{ fontSize: 12 }}>
                  {dataStatus.message}
                </div>
              )}

              <button
                className="btn btn-primary"
                style={{ width: '100%' }}
                onClick={handleGenerate}
                disabled={dataRunning}
              >
                {dataRunning ? (
                  <><span className="spinner" style={{ width: 14, height: 14 }} /> Generating...</>
                ) : (
                  dataReady ? 'Re-generate Dataset' : 'Generate Dataset'
                )}
              </button>
            </div>

            {/* Step 2 */}
            <div className="card" style={{ opacity: dataReady ? 1 : 0.6 }}>
              <div className="flex items-center gap-12 mb-16">
                <PipelineStepBadge num={2} done={modelsReady} running={trainRunning} />
                <div>
                  <div className="card-title">Train Models</div>
                  <div className="card-subtitle">Feature engineering → segmentation → churn / LTV / conversion models</div>
                </div>
              </div>

              <div className="grid-2" style={{ gap: 12, marginBottom: 16 }}>
                <div className="form-group" style={{ marginBottom: 0 }}>
                  <label className="form-label">Clusters</label>
                  <select
                    className="form-select"
                    value={nClusters}
                    onChange={e => setNClusters(+e.target.value)}
                    disabled={!dataReady}
                  >
                    {[3,4,5,6,7,8].map(n => <option key={n} value={n}>{n} clusters</option>)}
                  </select>
                </div>
                <div className="form-group" style={{ marginBottom: 0 }}>
                  <label className="form-label">Method</label>
                  <select
                    className="form-select"
                    value={method}
                    onChange={e => setMethod(e.target.value)}
                    disabled={!dataReady}
                  >
                    <option value="kmeans">KMeans</option>
                    <option value="gmm">Gaussian Mixture</option>
                  </select>
                </div>
              </div>

              {trainStatus && trainStatus.status !== 'idle' && (
                <div className={`alert ${trainStatus.status === 'failed' ? 'alert-error' : trainStatus.status === 'complete' ? 'alert-success' : 'alert-info'} mb-16`} style={{ fontSize: 12 }}>
                  {trainStatus.message}
                </div>
              )}

              <button
                className="btn btn-primary"
                style={{ width: '100%' }}
                onClick={handleTrain}
                disabled={!dataReady || trainRunning}
              >
                {trainRunning ? (
                  <><span className="spinner" style={{ width: 14, height: 14 }} /> Training...</>
                ) : (
                  modelsReady ? 'Re-train Models' : 'Train Models'
                )}
              </button>
            </div>

            {/* Pipeline summary */}
            {modelsReady && trainStatus?.predictions && (
              <div className="card">
                <div className="card-title mb-16">Training Results</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  <ResultRow label="Segmentation Silhouette" value={trainStatus.segmentation?.silhouette_score?.toFixed(4)} />
                  <ResultRow label="Churn Model Accuracy" value={trainStatus.predictions?.churn_accuracy ? `${(trainStatus.predictions.churn_accuracy * 100).toFixed(1)}%` : '--'} />
                  <ResultRow label="LTV Model R²" value={trainStatus.predictions?.ltv_r2?.toFixed(4)} />
                  <ResultRow label="Conversion Accuracy" value={trainStatus.predictions?.conversion_accuracy ? `${(trainStatus.predictions.conversion_accuracy * 100).toFixed(1)}%` : '--'} />
                </div>
              </div>
            )}
          </div>

          {/* Activity log */}
          <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
            <div className="card-header">
              <div className="card-title">Activity Log</div>
              <button
                className="btn btn-secondary btn-sm"
                onClick={() => setLog([])}
              >
                Clear
              </button>
            </div>
            <div
              ref={logRef}
              style={{
                flex: 1,
                minHeight: 420,
                maxHeight: 600,
                overflowY: 'auto',
                background: 'var(--bg-primary)',
                borderRadius: 'var(--radius-md)',
                padding: '12px 14px',
                fontFamily: 'var(--font-mono)',
                fontSize: 11,
                display: 'flex',
                flexDirection: 'column',
                gap: 4,
              }}
            >
              {log.length === 0 && (
                <div style={{ color: 'var(--text-muted)', fontSize: 11 }}>
                  No activity yet. Run a pipeline step to see logs.
                </div>
              )}
              {log.map((entry, i) => (
                <div key={i} style={{ display: 'flex', gap: 10 }}>
                  <span style={{ color: 'var(--text-muted)', flexShrink: 0 }}>{entry.ts}</span>
                  <span style={{
                    color: entry.type === 'error' ? 'var(--accent-red)'
                      : entry.type === 'success' ? 'var(--accent-green)'
                      : 'var(--text-secondary)',
                  }}>
                    {entry.msg}
                  </span>
                </div>
              ))}
              {(dataRunning || trainRunning) && (
                <div style={{ display: 'flex', gap: 8, alignItems: 'center', color: 'var(--accent-blue)' }}>
                  <span className="spinner" style={{ width: 10, height: 10 }} />
                  <span>{dataRunning ? dataStatus?.message : trainStatus?.message}</span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Dataset preview */}
        {dataInfo?.sample && (
          <div className="card mt-24">
            <div className="card-header">
              <div className="card-title">Dataset Preview</div>
              <span className="badge badge-muted">{dataInfo.rows?.toLocaleString()} total rows</span>
            </div>
            <div className="data-table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    {Object.keys(dataInfo.sample[0] ?? {}).slice(0, 10).map(col => (
                      <th key={col}>{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {dataInfo.sample.slice(0, 5).map((row, i) => (
                    <tr key={i}>
                      {Object.values(row).slice(0, 10).map((val, j) => (
                        <td key={j} className={typeof val === 'number' ? 'mono' : ''}>
                          {typeof val === 'number' ? (Number.isInteger(val) ? val : val.toFixed(3)) : String(val).slice(0, 20)}
                        </td>
                      ))}
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

function PipelineStepBadge({ num, done, running }) {
  const cls = done ? 'done' : running ? 'running' : 'pending'
  return (
    <div className={`pipeline-step-icon ${cls}`} style={{ width: 32, height: 32, fontSize: 13 }}>
      {done ? '✓' : running ? '⟳' : num}
    </div>
  )
}

function StatusCard({ label, status, value, detail }) {
  const colors = { ok: 'var(--accent-green)', warning: 'var(--accent-yellow)', pending: 'var(--text-muted)', error: 'var(--accent-red)' }
  return (
    <div className="card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 8 }}>
        <span style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>{label}</span>
        <span className="status-dot" style={{ background: colors[status], marginTop: 4 }} />
      </div>
      <div style={{ fontSize: 17, fontWeight: 700, color: colors[status], marginBottom: 4 }}>{value}</div>
      <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{detail}</div>
    </div>
  )
}

function ResultRow({ label, value }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, padding: '6px 0', borderBottom: '1px solid var(--border-muted)' }}>
      <span style={{ color: 'var(--text-secondary)' }}>{label}</span>
      <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-green)', fontWeight: 600 }}>{value ?? '--'}</span>
    </div>
  )
}
