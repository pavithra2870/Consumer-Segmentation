import { Routes, Route } from 'react-router-dom'
import { useState, useEffect } from 'react'
import Sidebar from './components/Layout/Sidebar'
import Overview from './components/Overview/Overview'
import SegmentExplorer from './components/Segmentation/SegmentExplorer'
import PersonaCards from './components/Personas/PersonaCards'
import PredictionsPanel from './components/Predictions/PredictionsPanel'
import ExplainPanel from './components/Explainability/ExplainPanel'
import BiasReport from './components/Bias/BiasReport'
import WhatIfTool from './components/Simulation/WhatIfTool'
import Pipeline from './components/Pipeline/Pipeline'
import Benchmarks from './components/Benchmarks/Benchmarks'
import { api } from './services/api'
import { useDataset } from './hooks/DatasetContext'

export default function App() {
  const [health, setHealth] = useState(null)
  const { dataset } = useDataset()

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const h = await api.health()
        setHealth(h)
      } catch {
        setHealth({ ollama_available: false, data_ready: false, models_ready: false })
      }
    }
    fetchHealth()
    const interval = setInterval(fetchHealth, 15000)
    return () => clearInterval(interval)
  }, [dataset])

  return (
    <div className="app-layout">
      <Sidebar health={health} />
      <main className="main-content" key={dataset}>
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/segments" element={<SegmentExplorer />} />
          <Route path="/personas" element={<PersonaCards />} />
          <Route path="/predictions" element={<PredictionsPanel />} />
          <Route path="/explainability" element={<ExplainPanel />} />
          <Route path="/bias" element={<BiasReport />} />
          <Route path="/simulation" element={<WhatIfTool />} />
          <Route path="/benchmarks" element={<Benchmarks />} />
          <Route path="/pipeline" element={<Pipeline />} />
        </Routes>
      </main>
    </div>
  )
}
