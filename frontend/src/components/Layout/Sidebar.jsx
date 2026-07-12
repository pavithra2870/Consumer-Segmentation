import { NavLink } from 'react-router-dom'

const NAV = [
  {
    section: 'Analytics',
    items: [
      { to: '/', label: 'Overview', icon: <GridIcon /> },
      { to: '/segments', label: 'Segmentation', icon: <SegmentIcon /> },
      { to: '/personas', label: 'Personas', icon: <PersonIcon /> },
      { to: '/predictions', label: 'Predictions', icon: <ChartIcon /> },
    ],
  },
  {
    section: 'Intelligence',
    items: [
      { to: '/explainability', label: 'Explainability', icon: <LightbulbIcon /> },
      { to: '/bias', label: 'Fairness Audit', icon: <ShieldIcon /> },
      { to: '/simulation', label: 'What-If Simulator', icon: <SimIcon /> },
    ],
  },
  {
    section: 'System',
    items: [
      { to: '/pipeline', label: 'Data Pipeline', icon: <PipelineIcon /> },
    ],
  },
]

export default function Sidebar({ health }) {
  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <div className="sidebar-logo-text">Consumer Intel</div>
        <div className="sidebar-logo-sub">Intelligence Platform</div>
      </div>

      <nav className="sidebar-nav">
        {NAV.map(({ section, items }) => (
          <div key={section}>
            <div className="sidebar-section-label">{section}</div>
            {items.map(({ to, label, icon }) => (
              <NavLink
                key={to}
                to={to}
                end={to === '/'}
                className={({ isActive }) => `sidebar-link${isActive ? ' active' : ''}`}
              >
                <span className="sidebar-icon">{icon}</span>
                {label}
              </NavLink>
            ))}
          </div>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div className="sidebar-status">
          <span className={`status-dot ${health?.ollama_available ? 'green' : 'red'}`} />
          <span>Ollama {health?.ollama_available ? 'connected' : 'offline'}</span>
        </div>
        <div className="sidebar-status" style={{ marginTop: 4 }}>
          <span className={`status-dot ${health?.data_ready ? 'green' : 'yellow'}`} />
          <span>Data {health?.data_ready ? 'ready' : 'pending'}</span>
        </div>
        <div className="sidebar-status" style={{ marginTop: 4 }}>
          <span className={`status-dot ${health?.models_ready ? 'green' : 'yellow'}`} />
          <span>Models {health?.models_ready ? 'trained' : 'pending'}</span>
        </div>
      </div>
    </aside>
  )
}

function GridIcon() {
  return (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
      <rect x="1" y="1" width="6" height="6" rx="1"/><rect x="9" y="1" width="6" height="6" rx="1"/>
      <rect x="1" y="9" width="6" height="6" rx="1"/><rect x="9" y="9" width="6" height="6" rx="1"/>
    </svg>
  )
}

function SegmentIcon() {
  return (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
      <circle cx="8" cy="8" r="6"/><path d="M8 2v6l4 2"/>
    </svg>
  )
}

function PersonIcon() {
  return (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
      <circle cx="8" cy="5" r="3"/><path d="M2 14c0-3.3 2.7-6 6-6s6 2.7 6 6"/>
    </svg>
  )
}

function ChartIcon() {
  return (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M2 12L6 7l3 3 5-6"/><path d="M1 14h14"/>
    </svg>
  )
}

function LightbulbIcon() {
  return (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M8 1a5 5 0 0 1 3 9H5a5 5 0 0 1 3-9z"/><path d="M6 13h4M7 15h2"/>
    </svg>
  )
}

function ShieldIcon() {
  return (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M8 1L2 4v5c0 3.3 2.7 5.7 6 6 3.3-.3 6-2.7 6-6V4L8 1z"/>
    </svg>
  )
}

function SimIcon() {
  return (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M3 8h10M8 3v10"/><circle cx="8" cy="8" r="2"/>
    </svg>
  )
}

function PipelineIcon() {
  return (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
      <rect x="1" y="4" width="4" height="3" rx="1"/><rect x="6" y="4" width="4" height="3" rx="1"/>
      <rect x="11" y="4" width="4" height="3" rx="1"/><path d="M3 7v2M8 7v2M13 7v2"/>
      <path d="M5 5.5h1M10 5.5h1"/>
    </svg>
  )
}
