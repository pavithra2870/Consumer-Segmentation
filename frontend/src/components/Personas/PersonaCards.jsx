import { useState } from 'react'
import Header from '../Layout/Header'
import LoadingSpinner, { EmptyState } from '../Common/LoadingSpinner'
import { useApi } from '../../hooks/useApi'
import { api } from '../../services/api'

const SEGMENT_COLORS = ['#388bfd','#f85149','#3fb950','#d29922','#8957e5','#db6d28','#39c5cf','#e36d9b']

const AVATAR_BG = ['#1f3a5f','#3d1f1f','#1f3d21','#3d351f','#2e1f3d','#3d2a1f','#1f3337','#3d1f2f']

function initials(name) {
  return name?.split(' ').map(n => n[0]).join('').slice(0, 2).toUpperCase() || '??'
}

export default function PersonaCards() {
  const { data, loading, error } = useApi(api.getPersonas)
  const [expanded, setExpanded] = useState(null)

  const personas = data?.personas ?? []

  return (
    <>
      <Header
        title="Consumer Personas"
        subtitle="LLM-generated persona profiles per customer segment"
      />
      <div className="page-body">
        {loading && <LoadingSpinner message="Generating personas..." />}
        {error && <div className="alert alert-error">{error}</div>}
        {!loading && personas.length === 0 && (
          <EmptyState
            icon="◻"
            title="No personas available"
            message="Run the training pipeline to generate segment personas."
          />
        )}
        <div className="grid-auto">
          {personas.map(p => (
            <PersonaCard
              key={p.segment_id}
              persona={p}
              color={SEGMENT_COLORS[p.segment_id % SEGMENT_COLORS.length]}
              avatarBg={AVATAR_BG[p.segment_id % AVATAR_BG.length]}
              expanded={expanded === p.segment_id}
              onToggle={() => setExpanded(expanded === p.segment_id ? null : p.segment_id)}
            />
          ))}
        </div>
      </div>
    </>
  )
}

function PersonaCard({ persona, color, avatarBg, expanded, onToggle }) {
  const churnBadge = persona.behaviors.churn_risk_pct > 50 ? 'badge-red'
    : persona.behaviors.churn_risk_pct > 30 ? 'badge-yellow' : 'badge-green'

  return (
    <div className="persona-card">
      <div className="persona-card-header">
        <div className="flex items-center gap-12 mb-16">
          <div
            className="persona-avatar"
            style={{ background: avatarBg, color, border: `1.5px solid ${color}` }}
          >
            {initials(persona.name)}
          </div>
          <div>
            <div className="persona-name">{persona.name}</div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 4 }}>
              <span className="badge badge-muted" style={{ fontSize: 10 }}>
                Segment {persona.segment_id}
              </span>
              <span className={`badge ${churnBadge}`} style={{ fontSize: 10 }}>
                {persona.behaviors.churn_risk_pct}% churn
              </span>
            </div>
          </div>
        </div>
        <div className="persona-tagline">"{persona.tagline}"</div>
      </div>

      <div className="persona-card-body">
        <div className="persona-metrics">
          <div className="persona-metric">
            <div className="persona-metric-value" style={{ color }}>
              ${Number(persona.behaviors.avg_ltv).toLocaleString(undefined, { maximumFractionDigits: 0 })}
            </div>
            <div className="persona-metric-label">Avg LTV</div>
          </div>
          <div className="persona-metric">
            <div className="persona-metric-value">{persona.behaviors.avg_nps}</div>
            <div className="persona-metric-label">NPS Score</div>
          </div>
          <div className="persona-metric">
            <div className="persona-metric-value">{persona.demographics.segment_size?.toLocaleString()}</div>
            <div className="persona-metric-label">Customers</div>
          </div>
        </div>

        <div style={{ borderTop: '1px solid var(--border)', marginTop: 16, paddingTop: 14 }}>
          <div className="persona-section-title">Motivations</div>
          <ul className="persona-list">
            {persona.motivations?.map((m, i) => <li key={i}>{m}</li>)}
          </ul>
        </div>

        {expanded && (
          <>
            <div style={{ marginTop: 14 }}>
              <div className="persona-section-title">Pain Points</div>
              <ul className="persona-list">
                {persona.pain_points?.map((p, i) => <li key={i}>{p}</li>)}
              </ul>
            </div>

            <div style={{ marginTop: 14 }}>
              <div className="persona-section-title">Narrative</div>
              <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                {persona.narrative}
              </div>
            </div>

            <div style={{ marginTop: 14 }}>
              <div className="persona-section-title">Demographics</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
                {Object.entries(persona.demographics || {}).map(([k, v]) => (
                  <div key={k} style={{ fontSize: 11 }}>
                    <span style={{ color: 'var(--text-muted)' }}>{k.replace(/_/g, ' ')}: </span>
                    <span style={{ color: 'var(--text-primary)' }}>{String(v)}</span>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
      </div>

      <div className="persona-card-footer">
        <div className="persona-section-title" style={{ marginBottom: 8 }}>Recommended Actions</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
          {persona.recommended_actions?.map((a, i) => (
            <div key={i} style={{ display: 'flex', gap: 6, alignItems: 'flex-start' }}>
              <span style={{ color, fontWeight: 700, flexShrink: 0, fontSize: 11 }}>{i + 1}.</span>
              <span style={{ fontSize: 11, color: 'var(--text-secondary)', lineHeight: 1.4 }}>{a}</span>
            </div>
          ))}
        </div>
        <button
          className="btn btn-secondary btn-sm"
          style={{ marginTop: 12, width: '100%' }}
          onClick={onToggle}
        >
          {expanded ? 'Show less' : 'Show full profile'}
        </button>
      </div>
    </div>
  )
}
