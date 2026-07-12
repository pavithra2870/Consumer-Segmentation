export default function LoadingSpinner({ message = 'Loading...' }) {
  return (
    <div className="loading-state">
      <div className="spinner" />
      <span>{message}</span>
    </div>
  )
}

export function EmptyState({ icon = '○', title, message, action }) {
  return (
    <div className="empty-state">
      <div className="empty-state-icon">{icon}</div>
      {title && <div style={{ fontWeight: 600, color: 'var(--text-secondary)', fontSize: 14 }}>{title}</div>}
      {message && <div>{message}</div>}
      {action}
    </div>
  )
}
