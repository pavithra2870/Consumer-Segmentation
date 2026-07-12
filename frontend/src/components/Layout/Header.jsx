export default function Header({ title, subtitle, actions }) {
  return (
    <header className="page-header">
      <div>
        <div className="page-header-title">{title}</div>
        {subtitle && <div className="page-header-subtitle">{subtitle}</div>}
      </div>
      {actions && <div className="header-actions">{actions}</div>}
    </header>
  )
}
