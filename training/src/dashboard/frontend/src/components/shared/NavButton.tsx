import type { NavButtonProps } from '../../types'

type InteractiveNavButtonProps = NavButtonProps & { onClick: () => void }

export function NavButton({ active, icon: Icon, label, onClick }: InteractiveNavButtonProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={[
        'flex items-center gap-1.5 rounded px-2.5 py-1.5 text-xs font-medium transition-colors',
        active
          ? 'bg-background text-foreground shadow-sm'
          : 'text-muted-foreground hover:bg-secondary hover:text-foreground',
      ].join(' ')}
    >
      <Icon className="h-3.5 w-3.5" />
      {label}
    </button>
  )
}
