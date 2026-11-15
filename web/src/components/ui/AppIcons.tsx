import type { SVGProps } from 'react'

export type AppIconProps = SVGProps<SVGSVGElement>

const baseProps = {
  width: 16,
  height: 16,
  fill: 'none',
  viewBox: '0 0 24 24',
  stroke: 'currentColor',
  strokeWidth: 1.75,
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
  'aria-hidden': true as const,
}

export function SparklesIcon({ className = 'w-4 h-4', ...rest }: AppIconProps) {
  return (
    <svg {...baseProps} className={className} {...rest}>
      <path d="M12 3l1.75 4.25L18 9l-4.25 1.75L12 15l-1.75-4.25L6 9l4.25-1.75L12 3z" />
      <path d="M5 16l.8 2L8 18.8 5.8 19.6 5 21.5l-.8-1.9L2 18.8l2.2-.8L5 16z" />
      <path d="M19 14l.65 1.6L21.25 16.25l-1.6.65L19 18.5l-.65-1.6-1.6-.65 1.6-.65L19 14z" />
    </svg>
  )
}

export function MicStageIcon({ className = 'w-4 h-4', ...rest }: AppIconProps) {
  return (
    <svg {...baseProps} className={className} {...rest}>
      <rect x="9" y="2.75" width="6" height="10.5" rx="3" />
      <path d="M5.5 10.5a6.5 6.5 0 0013 0" />
      <path d="M12 17v4.25" />
      <path d="M8.5 21.25h7" />
    </svg>
  )
}

export function NewspaperIcon({ className = 'w-4 h-4', ...rest }: AppIconProps) {
  return (
    <svg {...baseProps} className={className} {...rest}>
      <path d="M4 5.5A1.5 1.5 0 015.5 4h11A1.5 1.5 0 0118 5.5V18a2 2 0 01-2 2H6a2 2 0 01-2-2V5.5z" />
      <path d="M18 8h1.5A1.5 1.5 0 0121 9.5V18a2 2 0 01-2 2h-1" />
      <path d="M7.5 8h7" />
      <path d="M7.5 12h7" />
      <path d="M7.5 16h4.5" />
    </svg>
  )
}

export function BriefcasePortfolioIcon({ className = 'w-4 h-4', ...rest }: AppIconProps) {
  return (
    <svg {...baseProps} className={className} {...rest}>
      <rect x="3" y="7" width="18" height="13" rx="2.5" />
      <path d="M8.5 7V5a1.5 1.5 0 011.5-1.5h4A1.5 1.5 0 0115.5 5v2" />
      <path d="M3 12.5h18" />
    </svg>
  )
}

export function BoltStrategyIcon({ className = 'w-4 h-4', ...rest }: AppIconProps) {
  return (
    <svg {...baseProps} className={className} {...rest}>
      <path d="M13 2.75L4.5 13.5h6.75L10.5 21.25 19.5 10.5h-6.75L13 2.75z" />
    </svg>
  )
}

export function GlobeMacroIcon({ className = 'w-4 h-4', ...rest }: AppIconProps) {
  return (
    <svg {...baseProps} className={className} {...rest}>
      <circle cx="12" cy="12" r="9" />
      <path d="M3.6 9h16.8" />
      <path d="M3.6 15h16.8" />
      <path d="M12 3a13.5 13.5 0 010 18 13.5 13.5 0 010-18z" />
    </svg>
  )
}

export function ChartTrendIcon({ className = 'w-4 h-4', ...rest }: AppIconProps) {
  return (
    <svg {...baseProps} className={className} {...rest}>
      <path d="M3.5 19.5h17" />
      <path d="M4.5 15l4.75-4.75 3.5 3.25L19.5 6.5" />
      <path d="M15.5 6.5h4v4" />
    </svg>
  )
}

export function ShieldCheckIcon({ className = 'w-4 h-4', ...rest }: AppIconProps) {
  return (
    <svg {...baseProps} className={className} {...rest}>
      <path d="M12 3l7 3v5.5c0 4.6-2.95 8.15-7 9.5-4.05-1.35-7-4.9-7-9.5V6l7-3z" />
      <path d="M9.25 12.25l1.9 1.9 3.6-3.9" />
    </svg>
  )
}

export function TempleReviewIcon({ className = 'w-4 h-4', ...rest }: AppIconProps) {
  return (
    <svg {...baseProps} className={className} {...rest}>
      <path d="M3 9l9-5 9 5" />
      <path d="M4.5 9.5v9.5" />
      <path d="M9.5 9.5v9.5" />
      <path d="M14.5 9.5v9.5" />
      <path d="M19.5 9.5v9.5" />
      <path d="M3 19.5h18" />
    </svg>
  )
}

export function CompassIcon({ className = 'w-4 h-4', ...rest }: AppIconProps) {
  return (
    <svg {...baseProps} className={className} {...rest}>
      <circle cx="12" cy="12" r="9" />
      <polygon points="15.5,8.5 13.5,13.5 8.5,15.5 10.5,10.5" />
    </svg>
  )
}

export function RefreshIcon({ className = 'w-4 h-4', ...rest }: AppIconProps) {
  return (
    <svg {...baseProps} className={className} {...rest}>
      <path d="M20 11a8 8 0 00-14.6-4.4L3.5 8.5" />
      <path d="M3.5 4.5v4h4" />
      <path d="M4 13a8 8 0 0014.6 4.4l1.9-1.9" />
      <path d="M20.5 19.5v-4h-4" />
    </svg>
  )
}

export function BookOpenIcon({ className = 'w-4 h-4', ...rest }: AppIconProps) {
  return (
    <svg {...baseProps} className={className} {...rest}>
      <path d="M2.5 5.5A2.5 2.5 0 015 3h5.5A1.5 1.5 0 0112 4.5v15a2 2 0 00-2-2H5a2.5 2.5 0 00-2.5 2.5V5.5z" />
      <path d="M21.5 5.5A2.5 2.5 0 0019 3h-5.5A1.5 1.5 0 0012 4.5v15a2 2 0 012-2h5a2.5 2.5 0 012.5 2.5V5.5z" />
    </svg>
  )
}

export function SlidersIcon({ className = 'w-4 h-4', ...rest }: AppIconProps) {
  return (
    <svg {...baseProps} className={className} {...rest}>
      <path d="M4 6h16" />
      <path d="M4 12h16" />
      <path d="M4 18h16" />
      <circle cx="9" cy="6" r="2" fill="currentColor" stroke="none" />
      <circle cx="15" cy="12" r="2" fill="currentColor" stroke="none" />
      <circle cx="10" cy="18" r="2" fill="currentColor" stroke="none" />
    </svg>
  )
}

export function EditPenIcon({ className = 'w-4 h-4', ...rest }: AppIconProps) {
  return (
    <svg {...baseProps} className={className} {...rest}>
      <path d="M16.5 3.5a2.12 2.12 0 013 3L7 19l-4 1 1-4L16.5 3.5z" />
    </svg>
  )
}

export function TrashIcon({ className = 'w-4 h-4', ...rest }: AppIconProps) {
  return (
    <svg {...baseProps} className={className} {...rest}>
      <path d="M3 6h18" />
      <path d="M8 6V4a1 1 0 011-1h6a1 1 0 011 1v2" />
      <path d="M6 6l1 13a2 2 0 002 2h6a2 2 0 002-2l1-13" />
    </svg>
  )
}

export function FolderIcon({ className = 'w-4 h-4', ...rest }: AppIconProps) {
  return (
    <svg {...baseProps} className={className} {...rest}>
      <path d="M3 6.5A2 2 0 015 4.5h4.2l1.8 2H19a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2v-10z" />
    </svg>
  )
}

export function MoreHorizontalIcon({ className = 'w-4 h-4', ...rest }: AppIconProps) {
  return (
    <svg {...baseProps} className={className} {...rest}>
      <circle cx="6" cy="12" r="1.25" fill="currentColor" stroke="none" />
      <circle cx="12" cy="12" r="1.25" fill="currentColor" stroke="none" />
      <circle cx="18" cy="12" r="1.25" fill="currentColor" stroke="none" />
    </svg>
  )
}

export function SidebarCollapseIcon({ className = 'w-4 h-4', ...rest }: AppIconProps) {
  return (
    <svg {...baseProps} className={className} {...rest}>
      <rect x="3" y="4" width="18" height="16" rx="2.5" />
      <path d="M9 4v16" />
      <path d="M15.5 9.5L13 12l2.5 2.5" />
    </svg>
  )
}

export function SidebarExpandIcon({ className = 'w-4 h-4', ...rest }: AppIconProps) {
  return (
    <svg {...baseProps} className={className} {...rest}>
      <rect x="3" y="4" width="18" height="16" rx="2.5" />
      <path d="M9 4v16" />
      <path d="M13.5 9.5L16 12l-2.5 2.5" />
    </svg>
  )
}

export function LogoutIcon({ className = 'w-4 h-4', ...rest }: AppIconProps) {
  return (
    <svg {...baseProps} className={className} {...rest}>
      <path d="M9 21H5a2 2 0 01-2-2V5a2 2 0 012-2h4" />
      <path d="M16 17l5-5-5-5" />
      <path d="M21 12H9" />
    </svg>
  )
}

export function AlertTriangleIcon({ className = 'w-4 h-4', ...rest }: AppIconProps) {
  return (
    <svg {...baseProps} className={className} {...rest}>
      <path d="M10.3 3.86L2.4 17.5A1.75 1.75 0 003.9 20h16.2a1.75 1.75 0 001.5-2.5L13.7 3.86a1.95 1.95 0 00-3.4 0z" />
      <path d="M12 9v4.5" />
      <circle cx="12" cy="16.5" r="0.75" fill="currentColor" stroke="none" />
    </svg>
  )
}

export function InfoCircleIcon({ className = 'w-4 h-4', ...rest }: AppIconProps) {
  return (
    <svg {...baseProps} className={className} {...rest}>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 11v5" />
      <circle cx="12" cy="8" r="0.75" fill="currentColor" stroke="none" />
    </svg>
  )
}

export function VolumeOnIcon({ className = 'w-4 h-4', ...rest }: AppIconProps) {
  return (
    <svg {...baseProps} className={className} {...rest}>
      <path d="M11 5L6 9H3v6h3l5 4V5z" />
      <path d="M15.5 8.5a5 5 0 010 7" />
      <path d="M18.5 5.5a9 9 0 010 13" />
    </svg>
  )
}

export function VolumeOffIcon({ className = 'w-4 h-4', ...rest }: AppIconProps) {
  return (
    <svg {...baseProps} className={className} {...rest}>
      <path d="M11 5L6 9H3v6h3l5 4V5z" />
      <path d="M16 9l5 6" />
      <path d="M21 9l-5 6" />
    </svg>
  )
}

export function StarIcon({ className = 'w-4 h-4', ...rest }: AppIconProps) {
  return (
    <svg {...baseProps} className={className} {...rest}>
      <polygon points="12 2.75 14.9 8.65 21.4 9.6 16.7 14.15 17.8 20.6 12 17.55 6.2 20.6 7.3 14.15 2.6 9.6 9.1 8.65 12 2.75" />
    </svg>
  )
}

export function SunIcon({ className = 'w-4 h-4', ...rest }: AppIconProps) {
  return (
    <svg {...baseProps} className={className} {...rest}>
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" />
    </svg>
  )
}

export function CalendarIcon({ className = 'w-4 h-4', ...rest }: AppIconProps) {
  return (
    <svg {...baseProps} className={className} {...rest}>
      <rect x="3" y="4.5" width="18" height="16" rx="2.5" />
      <path d="M8 2.5v4M16 2.5v4M3 9.5h18" />
    </svg>
  )
}

export function CheckCircleIcon({ className = 'w-4 h-4', ...rest }: AppIconProps) {
  return (
    <svg {...baseProps} className={className} {...rest}>
      <circle cx="12" cy="12" r="9" />
      <path d="M8.5 12.25l2.3 2.3 4.7-4.8" />
    </svg>
  )
}

export function ScrollHistoryIcon({ className = 'w-4 h-4', ...rest }: AppIconProps) {
  return (
    <svg {...baseProps} className={className} {...rest}>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 7v5l3 2" />
    </svg>
  )
}

export function CpuIcon({ className = 'w-4 h-4', ...rest }: AppIconProps) {
  return (
    <svg {...baseProps} className={className} {...rest}>
      <rect x="6" y="6" width="12" height="12" rx="2" />
      <path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4" />
    </svg>
  )
}

export function ChevronUpIcon({ className = 'w-4 h-4', ...rest }: AppIconProps) {
  return (
    <svg {...baseProps} className={className} {...rest}>
      <path d="M6 14l6-6 6 6" />
    </svg>
  )
}

export function ChevronDownIcon({ className = 'w-4 h-4', ...rest }: AppIconProps) {
  return (
    <svg {...baseProps} className={className} {...rest}>
      <path d="M6 10l6 6 6-6" />
    </svg>
  )
}

export function SearchIcon({ className = 'w-4 h-4', ...rest }: AppIconProps) {
  return (
    <svg {...baseProps} className={className} {...rest}>
      <circle cx="11" cy="11" r="7" />
      <path d="M20 20l-3.5-3.5" />
    </svg>
  )
}

export function CloseIcon({ className = 'w-4 h-4', ...rest }: AppIconProps) {
  return (
    <svg {...baseProps} className={className} {...rest}>
      <path d="M18 6L6 18" />
      <path d="M6 6l12 12" />
    </svg>
  )
}

export function KanbanBoardIcon({ className = 'w-4 h-4', ...rest }: AppIconProps) {
  return (
    <svg {...baseProps} className={className} {...rest}>
      <rect x="3.5" y="4" width="17" height="16" rx="2.5" />
      <path d="M9 4v16" />
      <path d="M15 4v11" />
    </svg>
  )
}

export function SwarmNodesIcon({ className = 'w-4 h-4', ...rest }: AppIconProps) {
  return (
    <svg {...baseProps} className={className} {...rest}>
      <circle cx="12" cy="5.5" r="2.25" />
      <circle cx="6" cy="16.5" r="2.25" />
      <circle cx="18" cy="16.5" r="2.25" />
      <path d="M10.75 7.5l-3.5 7" />
      <path d="M13.25 7.5l3.5 7" />
      <path d="M8.25 16.5h7.5" />
    </svg>
  )
}

export function ExpandIcon({ className = 'w-4 h-4', ...rest }: AppIconProps) {
  return (
    <svg {...baseProps} className={className} {...rest}>
      <path d="M15 3.5h5.5V9" />
      <path d="M9 20.5H3.5V15" />
      <path d="M20.5 3.5L14 10" />
      <path d="M3.5 20.5L10 14" />
    </svg>
  )
}

export function CompactLayoutIcon({ className = 'w-4 h-4', ...rest }: AppIconProps) {
  return (
    <svg {...baseProps} className={className} {...rest}>
      <rect x="4" y="4" width="16" height="16" rx="2" />
      <path d="M9.5 4v16" />
    </svg>
  )
}

export function TargetIcon({ className = 'w-4 h-4', ...rest }: AppIconProps) {
  return (
    <svg {...baseProps} className={className} {...rest}>
      <circle cx="12" cy="12" r="8.5" />
      <circle cx="12" cy="12" r="4.5" />
      <circle cx="12" cy="12" r="1.25" fill="currentColor" stroke="none" />
    </svg>
  )
}

export function CheckSmallIcon({ className = 'w-4 h-4', ...rest }: AppIconProps) {
  return (
    <svg {...baseProps} className={className} {...rest}>
      <path d="M5 12.5l4.25 4.25L19 7" />
    </svg>
  )
}

export function PlayIcon({ className = 'w-4 h-4', ...rest }: AppIconProps) {
  return (
    <svg {...baseProps} className={className} {...rest}>
      <polygon points="6.5 4.5 19.5 12 6.5 19.5 6.5 4.5" />
    </svg>
  )
}

export function PauseIcon({ className = 'w-4 h-4', ...rest }: AppIconProps) {
  return (
    <svg {...baseProps} className={className} {...rest}>
      <rect x="6.5" y="4.5" width="3.5" height="15" rx="1" />
      <rect x="14" y="4.5" width="3.5" height="15" rx="1" />
    </svg>
  )
}

export function SkipForwardIcon({ className = 'w-4 h-4', ...rest }: AppIconProps) {
  return (
    <svg {...baseProps} className={className} {...rest}>
      <polygon points="5 5.5 14 12 5 18.5 5 5.5" />
      <path d="M18.5 5.5v13" />
    </svg>
  )
}

export function ScaleIcon({ className = 'w-4 h-4', ...rest }: AppIconProps) {
  return (
    <svg {...baseProps} className={className} {...rest}>
      <path d="M12 3v18" />
      <path d="M5 7h14" />
      <path d="M5 7l-2.5 6h5L5 7z" />
      <path d="M19 7l-2.5 6h5L19 7z" />
      <path d="M8.5 21h7" />
    </svg>
  )
}

export function BulbIcon({ className = 'w-4 h-4', ...rest }: AppIconProps) {
  return (
    <svg {...baseProps} className={className} {...rest}>
      <path d="M9 18h6" />
      <path d="M10 21h4" />
      <path d="M12 3a6 6 0 00-4 10.45c.65.58 1 1.4 1 2.3V16h6v-.25c0-.9.35-1.72 1-2.3A6 6 0 0012 3z" />
    </svg>
  )
}

export function LinkChainIcon({ className = 'w-4 h-4', ...rest }: AppIconProps) {
  return (
    <svg {...baseProps} className={className} {...rest}>
      <path d="M10 13.5a4.5 4.5 0 006.36 0l2.12-2.12a4.5 4.5 0 00-6.36-6.36l-1.2 1.2" />
      <path d="M14 10.5a4.5 4.5 0 00-6.36 0l-2.12 2.12a4.5 4.5 0 006.36 6.36l1.2-1.2" />
    </svg>
  )
}

export function RadarIcon({ className = 'w-4 h-4', ...rest }: AppIconProps) {
  return (
    <svg {...baseProps} className={className} {...rest}>
      <circle cx="12" cy="12" r="9" />
      <circle cx="12" cy="12" r="4.5" />
      <path d="M12 12l5.5-5.5" />
    </svg>
  )
}

export function ExternalLinkIcon({ className = 'w-4 h-4', ...rest }: AppIconProps) {
  return (
    <svg {...baseProps} className={className} {...rest}>
      <path d="M14 4h6v6" />
      <path d="M10 14L20 4" />
      <path d="M19 13v5a2 2 0 01-2 2H6a2 2 0 01-2-2V7a2 2 0 012-2h5" />
    </svg>
  )
}

export function MenuIcon({ className = 'w-4 h-4', ...rest }: AppIconProps) {
  return (
    <svg {...baseProps} className={className} {...rest}>
      <path d="M4 6h16" />
      <path d="M4 12h16" />
      <path d="M4 18h16" />
    </svg>
  )
}
