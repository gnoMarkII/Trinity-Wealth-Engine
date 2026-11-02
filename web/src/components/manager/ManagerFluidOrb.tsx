interface ManagerFluidOrbProps {
  label?: string
  size?: 'avatar' | 'sm' | 'md' | 'lg'
  mode?: 'orb' | 'avatar'
}

export default function ManagerFluidOrb({
  label = 'กำลังประมวลผลมุมมองสำหรับคุณ...',
  size = 'md',
  mode = 'orb',
}: ManagerFluidOrbProps) {
  if (mode === 'avatar' || size === 'avatar') {
    return (
      <div className="relative flex h-7 w-7 shrink-0 items-center justify-center">
        <div className="absolute h-7 w-7 rounded-full bg-gradient-to-tr from-flow-sky via-flow-cyan to-flow-blue opacity-40 blur-sm" />
        <div className="relative flex h-7 w-7 items-center justify-center rounded-full bg-gradient-to-tr from-flow-sky via-flow-cyan to-flow-blue p-0.5 shadow-[0_0_14px_rgba(6,182,212,0.35)] animate-manager-orb">
          <div className="flex h-full w-full items-center justify-center rounded-full bg-white/40 text-[10px] font-bold text-sky-950 backdrop-blur-sm">
            ✨
          </div>
        </div>
      </div>
    )
  }

  const orbDimensions = {
    avatar: 'w-7 h-7',
    sm: 'w-6 h-6',
    md: 'w-10 h-10',
    lg: 'w-14 h-14',
  }[size]

  return (
    <div className="flex flex-col items-center justify-center gap-3 py-4 text-center">
      {/* 60fps Breathing Radiant Orb */}
      <div className="relative flex items-center justify-center">
        {/* Outer ambient glow */}
        <div
          className={`absolute rounded-full bg-gradient-to-tr from-flow-sky via-flow-cyan to-flow-blue opacity-40 blur-xl ${
            size === 'lg' ? 'h-24 w-24' : 'h-16 w-16'
          }`}
        />

        {/* Morphing energy ring */}
        <div
          className={`relative rounded-full bg-gradient-to-tr from-flow-sky via-flow-cyan to-flow-blue shadow-[0_0_24px_rgba(6,182,212,0.35)] ${orbDimensions} animate-manager-orb flex items-center justify-center p-0.5`}
        >
          <div className="h-full w-full rounded-full bg-white/30 backdrop-blur-sm" />
        </div>
      </div>

      {/* Gentle thinking text and wave dots */}
      <div className="flex items-center gap-2 text-xs font-medium text-muted">
        <span>{label}</span>
        <span className="flex items-center gap-1">
          <span
            className="h-1.5 w-1.5 rounded-full bg-flow-cyan animate-wave-dot"
            style={{ animationDelay: '0ms' }}
          />
          <span
            className="h-1.5 w-1.5 rounded-full bg-flow-sky animate-wave-dot"
            style={{ animationDelay: '200ms' }}
          />
          <span
            className="h-1.5 w-1.5 rounded-full bg-flow-blue animate-wave-dot"
            style={{ animationDelay: '400ms' }}
          />
        </span>
      </div>
    </div>
  )
}
