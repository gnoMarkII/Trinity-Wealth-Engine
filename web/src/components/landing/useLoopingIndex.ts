import { useEffect, useState } from 'react'

export function useLoopingIndex(itemCount: number, intervalMs: number): number {
  const [activeIndex, setActiveIndex] = useState(0)

  useEffect(() => {
    if (itemCount <= 1) return

    let timerId: ReturnType<typeof setInterval> | null = null
    const start = () => {
      if (timerId || document.hidden) return
      timerId = setInterval(() => {
        setActiveIndex((currentIndex) => (currentIndex + 1) % itemCount)
      }, intervalMs)
    }
    const stop = () => {
      if (timerId) clearInterval(timerId)
      timerId = null
    }
    const handleVisibilityChange = () => {
      if (document.hidden) stop()
      else start()
    }

    start()
    document.addEventListener('visibilitychange', handleVisibilityChange)
    return () => {
      stop()
      document.removeEventListener('visibilitychange', handleVisibilityChange)
    }
  }, [intervalMs, itemCount])

  return activeIndex
}
