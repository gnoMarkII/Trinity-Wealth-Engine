import { useEffect, useState } from 'react'

export function usePageVisibility(): boolean {
  const [isVisible, setIsVisible] = useState(() => !document.hidden)

  useEffect(() => {
    const updateVisibility = () => setIsVisible(!document.hidden)
    document.addEventListener('visibilitychange', updateVisibility)
    return () => document.removeEventListener('visibilitychange', updateVisibility)
  }, [])

  return isVisible
}
