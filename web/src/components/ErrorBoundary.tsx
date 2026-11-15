import { Component, type ErrorInfo, type ReactNode } from 'react'
import { AlertTriangleIcon } from './ui/AppIcons'

interface Props {
  children: ReactNode
}

interface State {
  error: Error | null
}

// ไม่มี Error Boundary ที่ไหนในแอปมาก่อน — ถ้า component ไหน throw ตอน render (เช่น
// ข้อมูลจาก API ผิดรูปแบบไม่ตรงกับที่ type คาดไว้) แอปทั้งหน้าจะ white-screen ทันทีไม่มีทาง
// กู้คืนโดยไม่ reload เอง
export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('Unhandled render error:', error, info.componentStack)
  }

  render() {
    if (this.state.error) {
      return (
        <div className="flow-theme flex min-h-full flex-col items-center justify-center gap-3 p-8 text-center">
          <span className="flex h-12 w-12 items-center justify-center rounded-2xl border border-amber-200 bg-amber-50 text-amber-600 shadow-sm">
            <AlertTriangleIcon className="h-6 w-6" />
          </span>
          <h1 className="text-lg font-semibold text-slate-900">เกิดข้อผิดพลาดที่ไม่คาดคิด</h1>
          <p className="max-w-md text-sm text-slate-500">
            หน้านี้พังกลางทาง ลองโหลดหน้าใหม่อีกครั้ง — ถ้ายังเกิดซ้ำ กรุณาแจ้งทีมพัฒนา
          </p>
          <button
            onClick={() => window.location.reload()}
            className="rounded-xl bg-sky-600 px-4 py-2 text-sm font-medium text-white shadow-sm transition-all duration-150 hover:bg-sky-700 focus-visible:ring-2 focus-visible:ring-sky-500/50 active:scale-95"
          >
            โหลดหน้าใหม่
          </button>
        </div>
      )
    }
    return this.props.children
  }
}
