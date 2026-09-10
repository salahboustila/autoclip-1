import { useEffect, useState } from 'react'
import { NavLink, Outlet } from 'react-router-dom'

import { api, type SystemStatus } from './api'

/**
 * Application shell.
 *
 * A masthead rather than a sidebar: this is a four-screen tool, and a
 * persistent nav rail would spend a fifth of the width restating that.
 */
export function App() {
  const [system, setSystem] = useState<SystemStatus | null>(null)

  useEffect(() => {
    api.system().then(setSystem).catch(() => setSystem(null))
  }, [])

  return (
    <div className="min-h-screen bg-ink-900">
      <header className="sticky top-0 z-20 border-b border-ink-800 bg-ink-900/85 backdrop-blur-md">
        <div className="mx-auto flex max-w-[1600px] items-center gap-8 px-6 py-3.5 lg:px-10">
          <NavLink to="/" className="flex items-center gap-2.5">
            <span
              aria-hidden
              className="grid size-7 place-items-center rounded-lg bg-mint-500 text-[0.9rem] font-bold text-white"
            >
              ✦
            </span>
            <span className="headline text-lg text-ink-100">AutoClip</span>
          </NavLink>

          <nav className="flex items-center gap-1">
            <TopLink to="/" end>
              New
            </TopLink>
            <TopLink to="/settings">Settings</TopLink>
          </nav>

          <div className="ml-auto flex items-center gap-5">
            {system && <SystemBadge system={system} />}
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[1600px] px-6 pb-24 lg:px-10">
        <Outlet />
      </main>
    </div>
  )
}

function TopLink({
  to,
  end,
  children,
}: {
  to: string
  end?: boolean
  children: React.ReactNode
}) {
  return (
    <NavLink
      to={to}
      end={end}
      className={({ isActive }) =>
        [
          'rounded-full px-3.5 py-1.5 text-sm font-medium transition-colors duration-200',
          isActive
            ? 'bg-ink-850 text-ink-100'
            : 'text-ink-400 hover:bg-ink-850 hover:text-ink-200',
        ].join(' ')
      }
    >
      {children}
    </NavLink>
  )
}

/**
 * Compact machine status.
 *
 * Surfaced permanently rather than hidden in settings because the two things it
 * reports — whether ffmpeg can render, and whether the GPU is being used — are
 * the two that change how long everything takes.
 */
function SystemBadge({ system }: { system: SystemStatus }) {
  const accel = system.accel.toUpperCase()
  return (
    <div className="hidden items-center gap-2.5 text-xs md:flex">
      <span className="numeric rounded-full bg-ink-850 px-2.5 py-1 text-ink-400">
        {accel}
        {system.gpu_name && accel === 'CUDA' && (
          <span className="text-ink-600"> · {system.gpu_name.replace('NVIDIA GeForce ', '')}</span>
        )}
      </span>
      <span
        className={[
          'inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 font-medium',
          system.ready ? 'bg-mint-300/35 text-signal-good' : 'bg-signal-bad/12 text-signal-bad',
        ].join(' ')}
        title={system.ready ? 'All required components present' : 'Run autoclip doctor'}
      >
        <span
          aria-hidden
          className={[
            'size-1.5 rounded-full',
            system.ready ? 'bg-signal-good' : 'bg-signal-bad',
          ].join(' ')}
        />
        {system.ready ? 'ready' : 'not ready'}
      </span>
    </div>
  )
}
