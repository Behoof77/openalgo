import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import { ErrorBoundary } from '@/components/ErrorBoundary'
import { clearChunkReloadFlag } from '@/utils/chunkReload'
import { installGlobalErrorReporter } from '@/utils/errorReporter'
import App from './App.tsx'

installGlobalErrorReporter()

// Cross-origin fetch interceptor: prefixes VITE_API_URL to bare "/path" URLs
// so ~80 fetch() call-sites work when Vercel frontend ≠ Oracle VM backend.
// Only fires on string URLs starting with "/"; full URLs pass through untouched.
// When rewriting to cross-origin, also sets credentials: 'include' so session
// cookies are sent — required by backend session auth (get_username_from_session).
const API_BASE_URL = import.meta.env.VITE_API_URL || ''
if (API_BASE_URL) {
  const _origFetch = window.fetch.bind(window)
  window.fetch = (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    if (typeof input === 'string' && input.startsWith('/') && !input.startsWith('//')) {
      input = `${API_BASE_URL}${input}`
      // Ensure cookies are sent cross-origin for session-based auth
      init = { ...init, credentials: 'include' }
    }
    return _origFetch(input, init)
  }
}

// We mounted successfully — the bundle is fresh. Clear the
// stale-chunk reload-attempt flag so a *future* stale-chunk navigation
// later in this tab session can auto-recover too. See #1393.
clearChunkReloadFlag()

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </StrictMode>
)
