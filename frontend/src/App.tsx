import { useState, useCallback, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

// Components
import ImageUploader    from './components/ImageUploader'
import SearchInput      from './components/SearchInput'
import MethodSelector   from './components/MethodSelector'
import SearchButton     from './components/SearchButton'
import ResultsGrid      from './components/ResultsGrid'
import DiagnosticsPanel from './components/DiagnosticsPanel'

// Services & Types
import { searchProducts, ApiError } from './services/api'
import type { SearchMethod, SearchState, SearchResponse } from './types/search'

// ─── Initial State ────────────────────────────────────────────────────────────
const INITIAL_STATE: SearchState = {
  status:       'idle',
  results:      [],
  diagnostics:  null,
  error:        null,
  totalResults: 0,
}

export default function App() {
  // ── Input State ──────────────────────────────────────────────────────────────
  const [image,  setImage]  = useState<File | null>(null)
  const [query,  setQuery]  = useState('')
  const [method, setMethod] = useState<SearchMethod>('symbolic_early')


  // ── Search State ─────────────────────────────────────────────────────────────
  const [search, setSearch] = useState<SearchState>(INITIAL_STATE)

  // ── Abort Controller for cancellation ────────────────────────────────────────
  const abortRef = useRef<AbortController | null>(null)

  // ── Validation ────────────────────────────────────────────────────────────────
  const canSearch = (image !== null || query.trim().length > 0) && search.status !== 'loading'

  // ── Run Search ────────────────────────────────────────────────────────────────
  const handleSearch = useCallback(async () => {
    if (!canSearch) return

    // Cancel any in-flight request
    abortRef.current?.abort()
    abortRef.current = new AbortController()

    setSearch(prev => ({ ...prev, status: 'loading', error: null }))

    try {
      const response: SearchResponse = await searchProducts(
        { text: query, method, image: image ?? undefined },
        abortRef.current.signal,
      )

      setSearch({
        status:       'success',
        results:      response.results,
        diagnostics:  response.diagnostics,
        error:        null,
        totalResults: response.total,
      })
    } catch (err) {
      if ((err as Error).name === 'AbortError') return // Cancelled — ignore

      const message = err instanceof ApiError
        ? `${err.message} (${err.status})`
        : err instanceof Error
        ? err.message
        : 'Unknown error occurred'

      setSearch(prev => ({
        ...prev,
        status: 'error',
        error:  message,
      }))
    }
  }, [canSearch, image, query, method])

  // ─────────────────────────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen font-body">
      {/* Background glow orbs */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden" aria-hidden="true">
        <div className="absolute -top-40 -left-40 w-[500px] h-[500px] rounded-full opacity-[0.04]"
          style={{ background: 'radial-gradient(circle, #00e5ff, transparent 70%)' }}
        />
        <div className="absolute -top-20 -right-40 w-[600px] h-[600px] rounded-full opacity-[0.03]"
          style={{ background: 'radial-gradient(circle, #a855f7, transparent 70%)' }}
        />
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-10">

        {/* ── Header ── */}
        <motion.header
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
          className="text-center space-y-4"
        >
          {/* Top badge */}
          <motion.div
            animate={{ y: [0, -4, 0] }}
            transition={{ duration: 4, repeat: Infinity, ease: 'easeInOut' }}
            className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full
                       border border-cyan-400/20 bg-cyan-400/[0.05] backdrop-blur-sm"
          >
            <motion.span
              animate={{ opacity: [0.4, 1, 0.4] }}
              transition={{ duration: 2, repeat: Infinity }}
              className="w-1.5 h-1.5 rounded-full bg-cyan-400 block"
            />
            <span className="font-mono text-[11px] text-cyan-400/70 tracking-[0.2em] uppercase">
              CLIP · Symbolic Fusion · Intent Search
            </span>
          </motion.div>

          {/* Title */}
          <div>
            <h1 className="font-display text-4xl sm:text-5xl lg:text-6xl font-black tracking-wider">
              <span className="text-gradient-cyan">NEXUS</span>
            </h1>
            <p className="font-display text-xs sm:text-sm text-white/20 tracking-[0.4em] uppercase mt-1">
              Multimodal AI Product Search
            </p>
          </div>

          {/* Subtitle */}
          <p className="font-body text-sm text-white/35 max-w-md mx-auto leading-relaxed">
            Semantic retrieval powered by CLIP image embeddings,
            symbolic fusion, and natural language understanding.
          </p>
        </motion.header>

        {/* ── Search Panel ── */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.15, ease: [0.22, 1, 0.36, 1] }}
          className="max-w-3xl mx-auto"
        >
          <div className="glass-panel p-6 sm:p-8 space-y-6">
            {/* Decorative top bar */}
            <div className="flex items-center gap-3 pb-2 border-b border-white/[0.05]">
              <div className="flex gap-1.5">
                <div className="w-2.5 h-2.5 rounded-full bg-red-400/30 border border-red-400/20" />
                <div className="w-2.5 h-2.5 rounded-full bg-yellow-400/30 border border-yellow-400/20" />
                <div className="w-2.5 h-2.5 rounded-full bg-green-400/30 border border-green-400/20" />
              </div>
              <span className="font-mono text-[10px] text-white/15 tracking-widest uppercase">
                Search Interface v1.0
              </span>
            </div>

            <ImageUploader image={image} onImageChange={setImage} />
            <SearchInput
              value={query}
              onChange={setQuery}
              disabled={search.status === 'loading'}
            />
            <MethodSelector
              selected={method}
              onSelect={setMethod}
              disabled={search.status === 'loading'}
            />

            {/* Validation hint */}
            <AnimatePresence>
              {!canSearch && search.status !== 'loading' && (image || query) === null && (
                <motion.p
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="font-mono text-[11px] text-white/25 text-center"
                >
                  Add an image or enter a query to begin
                </motion.p>
              )}
            </AnimatePresence>

            <SearchButton
              onClick={handleSearch}
              loading={search.status === 'loading'}
              disabled={!canSearch}
            />
          </div>
        </motion.div>

        {/* ── Loading shimmer ── */}
        <AnimatePresence>
          {search.status === 'loading' && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="max-w-3xl mx-auto"
            >
              <div className="flex flex-col items-center gap-6 py-10">
                {/* Multi-ring spinner */}
                <div className="relative w-16 h-16">
                  {[0, 1, 2].map(i => (
                    <motion.div
                      key={i}
                      className="absolute inset-0 rounded-full border"
                      style={{
                        borderColor: i === 0
                          ? 'rgba(0,229,255,0.5)'
                          : i === 1
                          ? 'rgba(168,85,247,0.3)'
                          : 'rgba(0,229,255,0.1)',
                        margin: `${i * 6}px`,
                      }}
                      animate={{ rotate: i % 2 === 0 ? 360 : -360 }}
                      transition={{
                        duration: 1.5 + i * 0.5,
                        repeat: Infinity,
                        ease: 'linear',
                      }}
                    />
                  ))}
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="w-3 h-3 rounded-full bg-cyan-400/50" />
                  </div>
                </div>

                <div className="text-center space-y-1">
                  <p className="font-display text-sm text-cyan-400/70 tracking-widest uppercase">
                    Processing Query
                  </p>
                  <p className="font-mono text-xs text-white/25">
                    Running {method.replace('_', ' ')} inference…
                  </p>
                </div>

                {/* Progress bar */}
                <div className="w-48 h-0.5 bg-white/[0.05] rounded-full overflow-hidden">
                  <motion.div
                    className="h-full rounded-full bg-gradient-to-r from-cyan-400 to-violet-500"
                    animate={{ x: ['-100%', '100%'] }}
                    transition={{ duration: 1.5, repeat: Infinity, ease: 'easeInOut' }}
                  />
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── Error State ── */}
        <AnimatePresence>
          {search.status === 'error' && search.error && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="max-w-3xl mx-auto"
            >
              <div className="rounded-2xl border border-red-500/20 bg-red-500/[0.05] p-6 text-center space-y-3">
                <div className="text-3xl">⚠</div>
                <div>
                  <p className="font-display text-sm text-red-400/80 uppercase tracking-widest">
                    Query Failed
                  </p>
                  <p className="font-mono text-xs text-white/40 mt-1">{search.error}</p>
                </div>
                <button
                  onClick={() => setSearch(INITIAL_STATE)}
                  className="font-mono text-xs text-red-400/60 hover:text-red-400
                             underline underline-offset-4 transition-colors duration-200"
                >
                  Dismiss
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── Results ── */}
        <AnimatePresence>
          {search.status === 'success' && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="space-y-8"
            >
              {/* Diagnostics panel */}
              {search.diagnostics && (
                <div className="max-w-3xl mx-auto">
                  <DiagnosticsPanel diagnostics={search.diagnostics} />
                </div>
              )}

              {/* Results grid */}
              <ResultsGrid
                products={search.results}
                method={method}
                total={search.totalResults}
              />
            </motion.div>
          )}
        </AnimatePresence>

      </div>

      {/* ── Footer ── */}
      <footer className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-24 pt-4">
        <div className="flex flex-wrap items-center justify-center gap-x-6 gap-y-2">
          {[
            'CLIP ViT-L/14',
            'Symbolic Fusion',
            'Late Fusion',
            'Intent Search',
            'FastAPI',
            'React 18',
            'TypeScript',
          ].map((tag) => (
            <span
              key={tag}
              className="font-mono text-[10px] text-white/15 tracking-[0.15em] uppercase"
            >
              {tag}
            </span>
          ))}
        </div>
        <p className="text-center font-mono text-[10px] text-white/10 mt-3 tracking-widest uppercase">
          Nexus · Multimodal AI Product Search · v1.0
        </p>
      </footer>

      {/* Bottom fade */}
      <div className="fixed bottom-0 left-0 right-0 h-20 pointer-events-none"
        style={{ background: 'linear-gradient(to top, #0b0f1a, transparent)' }}
        aria-hidden="true"
      />
    </div>
  )
}
