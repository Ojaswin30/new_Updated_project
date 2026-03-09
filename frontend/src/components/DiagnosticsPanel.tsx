import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import type { Diagnostics } from '../types/search'

interface Props {
  diagnostics: Diagnostics
}

// ── Sub-panel: a collapsible section within diagnostics
function ConsoleSection({
  title, icon, children, defaultOpen = false,
}: {
  title:       string
  icon:        string
  children:    React.ReactNode
  defaultOpen?: boolean
}) {
  const [open, setOpen] = useState(defaultOpen)

  return (
    <div className="border border-white/[0.06] rounded-xl overflow-hidden">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-4 py-2.5
                   hover:bg-white/[0.03] transition-colors duration-200 text-left"
      >
        <span className="flex items-center gap-2">
          <span className="text-cyan-400/60 text-sm">{icon}</span>
          <span className="font-mono text-[11px] text-cyan-400/70 uppercase tracking-wider">{title}</span>
        </span>
        <motion.span
          animate={{ rotate: open ? 180 : 0 }}
          transition={{ duration: 0.2 }}
          className="text-white/20 text-xs font-mono"
        >
          ▾
        </motion.span>
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25, ease: 'easeInOut' }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-4 pt-1 border-t border-white/[0.04]">
              {children}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

// ── Key-value row
function KVRow({ label, value }: { label: string; value: string | number | undefined | null }) {
  if (value === undefined || value === null) return null
  return (
    <div className="flex gap-3 items-start py-0.5">
      <span className="font-mono text-[10px] text-white/25 min-w-[120px] shrink-0 pt-px uppercase tracking-wider">
        {label}
      </span>
      <span className="font-mono text-[10px] text-cyan-300/70 break-all">{String(value)}</span>
    </div>
  )
}

// ── SQL block with syntax highlight
function SqlBlock({ sql }: { sql: string }) {
  const highlighted = sql
    .replace(/\b(SELECT|FROM|WHERE|AND|OR|ORDER BY|LIMIT|JOIN|LEFT|RIGHT|ON|AS|IN|LIKE|BETWEEN|GROUP BY|HAVING|INNER|OUTER)\b/g,
      '<span class="text-violet-400">$1</span>')
    .replace(/\b(products|categories|ratings)\b/g,
      '<span class="text-cyan-300">$1</span>')
    .replace(/('[^']*')/g,
      '<span class="text-amber-300/80">$1</span>')
    .replace(/\b(\d+\.?\d*)\b/g,
      '<span class="text-emerald-300/80">$1</span>')

  return (
    <div className="relative mt-2 rounded-xl overflow-hidden border border-white/[0.06]">
      {/* Header bar */}
      <div className="flex items-center justify-between px-3 py-1.5 bg-white/[0.03] border-b border-white/[0.05]">
        <div className="flex gap-1.5">
          <div className="w-2 h-2 rounded-full bg-red-400/40" />
          <div className="w-2 h-2 rounded-full bg-yellow-400/40" />
          <div className="w-2 h-2 rounded-full bg-green-400/40" />
        </div>
        <span className="font-mono text-[9px] text-white/20 uppercase tracking-wider">Generated SQL</span>
      </div>
      <pre
        className="p-4 font-mono text-[10px] leading-relaxed text-white/50 overflow-x-auto"
        dangerouslySetInnerHTML={{ __html: highlighted }}
      />
    </div>
  )
}

// ── Fusion stats bars
function FusionBar({ label, value, max = 1 }: { label: string; value: number; max?: number }) {
  const pct = Math.min((value / max) * 100, 100)
  return (
    <div className="space-y-1">
      <div className="flex justify-between items-center">
        <span className="font-mono text-[9px] text-white/30 uppercase tracking-wider">{label}</span>
        <span className="font-mono text-[9px] text-cyan-400/70">{typeof value === 'number' && value < 2 ? (value * 100).toFixed(0) + '%' : value}</span>
      </div>
      <div className="h-1 bg-white/[0.05] rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.7, ease: 'easeOut' }}
          className="h-full rounded-full bg-gradient-to-r from-cyan-400 to-violet-500"
        />
      </div>
    </div>
  )
}

export default function DiagnosticsPanel({ diagnostics: diag }: Props) {
  const [panelOpen, setPanelOpen] = useState(false)

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.3 }}
      className="rounded-2xl border border-cyan-400/10 overflow-hidden"
      style={{
        background: 'rgba(0, 10, 20, 0.85)',
        backdropFilter: 'blur(20px)',
      }}
    >
      {/* Toggle header */}
      <button
        onClick={() => setPanelOpen(o => !o)}
        className="w-full flex items-center justify-between px-5 py-4
                   hover:bg-cyan-400/[0.03] transition-colors duration-200"
      >
        <div className="flex items-center gap-3">
          {/* Blinking indicator */}
          <motion.div
            animate={{ opacity: [1, 0.3, 1] }}
            transition={{ duration: 1.5, repeat: Infinity }}
            className="w-2 h-2 rounded-full bg-cyan-400"
          />
          <span className="font-display text-xs text-cyan-400/80 uppercase tracking-[0.2em]">
            AI Diagnostics Console
          </span>
          {diag.processing_time_ms !== undefined && (
            <span className="font-mono text-[10px] text-white/20">
              {diag.processing_time_ms.toFixed(0)}ms
            </span>
          )}
        </div>

        <div className="flex items-center gap-3">
          <span className="font-mono text-[10px] text-white/30">
            {panelOpen ? 'collapse' : 'expand'}
          </span>
          <motion.div
            animate={{ rotate: panelOpen ? 180 : 0 }}
            transition={{ duration: 0.25 }}
            className="text-cyan-400/40 text-sm"
          >
            ▾
          </motion.div>
        </div>
      </button>

      <AnimatePresence>
        {panelOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: 'easeInOut' }}
            className="overflow-hidden"
          >
            <div className="px-5 pb-5 space-y-3 border-t border-cyan-400/10">
              {/* Scan line animation */}
              <div className="relative h-px overflow-hidden">
                <motion.div
                  animate={{ x: ['-100%', '100%'] }}
                  transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                  className="absolute h-full w-1/3 bg-gradient-to-r from-transparent via-cyan-400/50 to-transparent"
                />
              </div>

              {/* ── Image Signal ── */}
              {diag.image_signal && (
                <ConsoleSection title="Image Signal" icon="◉" defaultOpen>
                  <div className="space-y-0.5 mt-2">
                    {diag.image_signal.categories && (
                      <KVRow label="Categories" value={diag.image_signal.categories.join(', ')} />
                    )}
                    {diag.image_signal.colors && (
                      <KVRow label="Colors" value={diag.image_signal.colors.join(', ')} />
                    )}
                    {diag.image_signal.style && (
                      <KVRow label="Style" value={diag.image_signal.style} />
                    )}
                    {diag.image_signal.embedding_dim && (
                      <KVRow label="Embedding dim" value={`${diag.image_signal.embedding_dim}D vector`} />
                    )}
                    {diag.image_signal.attributes && Object.entries(diag.image_signal.attributes).map(([k, v]) => (
                      <KVRow key={k} label={k} value={String(v)} />
                    ))}
                    {diag.image_signal.raw && (
                      <div className="mt-2 p-3 rounded-lg bg-black/30 border border-white/[0.05]">
                        <pre className="font-mono text-[9px] text-white/30 overflow-x-auto">
                          {JSON.stringify(diag.image_signal.raw, null, 2)}
                        </pre>
                      </div>
                    )}
                  </div>
                </ConsoleSection>
              )}

              {/* ── Text Constraints ── */}
              {diag.text_constraints && (
                <ConsoleSection title="Text Constraints" icon="⊡" defaultOpen>
                  <div className="space-y-0.5 mt-2">
                    {diag.text_constraints.keywords && (
                      <KVRow label="Keywords" value={diag.text_constraints.keywords.join(', ')} />
                    )}
                    {diag.text_constraints.category && (
                      <KVRow label="Category" value={diag.text_constraints.category} />
                    )}
                    {diag.text_constraints.price_range && (
                      <KVRow label="Price range"
                        value={`$${diag.text_constraints.price_range.min ?? '0'} – $${diag.text_constraints.price_range.max ?? '∞'}`}
                      />
                    )}
                    {diag.text_constraints.filters && Object.entries(diag.text_constraints.filters).map(([k, v]) => (
                      <KVRow key={k} label={k} value={String(v)} />
                    ))}
                  </div>
                </ConsoleSection>
              )}

              {/* ── Final Constraints ── */}
              {diag.final_constraints && Object.keys(diag.final_constraints).length > 0 && (
                <ConsoleSection title="Final Constraints" icon="⊞">
                  <div className="mt-2 p-3 rounded-lg bg-black/30 border border-white/[0.05]">
                    <pre className="font-mono text-[9px] text-white/30 overflow-x-auto">
                      {JSON.stringify(diag.final_constraints, null, 2)}
                    </pre>
                  </div>
                </ConsoleSection>
              )}

              {/* ── Generated SQL ── */}
              {diag.generated_sql && (
                <ConsoleSection title="Generated SQL" icon="◈" defaultOpen>
                  <SqlBlock sql={diag.generated_sql} />
                </ConsoleSection>
              )}

              {/* ── Fusion Stats ── */}
              {diag.fusion_stats && (
                <ConsoleSection title="Fusion Statistics" icon="◎" defaultOpen>
                  <div className="mt-2 space-y-3">
                    <KVRow label="Method" value={diag.fusion_stats.method} />
                    {diag.fusion_stats.candidates_ranked !== undefined && (
                      <KVRow label="Candidates ranked" value={diag.fusion_stats.candidates_ranked} />
                    )}
                    {diag.fusion_stats.fusion_time_ms !== undefined && (
                      <KVRow label="Fusion time" value={`${diag.fusion_stats.fusion_time_ms.toFixed(1)}ms`} />
                    )}
                    {diag.fusion_stats.image_weight !== undefined && (
                      <FusionBar label="Image weight" value={diag.fusion_stats.image_weight} />
                    )}
                    {diag.fusion_stats.text_weight !== undefined && (
                      <FusionBar label="Text weight" value={diag.fusion_stats.text_weight} />
                    )}
                  </div>
                </ConsoleSection>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}
