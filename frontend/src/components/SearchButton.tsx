import { motion, AnimatePresence } from 'framer-motion'

interface Props {
  onClick:   () => void
  loading:   boolean
  disabled:  boolean
}

export default function SearchButton({ onClick, loading, disabled }: Props) {
  return (
    <motion.button
      onClick={onClick}
      disabled={disabled || loading}
      whileHover={!disabled && !loading ? { scale: 1.02, y: -1 } : {}}
      whileTap={!disabled && !loading   ? { scale: 0.98 }        : {}}
      transition={{ type: 'spring', stiffness: 400, damping: 25 }}
      aria-busy={loading}
      aria-label={loading ? 'Searching…' : 'Run multimodal search'}
      className={`
        relative w-full py-4 rounded-2xl font-display text-sm font-semibold
        tracking-widest uppercase overflow-hidden
        transition-all duration-300
        ${disabled && !loading
          ? 'opacity-40 cursor-not-allowed bg-void-2 border border-white/10 text-white/30'
          : 'cursor-pointer text-void'}
      `}
      style={!disabled || loading ? {
        background: loading
          ? 'linear-gradient(135deg, #0d9488, #6d28d9)'
          : 'linear-gradient(135deg, #00e5ff, #a855f7)',
        boxShadow: loading
          ? '0 0 30px rgba(109,40,217,0.3)'
          : '0 0 30px rgba(0,229,255,0.25), 0 4px 20px rgba(168,85,247,0.2)',
      } : {}}
    >
      {/* Animated gradient sweep */}
      {!disabled && !loading && (
        <motion.div
          className="absolute inset-0 pointer-events-none"
          style={{
            background: 'linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.1) 50%, transparent 100%)',
          }}
          animate={{ x: ['-100%', '100%'] }}
          transition={{ duration: 2.5, repeat: Infinity, ease: 'linear' }}
        />
      )}

      <AnimatePresence mode="wait">
        {loading ? (
          <motion.span
            key="loading"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="relative flex items-center justify-center gap-3 text-white"
          >
            {/* Spinner */}
            <motion.svg
              viewBox="0 0 24 24" fill="none"
              className="w-4 h-4"
              animate={{ rotate: 360 }}
              transition={{ duration: 0.8, repeat: Infinity, ease: 'linear' }}
            >
              <circle cx="12" cy="12" r="10" stroke="rgba(255,255,255,0.2)" strokeWidth="2.5"/>
              <path d="M12 2a10 10 0 0110 10" stroke="white" strokeWidth="2.5" strokeLinecap="round"/>
            </motion.svg>
            <span className="font-mono">Processing Query…</span>
            {/* Animated dots */}
            <span className="flex gap-0.5">
              {[0, 1, 2].map(i => (
                <motion.span key={i} className="w-1 h-1 rounded-full bg-white/60 block"
                  animate={{ opacity: [0.3, 1, 0.3], y: [0, -3, 0] }}
                  transition={{ duration: 0.8, repeat: Infinity, delay: i * 0.15 }}
                />
              ))}
            </span>
          </motion.span>
        ) : (
          <motion.span
            key="idle"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="relative flex items-center justify-center gap-3"
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
              className="w-4 h-4" strokeWidth="2" strokeLinecap="round">
              <circle cx="11" cy="11" r="8"/>
              <path d="M21 21l-4.35-4.35"/>
            </svg>
            <span>Search with AI</span>
            {/* Arrow */}
            <motion.span
              animate={{ x: [0, 3, 0] }}
              transition={{ duration: 1.5, repeat: Infinity, ease: 'easeInOut' }}
            >
              →
            </motion.span>
          </motion.span>
        )}
      </AnimatePresence>
    </motion.button>
  )
}
