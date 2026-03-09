import { useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

interface Props {
  value:    string
  onChange: (v: string) => void
  disabled?: boolean
}

export default function SearchInput({ value, onChange, disabled = false }: Props) {
  const [focused, setFocused] = useState(false)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  const handleKeyDown = (e: React.KeyboardEvent) => {
    // Allow Shift+Enter for newlines; plain Enter triggers search
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
    }
  }

  return (
    <div className="space-y-3">
      <label className="flex items-center gap-2 text-xs font-mono text-cyan-400/60 uppercase tracking-[0.2em]">
        <span className="w-1.5 h-1.5 rounded-full bg-cyan-400/60 block" />
        Natural Language Query
      </label>

      <motion.div
        animate={focused ? { scale: 1.005 } : { scale: 1 }}
        transition={{ type: 'spring', stiffness: 400, damping: 30 }}
        className="relative"
      >
        {/* Glow border layer */}
        <AnimatePresence>
          {focused && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="absolute -inset-[1px] rounded-2xl pointer-events-none"
              style={{
                background: 'linear-gradient(135deg, rgba(0,229,255,0.3), rgba(168,85,247,0.3))',
                filter: 'blur(1px)',
              }}
            />
          )}
        </AnimatePresence>

        <div className={`
          relative rounded-2xl overflow-hidden
          border transition-all duration-300
          ${focused
            ? 'border-transparent bg-void-2'
            : 'border-white/[0.07] bg-void-2/80'}
        `}>
          {/* Shimmer animation on focus */}
          {focused && (
            <motion.div
              className="absolute inset-0 pointer-events-none"
              style={{
                background: 'linear-gradient(90deg, transparent 0%, rgba(0,229,255,0.03) 50%, transparent 100%)',
              }}
              animate={{ x: ['-100%', '100%'] }}
              transition={{ duration: 3, repeat: Infinity, ease: 'linear' }}
            />
          )}

          {/* Search icon */}
          <div className="absolute left-4 top-4 pointer-events-none">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
              className={`w-5 h-5 transition-colors duration-300 ${focused ? 'text-cyan-400' : 'text-white/20'}`}
              strokeWidth="1.8" strokeLinecap="round">
              <circle cx="11" cy="11" r="8"/>
              <path d="M21 21l-4.35-4.35"/>
            </svg>
          </div>

          {/* Textarea */}
          <textarea
            ref={inputRef}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onFocus={() => setFocused(true)}
            onBlur={() => setFocused(false)}
            onKeyDown={handleKeyDown}
            disabled={disabled}
            rows={3}
            placeholder="Describe what you're looking for…&#10;e.g. 'blue running shoes under $100' or 'minimalist wooden desk lamp'"
            className={`
              w-full pl-12 pr-12 py-4 bg-transparent
              font-body text-[15px] text-white/90 placeholder-white/20
              resize-none outline-none
              disabled:opacity-40 disabled:cursor-not-allowed
              transition-colors duration-200
              leading-relaxed
            `}
            aria-label="Search query"
            maxLength={500}
          />

          {/* Clear button */}
          <AnimatePresence>
            {value && (
              <motion.button
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                transition={{ duration: 0.15 }}
                onClick={() => { onChange(''); inputRef.current?.focus() }}
                className="absolute right-3 top-3 w-7 h-7 rounded-lg flex items-center justify-center
                           text-white/30 hover:text-white/60 hover:bg-white/[0.06]
                           transition-all duration-200 font-mono text-xs"
                aria-label="Clear input"
              >
                ✕
              </motion.button>
            )}
          </AnimatePresence>

          {/* Character count */}
          <div className="absolute right-3 bottom-3 font-mono text-[10px] text-white/20 select-none">
            {value.length}/500
          </div>
        </div>
      </motion.div>
    </div>
  )
}
