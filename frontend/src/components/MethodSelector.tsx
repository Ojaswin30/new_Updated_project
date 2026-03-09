import { motion } from 'framer-motion'
import type { SearchMethod, SearchMethodOption } from '../types/search'

const METHODS: SearchMethodOption[] = [
  {
    id:         'symbolic_early',
    label:      'Symbolic Early Fusion',
    shortLabel: 'Early Fusion',
    description:'Combines image embeddings and text constraints before retrieval. Highest precision for visual queries.',
    icon:       '⬡',
    gradient:   'from-cyan-400/20 to-blue-500/20',
  },
  {
    id:         'late_fusion',
    label:      'Late Fusion',
    shortLabel: 'Late Fusion',
    description:'Ranks image and text results independently, then merges. Robust and flexible for mixed queries.',
    icon:       '◈',
    gradient:   'from-violet-400/20 to-purple-600/20',
  },
  {
    id:         'intent_based',
    label:      'Intent-Based Search',
    shortLabel: 'Intent Search',
    description:'Infers user intent from natural language to generate dynamic query constraints automatically.',
    icon:       '⊛',
    gradient:   'from-emerald-400/20 to-teal-500/20',
  },
]

interface Props {
  selected:  SearchMethod
  onSelect:  (m: SearchMethod) => void
  disabled?: boolean
}

const GLOW_COLORS: Record<SearchMethod, string> = {
  symbolic_early: 'rgba(0, 229, 255, 0.25)',
  late_fusion:    'rgba(168, 85, 247, 0.25)',
  intent_based:      'rgba(52, 211, 153, 0.25)',
}

const BORDER_COLORS: Record<SearchMethod, string> = {
  symbolic_early: 'rgba(0, 229, 255, 0.5)',
  late_fusion:    'rgba(168, 85, 247, 0.5)',
  intent_based:      'rgba(52, 211, 153, 0.5)',
}

const TEXT_COLORS: Record<SearchMethod, string> = {
  symbolic_early: 'text-cyan-400',
  late_fusion:    'text-violet-400',
  intent_based:      'text-emerald-400',
}

export default function MethodSelector({ selected, onSelect, disabled = false }: Props) {
  return (
    <div className="space-y-3">
      <label className="flex items-center gap-2 text-xs font-mono text-cyan-400/60 uppercase tracking-[0.2em]">
        <span className="w-1.5 h-1.5 rounded-full bg-cyan-400/60 block" />
        Retrieval Strategy
      </label>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3" role="radiogroup" aria-label="Search method">
        {METHODS.map((method) => {
          const isSelected = selected === method.id

          return (
            <motion.label
              key={method.id}
              whileHover={!disabled ? { y: -2, scale: 1.01 } : {}}
              whileTap={!disabled   ? { scale: 0.98 }         : {}}
              transition={{ type: 'spring', stiffness: 400, damping: 25 }}
              className={`
                relative rounded-2xl p-4 cursor-pointer select-none
                border transition-all duration-300 overflow-hidden
                ${disabled ? 'opacity-40 cursor-not-allowed' : ''}
                ${isSelected
                  ? 'bg-void-2/90'
                  : 'bg-void-2/40 border-white/[0.07] hover:border-white/15'}
              `}
              style={isSelected ? {
                borderColor: BORDER_COLORS[method.id],
                boxShadow:   `0 0 20px ${GLOW_COLORS[method.id]}, inset 0 0 20px ${GLOW_COLORS[method.id]}`,
              } : {}}
              aria-label={`${method.label} — ${method.description}`}
            >
              <input
                type="radio"
                name="search-method"
                value={method.id}
                checked={isSelected}
                onChange={() => !disabled && onSelect(method.id)}
                className="sr-only"
              />

              {/* Background gradient when selected */}
              {isSelected && (
                <motion.div
                  layoutId="method-bg"
                  className={`absolute inset-0 bg-gradient-to-br ${method.gradient} opacity-40`}
                  transition={{ type: 'spring', stiffness: 300, damping: 30 }}
                />
              )}

              {/* Content */}
              <div className="relative z-10 space-y-2">
                <div className="flex items-center justify-between">
                  <span className={`
                    text-2xl transition-all duration-300
                    ${isSelected ? TEXT_COLORS[method.id] : 'text-white/20'}
                  `}>
                    {method.icon}
                  </span>

                  {/* Selected indicator */}
                  <motion.div
                    animate={isSelected
                      ? { scale: 1, opacity: 1 }
                      : { scale: 0.5, opacity: 0 }}
                    transition={{ type: 'spring', stiffness: 500 }}
                    className={`w-2 h-2 rounded-full ${TEXT_COLORS[method.id].replace('text-', 'bg-')}`}
                    style={{
                      boxShadow: isSelected ? `0 0 8px ${GLOW_COLORS[method.id]}` : 'none',
                    }}
                  />
                </div>

                <div>
                  <h3 className={`
                    font-display text-xs font-semibold tracking-wide transition-colors duration-300
                    ${isSelected ? TEXT_COLORS[method.id] : 'text-white/50'}
                  `}>
                    {method.shortLabel}
                  </h3>
                  <p className="font-mono text-[10px] text-white/25 mt-1 leading-relaxed">
                    {method.description}
                  </p>
                </div>
              </div>

              {/* Hover shimmer */}
              <div className="absolute inset-0 opacity-0 hover:opacity-100 transition-opacity duration-300 pointer-events-none overflow-hidden rounded-2xl">
                <div
                  className="absolute inset-0 translate-x-[-100%] hover:translate-x-[100%] transition-transform duration-700"
                  style={{
                    background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.03), transparent)',
                  }}
                />
              </div>
            </motion.label>
          )
        })}
      </div>
    </div>
  )
}
