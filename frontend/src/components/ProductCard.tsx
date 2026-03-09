import { motion } from 'framer-motion'
import type { Product } from '../types/search'

interface Props {
  product: Product
  index:   number
}

function StarRating({ rating }: { rating: number }) {
  const full    = Math.floor(rating)
  const partial = rating - full
  const empty   = 5 - Math.ceil(rating)

  return (
    <div className="flex items-center gap-0.5" aria-label={`${rating.toFixed(1)} out of 5`}>
      {Array.from({ length: full }).map((_, i) => (
        <svg key={`f-${i}`} viewBox="0 0 24 24" className="w-3 h-3 fill-amber-400 text-amber-400">
          <polygon points="12,2 15.09,8.26 22,9.27 17,14.14 18.18,21.02 12,17.77 5.82,21.02 7,14.14 2,9.27 8.91,8.26"/>
        </svg>
      ))}
      {partial > 0 && (
        <div className="relative w-3 h-3">
          <svg viewBox="0 0 24 24" className="absolute inset-0 w-3 h-3 fill-white/10 text-white/10">
            <polygon points="12,2 15.09,8.26 22,9.27 17,14.14 18.18,21.02 12,17.77 5.82,21.02 7,14.14 2,9.27 8.91,8.26"/>
          </svg>
          <div className="absolute inset-0 overflow-hidden" style={{ width: `${partial * 100}%` }}>
            <svg viewBox="0 0 24 24" className="w-3 h-3 fill-amber-400">
              <polygon points="12,2 15.09,8.26 22,9.27 17,14.14 18.18,21.02 12,17.77 5.82,21.02 7,14.14 2,9.27 8.91,8.26"/>
            </svg>
          </div>
        </div>
      )}
      {Array.from({ length: empty }).map((_, i) => (
        <svg key={`e-${i}`} viewBox="0 0 24 24" className="w-3 h-3 fill-white/10">
          <polygon points="12,2 15.09,8.26 22,9.27 17,14.14 18.18,21.02 12,17.77 5.82,21.02 7,14.14 2,9.27 8.91,8.26"/>
        </svg>
      ))}
      <span className="font-mono text-[10px] text-white/30 ml-1">{rating.toFixed(1)}</span>
    </div>
  )
}

function ScoreBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="space-y-0.5">
      <div className="flex justify-between items-center">
        <span className="font-mono text-[9px] text-white/30 uppercase tracking-wider">{label}</span>
        <span className="font-mono text-[9px]" style={{ color }}>{(value * 100).toFixed(0)}%</span>
      </div>
      <div className="h-0.5 bg-white/[0.05] rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${value * 100}%` }}
          transition={{ duration: 0.8, delay: 0.2, ease: 'easeOut' }}
          className="h-full rounded-full"
          style={{ background: color }}
        />
      </div>
    </div>
  )
}

export default function ProductCard({ product, index }: Props) {
  const scorePercent = Math.round(product.final_score * 100)

  // Determine score color tier
  const scoreColor =
    scorePercent >= 80 ? '#00e5ff' :
    scorePercent >= 60 ? '#a855f7' :
    scorePercent >= 40 ? '#f59e0b' :
    '#6b7280'

  return (
    <motion.article
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: index * 0.07, ease: [0.22, 1, 0.36, 1] }}
      whileHover={{ y: -4, scale: 1.01 }}
      className="group relative rounded-2xl overflow-hidden border border-white/[0.07] bg-void-2/60
                 backdrop-blur-sm transition-all duration-300 flex flex-col
                 hover:border-white/15 hover:shadow-card-hover"
    >
      {/* Image */}
      <div className="relative overflow-hidden bg-void-3 aspect-square">
        <motion.img
          src={product.image_url}
          alt={product.title}
          className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
          loading="lazy"
          onError={(e) => {
            // Fallback for broken images
            (e.target as HTMLImageElement).src = `https://placehold.co/400x400/0d1220/00e5ff?text=${encodeURIComponent(product.title.slice(0, 12))}`
          }}
        />

        {/* Score badge */}
        <div
          className="absolute top-2.5 right-2.5 font-mono text-xs font-semibold
                     px-2 py-1 rounded-xl backdrop-blur-md border"
          style={{
            background: `rgba(0,0,0,0.7)`,
            borderColor: `${scoreColor}40`,
            color: scoreColor,
            boxShadow: `0 0 12px ${scoreColor}30`,
          }}
        >
          {scorePercent}
          <span className="text-[8px] opacity-60 ml-0.5">%</span>
        </div>

        {/* Category badge */}
        {product.category && (
          <div className="absolute top-2.5 left-2.5 font-mono text-[9px] uppercase tracking-wider
                          px-2 py-1 rounded-xl bg-black/60 border border-white/10 text-white/40
                          backdrop-blur-sm">
            {product.category}
          </div>
        )}

        {/* Hover overlay */}
        <div className="absolute inset-0 bg-gradient-to-t from-void/90 via-transparent to-transparent
                        opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
      </div>

      {/* Content */}
      <div className="p-3.5 flex flex-col gap-2.5 flex-1">
        {/* Title */}
        <h3 className="font-body text-sm text-white/85 font-medium leading-snug line-clamp-2">
          {product.title}
        </h3>

        {/* Rating & Price row */}
        <div className="flex items-center justify-between">
          <StarRating rating={product.rating} />
          <span className="font-display text-sm font-bold"
            style={{ color: scoreColor }}>
            ${product.price.toFixed(2)}
          </span>
        </div>

        {/* Score bars */}
        <div className="space-y-1.5 pt-1 border-t border-white/[0.05]">
          {product.clip_score !== undefined && (
            <ScoreBar label="CLIP" value={product.clip_score} color="#00e5ff" />
          )}
          {product.text_score !== undefined && (
            <ScoreBar label="Text" value={product.text_score} color="#a855f7" />
          )}
          <ScoreBar label="Final" value={product.final_score} color={scoreColor} />
        </div>
      </div>

      {/* Bottom neon accent line */}
      <motion.div
        initial={{ scaleX: 0 }}
        animate={{ scaleX: 1 }}
        transition={{ delay: index * 0.07 + 0.3, duration: 0.4 }}
        className="h-[1px] origin-left"
        style={{ background: `linear-gradient(90deg, ${scoreColor}60, transparent)` }}
      />
    </motion.article>
  )
}
