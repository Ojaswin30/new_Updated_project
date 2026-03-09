import { motion } from 'framer-motion'
import ProductCard from './ProductCard'
import type { Product, SearchMethod } from '../types/search'

interface Props {
  products: Product[]
  method:   SearchMethod
  total:    number
}

const METHOD_LABELS: Record<SearchMethod, string> = {
  symbolic_early: 'Symbolic Early Fusion',
  late_fusion:    'Late Fusion',
  intent_based:   'Intent-Based Search',
}

export default function ResultsGrid({ products, method, total }: Props) {
  if (products.length === 0) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="text-center py-20"
      >
        <div className="text-4xl mb-4 opacity-30">◈</div>
        <p className="font-mono text-sm text-white/30">No products matched your query.</p>
        <p className="font-mono text-xs text-white/15 mt-1">Try adjusting your search or method.</p>
      </motion.div>
    )
  }

  return (
    <div className="space-y-5">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex flex-wrap items-center justify-between gap-3"
      >
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <motion.div
              animate={{ opacity: [0.5, 1, 0.5] }}
              transition={{ duration: 2, repeat: Infinity }}
              className="w-2 h-2 rounded-full bg-cyan-400"
            />
            <span className="font-mono text-xs text-white/50 uppercase tracking-wider">
              Results
            </span>
          </div>
          <span className="font-display text-lg font-bold text-gradient-cyan">
            {total}
          </span>
        </div>

        <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl border border-white/[0.07] bg-void-2/40">
          <span className="font-mono text-[10px] text-white/25 uppercase tracking-wider">via</span>
          <span className="font-mono text-xs text-cyan-400/70">{METHOD_LABELS[method]}</span>
        </div>
      </motion.div>

      {/* Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3 sm:gap-4 stagger-children">
        {products.map((product, i) => (
          <ProductCard key={product.id} product={product} index={i} />
        ))}
      </div>
    </div>
  )
}
