// ─── Search Method ────────────────────────────────────────────────────────────
export type SearchMethod = 'symbolic_early' | 'late_fusion' | 'intent_based'

export interface SearchMethodOption {
  id:          SearchMethod
  label:       string
  shortLabel:  string
  description: string
  icon:        string
  gradient:    string
}

// ─── API Request ──────────────────────────────────────────────────────────────
export interface SearchRequest {
  text:   string
  method: SearchMethod
  image?: File
}

// ─── Product Result ───────────────────────────────────────────────────────────
export interface Product {
  id:          number | string
  title:       string
  price:       number
  rating:      number
  image_url:   string
  category?:   string
  description?: string
  final_score: number
  // Fusion-specific scores
  clip_score?:    number
  text_score?:    number
  fusion_score?:  number
}

// ─── AI Diagnostics ───────────────────────────────────────────────────────────
export interface ImageSignal {
  categories?:   string[]
  colors?:        string[]
  style?:         string
  attributes?:    Record<string, string | number>
  embedding_dim?: number
  raw?:           Record<string, unknown>
}

export interface TextConstraints {
  keywords?:   string[]
  filters?:    Record<string, string | number>
  category?:   string
  price_range?: { min?: number; max?: number }
  raw?:        Record<string, unknown>
}

export interface FusionStats {
  method:              string
  image_weight?:       number
  text_weight?:        number
  candidates_ranked?:  number
  fusion_time_ms?:     number
  raw?:                Record<string, unknown>
}

export interface Diagnostics {
  image_signal?:     ImageSignal
  text_constraints?: TextConstraints
  final_constraints?: Record<string, unknown>
  generated_sql?:    string
  fusion_stats?:     FusionStats
  processing_time_ms?: number
}

// ─── API Response ─────────────────────────────────────────────────────────────
export interface SearchResponse {
  results:     Product[]
  diagnostics: Diagnostics
  total:       number
  query_id?:   string
}

// ─── App State ────────────────────────────────────────────────────────────────
export type SearchStatus = 'idle' | 'loading' | 'success' | 'error'

export interface SearchState {
  status:       SearchStatus
  results:      Product[]
  diagnostics:  Diagnostics | null
  error:        string | null
  totalResults: number
}
