import type { SearchRequest, SearchResponse } from '../types/search'

// ─── Config ───────────────────────────────────────────────────────────────────
const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'
const SEARCH_ENDPOINT = `${API_BASE}/api/search`

// ─── Custom Error ─────────────────────────────────────────────────────────────
export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public detail?: unknown,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

// ─── Core Search Function ────────────────────────────────────────────────────
/**
 * Sends a multipart/form-data POST to the backend search endpoint.
 * Image is sent as a raw File object — no base64 conversion.
 */
export async function searchProducts(
  request: SearchRequest,
  signal?: AbortSignal,
): Promise<SearchResponse> {
  const formData = new FormData()

  // Always include text and method
  formData.append('text', request.text.trim())
  formData.append('method', request.method)

  // Image is optional — only append if present
  if (request.image) {
    formData.append('image', request.image, request.image.name)
  }

  const response = await fetch(SEARCH_ENDPOINT, {
    method: 'POST',
    body:   formData,
    signal,
    // Do NOT set Content-Type header — browser sets multipart boundary automatically
  })

  if (!response.ok) {
    let detail: unknown
    try {
      detail = await response.json()
    } catch {
      detail = await response.text()
    }
    throw new ApiError(
      response.status,
      `Search failed: ${response.statusText}`,
      detail,
    )
  }

  const data: SearchResponse = await response.json()
  return data
}

// ─── Health Check ─────────────────────────────────────────────────────────────
export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`, { method: 'GET' })
    return res.ok
  } catch {
    return false
  }
}
