# NEXUS · Multimodal AI Product Search

A production-ready React + TypeScript frontend for a multimodal AI product search system. Supports image-based, text-based, and fused retrieval strategies powered by CLIP embeddings.

## Tech Stack

| Layer       | Technology                           |
|-------------|--------------------------------------|
| Framework   | React 18 + TypeScript                |
| Build Tool  | Vite 5                               |
| Styling     | Tailwind CSS v3 + custom theme       |
| Animation   | Framer Motion 11                     |
| Fonts       | Orbitron · DM Sans · JetBrains Mono  |
| API         | FastAPI backend (separate repo)      |

## Features

- **Image Upload** — drag-and-drop or click to browse, live preview
- **Text Query** — multi-line natural language input with character counter
- **Three Retrieval Methods:**
  - `Symbolic Early Fusion` — CLIP embeddings + NLP constraints merged before SQL generation
  - `Late Fusion` — independent image/text retrieval merged via reciprocal rank fusion
  - `Intent-Based Search` — NLU infers intent to generate dynamic query constraints
- **Results Grid** — animated product cards with CLIP, text, and final score bars
- **AI Diagnostics Console** — collapsible panel: image signals, text constraints, generated SQL, fusion stats
- **Loading & Error States** — multi-ring spinner, animated progress bar, dismissible error cards

## Getting Started

```bash
# Install dependencies
npm install

# Copy and configure environment
cp .env.example .env
# Edit .env — set VITE_API_BASE_URL to your FastAPI backend

# Start dev server (http://localhost:3000)
npm run dev

# Production build
npm run build
```

## Project Structure

```
src/
├── components/
│   ├── ImageUploader.tsx     # Drag-and-drop image input with preview
│   ├── SearchInput.tsx       # Textarea with animated focus ring
│   ├── MethodSelector.tsx    # Three-method radio selector with glow
│   ├── SearchButton.tsx      # Animated CTA with loading state
│   ├── ResultsGrid.tsx       # Product grid with header metrics
│   ├── ProductCard.tsx       # Product card with CLIP/text/final score bars
│   └── DiagnosticsPanel.tsx  # Collapsible AI diagnostics console
├── services/
│   └── api.ts                # multipart/form-data POST, error handling
├── types/
│   └── search.ts             # Full TypeScript type definitions
├── App.tsx                   # Root component, search state management
├── main.tsx                  # React entry point
└── index.css                 # Tailwind base + custom utilities + animations
```

## API Contract

`POST /api/search` as `multipart/form-data`:

| Field    | Type   | Notes                                              |
|----------|--------|----------------------------------------------------|
| `text`   | string | Natural language query (required if no image)      |
| `method` | string | `symbolic_early` / `late_fusion` / `intent_based`  |
| `image`  | File   | Product reference image (required if no text)      |

See `src/types/search.ts → SearchResponse` for the full response shape.

## Environment Variables

| Variable            | Default                  | Description       |
|---------------------|--------------------------|-------------------|
| `VITE_API_BASE_URL` | `http://localhost:8000`  | Backend base URL  |
