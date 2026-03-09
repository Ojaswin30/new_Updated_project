import { useRef, useState, useCallback, type DragEvent, type ChangeEvent } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

interface Props {
  image: File | null
  onImageChange: (file: File | null) => void
}

export default function ImageUploader({ image, onImageChange }: Props) {
  const inputRef           = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const [preview, setPreview]   = useState<string | null>(null)

  // Generate preview URL when file is selected
  const handleFile = useCallback((file: File | null) => {
    if (!file) {
      setPreview(null)
      onImageChange(null)
      return
    }
    if (!file.type.startsWith('image/')) return
    onImageChange(file)
    const url = URL.createObjectURL(file)
    setPreview(prev => {
      if (prev) URL.revokeObjectURL(prev)
      return url
    })
  }, [onImageChange])

  const handleDrop = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }, [handleFile])

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    handleFile(e.target.files?.[0] ?? null)
  }

  const handleRemove = () => {
    handleFile(null)
    if (inputRef.current) inputRef.current.value = ''
  }

  return (
    <div className="space-y-3">
      <label className="flex items-center gap-2 text-xs font-mono text-cyan-400/60 uppercase tracking-[0.2em]">
        <span className="w-1.5 h-1.5 rounded-full bg-cyan-400/60 block" />
        Image Input · Optional
      </label>

      <AnimatePresence mode="wait">
        {!image ? (
          /* ── Drop Zone ── */
          <motion.div
            key="dropzone"
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.96 }}
            transition={{ duration: 0.2 }}
            onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            onClick={() => inputRef.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
            className={`
              relative overflow-hidden cursor-pointer rounded-2xl border-2 border-dashed p-8
              flex flex-col items-center justify-center gap-4 min-h-[180px]
              transition-all duration-300 select-none
              ${dragging
                ? 'border-cyan-400/80 bg-cyan-400/5 shadow-neon-cyan'
                : 'border-white/10 bg-white/[0.02] hover:border-cyan-400/40 hover:bg-cyan-400/[0.03]'
              }
            `}
            aria-label="Upload image — drag and drop or click to browse"
          >
            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={handleChange}
              aria-hidden="true"
            />

            {/* Animated scan line on drag */}
            {dragging && (
              <motion.div
                className="absolute inset-x-0 h-px bg-gradient-to-r from-transparent via-cyan-400 to-transparent"
                animate={{ y: ['0%', '18rem'] }}
                transition={{ duration: 1.5, repeat: Infinity, ease: 'linear' }}
              />
            )}

            {/* Upload icon */}
            <motion.div
              animate={dragging ? { scale: 1.1, y: -4 } : { scale: 1, y: 0 }}
              transition={{ type: 'spring', stiffness: 300 }}
              className="relative"
            >
              <div className={`
                w-16 h-16 rounded-2xl flex items-center justify-center
                border transition-all duration-300
                ${dragging
                  ? 'border-cyan-400/50 bg-cyan-400/10'
                  : 'border-white/10 bg-white/[0.03]'}
              `}>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
                  className={`w-7 h-7 transition-colors duration-300 ${dragging ? 'text-cyan-400' : 'text-white/30'}`}
                  strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14"/>
                  <path d="M14 8h.01"/>
                  <path d="M4 4h16a1 1 0 011 1v14a1 1 0 01-1 1H4a1 1 0 01-1-1V5a1 1 0 011-1z"/>
                </svg>
              </div>
              {dragging && (
                <motion.div
                  className="absolute inset-0 rounded-2xl border border-cyan-400/30"
                  animate={{ scale: [1, 1.4], opacity: [0.6, 0] }}
                  transition={{ duration: 0.8, repeat: Infinity }}
                />
              )}
            </motion.div>

            <div className="text-center">
              <p className="text-sm font-body text-white/50">
                <span className="text-cyan-400/80 font-medium">Drop image</span> or click to browse
              </p>
              <p className="text-xs font-mono text-white/25 mt-1">PNG · JPG · WEBP · AVIF</p>
            </div>
          </motion.div>

        ) : (
          /* ── Preview ── */
          <motion.div
            key="preview"
            initial={{ opacity: 0, scale: 0.96 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.96 }}
            transition={{ duration: 0.25, type: 'spring', stiffness: 300 }}
            className="relative rounded-2xl overflow-hidden border border-cyan-400/20 shadow-neon-sm group"
          >
            <img
              src={preview ?? ''}
              alt="Upload preview"
              className="w-full h-52 object-cover"
            />

            {/* Overlay info */}
            <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-transparent to-transparent" />

            <div className="absolute bottom-0 left-0 right-0 p-3 flex items-end justify-between">
              <div>
                <p className="font-mono text-xs text-cyan-400/80">
                  {image.name.length > 32 ? image.name.slice(0, 30) + '…' : image.name}
                </p>
                <p className="font-mono text-[10px] text-white/30 mt-0.5">
                  {(image.size / 1024).toFixed(1)} KB · {image.type.split('/')[1].toUpperCase()}
                </p>
              </div>

              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={handleRemove}
                className="
                  px-3 py-1.5 rounded-xl text-xs font-mono
                  bg-black/50 border border-white/10 text-white/60
                  hover:border-red-400/50 hover:text-red-400 hover:bg-red-400/10
                  transition-all duration-200 backdrop-blur-sm
                "
              >
                ✕ Remove
              </motion.button>
            </div>

            {/* Neon corner accents */}
            <div className="absolute top-0 left-0 w-5 h-5 border-t-2 border-l-2 border-cyan-400/50 rounded-tl-2xl" />
            <div className="absolute top-0 right-0 w-5 h-5 border-t-2 border-r-2 border-cyan-400/50 rounded-tr-2xl" />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
