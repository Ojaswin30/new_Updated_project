/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        'void':    '#0b0f1a',
        'void-2':  '#0d1220',
        'void-3':  '#111827',
        'panel':   'rgba(13, 18, 32, 0.8)',
        'glass':   'rgba(255, 255, 255, 0.04)',
        'glass-2': 'rgba(255, 255, 255, 0.07)',
        'cyan': {
          DEFAULT: '#00e5ff',
          400: '#22d3ee',
          500: '#06b6d4',
          glow: 'rgba(0, 229, 255, 0.15)',
        },
        'violet': {
          DEFAULT: '#a855f7',
          400: '#c084fc',
          glow: 'rgba(168, 85, 247, 0.15)',
        },
        'neon-green': '#39ff14',
      },
      fontFamily: {
        display: ['Orbitron', 'monospace'],
        body:    ['DM Sans', 'sans-serif'],
        mono:    ['JetBrains Mono', 'monospace'],
      },
      backgroundImage: {
        'gradient-radial':   'radial-gradient(var(--tw-gradient-stops))',
        'gradient-conic':    'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))',
        'cyber-grid':
          'linear-gradient(rgba(0,229,255,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(0,229,255,0.03) 1px, transparent 1px)',
        'hero-glow':
          'radial-gradient(ellipse 80% 50% at 50% -10%, rgba(0,229,255,0.12), transparent)',
      },
      backgroundSize: {
        'grid': '40px 40px',
      },
      boxShadow: {
        'neon-cyan':   '0 0 20px rgba(0, 229, 255, 0.3), 0 0 60px rgba(0, 229, 255, 0.1)',
        'neon-violet': '0 0 20px rgba(168, 85, 247, 0.3), 0 0 60px rgba(168, 85, 247, 0.1)',
        'neon-sm':     '0 0 10px rgba(0, 229, 255, 0.2)',
        'glass':       '0 8px 32px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255,255,255,0.05)',
        'card':        '0 4px 24px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255,255,255,0.04)',
        'card-hover':  '0 8px 40px rgba(0, 0, 0, 0.6), 0 0 30px rgba(0, 229, 255, 0.08)',
      },
      borderColor: {
        'glass': 'rgba(255, 255, 255, 0.08)',
        'cyan-dim': 'rgba(0, 229, 255, 0.3)',
        'violet-dim': 'rgba(168, 85, 247, 0.3)',
      },
      animation: {
        'float':      'float 6s ease-in-out infinite',
        'pulse-slow': 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'scan':       'scan 2s linear infinite',
        'glow-pulse': 'glowPulse 2s ease-in-out infinite',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%':      { transform: 'translateY(-8px)' },
        },
        scan: {
          '0%':   { transform: 'translateX(-100%)' },
          '100%': { transform: 'translateX(400%)' },
        },
        glowPulse: {
          '0%, 100%': { boxShadow: '0 0 10px rgba(0,229,255,0.2)' },
          '50%':      { boxShadow: '0 0 25px rgba(0,229,255,0.5), 0 0 50px rgba(0,229,255,0.2)' },
        },
      },
    },
  },
  plugins: [],
}
