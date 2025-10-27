import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './hooks/**/*.{js,ts,jsx,tsx}',
    './lib/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        // Editorial neutral palette
        neutral: {
          50: '#fafafa',
          100: '#f5f5f5',
          200: '#e5e5e5',
          300: '#d4d4d4',
          400: '#a3a3a3',
          500: '#737373',
          600: '#525252',
          700: '#404040',
          800: '#262626',
          900: '#171717',
          950: '#0a0a0a', // Deep black for editorial backgrounds
        },
        // EPL Brand Colors - Extended Palette
        'epl-purple': {
          DEFAULT: '#37003c',
          light: '#4a0050',
          dark: '#2a0030',
          50: '#f5f1f6',
          100: '#ebe3ec',
          200: '#d6c7d9',
          300: '#c1abc6',
          400: '#9673a0',
          500: '#6b3b7a',
          600: '#5d326a',
          700: '#4e2959',
          800: '#3f2048',
          900: '#37003c',
        },
        'epl-green': {
          DEFAULT: '#00ff87',
          light: '#33ff9f',
          dark: '#00cc6c',
          50: '#f0fdf7',
          100: '#e1fcef',
          200: '#c4f9df',
          300: '#a6f6cf',
          400: '#6bf0af',
          500: '#30ea8f',
          600: '#00ff87',
          700: '#00e676',
          800: '#00cc6c',
          900: '#00b359',
        },
        'epl-pink': {
          DEFAULT: '#e90052',
          light: '#ed3370',
          dark: '#c70045',
          50: '#fef2f5',
          100: '#fde6eb',
          200: '#faccdb',
          300: '#f7b3cb',
          400: '#f080ab',
          500: '#ea4d8b',
          600: '#e90052',
          700: '#d1004a',
          800: '#b90041',
          900: '#a10038',
        },
        // Custom brand - Enhanced
        'oddsy-primary': {
          DEFAULT: '#37003c',
          light: '#4a0050',
          dark: '#2a0030',
        },
        'oddsy-secondary': {
          DEFAULT: '#00ff87',
          light: '#33ff9f',
          dark: '#00cc6c',
        },
        'oddsy-accent': {
          DEFAULT: '#e90052',
          light: '#ed3370',
          dark: '#c70045',
        },
        // Editorial emerald (main accent)
        emerald: {
          50: '#ecfdf5',
          100: '#d1fae5',
          200: '#a7f3d0',
          300: '#6ee7b7',
          400: '#34d399',
          500: '#10b981',
          600: '#059669',
          700: '#047857',
          800: '#065f46',
          900: '#064e3b',
          950: '#022c22',
        },
        // Glassmorphism support
        'glass': {
          white: 'rgba(255, 255, 255, 0.1)',
          black: 'rgba(0, 0, 0, 0.1)',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Menlo', 'Monaco', 'monospace'],
        display: ['Inter', 'system-ui', 'sans-serif'], // Clean editorial, pas de Bebas
      },
      fontSize: {
        '2xs': ['0.625rem', { lineHeight: '0.75rem' }],
        '5xl': ['3rem', { lineHeight: '1' }],
        '6xl': ['3.75rem', { lineHeight: '1' }],
        '7xl': ['4.5rem', { lineHeight: '1' }],
        '8xl': ['6rem', { lineHeight: '1' }],
        '9xl': ['8rem', { lineHeight: '1' }],
      },
      spacing: {
        '18': '4.5rem',
        '88': '22rem',
        '100': '25rem',
        '112': '28rem',
      },
      backdropBlur: {
        'xs': '2px',
        '2xl': '40px',
        '3xl': '64px',
      },
      boxShadow: {
        '3xl': '0 35px 60px -12px rgba(0, 0, 0, 0.25), 0 8px 25px -8px rgba(0, 0, 0, 0.1)',
        'glow': '0 0 20px rgba(0, 255, 135, 0.3)',
        'glow-lg': '0 0 40px rgba(0, 255, 135, 0.4)',
        'epl': '0 10px 30px rgba(55, 0, 60, 0.3)',
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-in-out',
        'slide-up': 'slideUp 0.3s ease-out',
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'pulse-editorial': 'pulseEditorial 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'fadeUp': 'fadeUp 0.6s ease-out forwards',
        'slideUp': 'slideUp 0.4s ease-out forwards',
        'progressBar': 'progressBar 1.5s ease-out 0.8s forwards',
        'shimmer': 'shimmer 2s linear infinite',
        'float': 'float 6s ease-in-out infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
        'bounce-gentle': 'bounceGentle 2s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        fadeUp: {
          '0%': { opacity: '0', transform: 'translateY(20px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        progressBar: {
          '0%': { width: '0%' },
          '100%': { width: 'var(--target-width)' },
        },
        shimmer: {
          '0%': { transform: 'translateX(-100%)' },
          '100%': { transform: 'translateX(100%)' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-20px)' },
        },
        glow: {
          '0%': { boxShadow: '0 0 20px rgba(0, 255, 135, 0.3)' },
          '100%': { boxShadow: '0 0 40px rgba(0, 255, 135, 0.6)' },
        },
        bounceGentle: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-10px)' },
        },
      },
      gradientColorStops: {
        'epl-gradient': {
          '0%': '#37003c',
          '50%': '#00ff87',
          '100%': '#e90052',
        },
      },
      scale: {
        '102': '1.02',
        '103': '1.03',
      },
      transitionDuration: {
        '400': '400ms',
        '600': '600ms',
        '800': '800ms',
        '1200': '1200ms',
      },
      transitionTimingFunction: {
        'bounce-in': 'cubic-bezier(0.68, -0.55, 0.265, 1.55)',
        'ease-out-quart': 'cubic-bezier(0.25, 1, 0.5, 1)',
      },
    },
  },
  plugins: [],
}

export default config