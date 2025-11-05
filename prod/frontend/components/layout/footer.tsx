'use client'

import { useCurrentGameweek } from '@/hooks/useCurrentGameweek'
import { APP_CONFIG } from '@/config/appConstants'

export function Footer() {
  const { data: gameweekData } = useCurrentGameweek()
  return (
    <footer className="relative overflow-hidden">
      {/* Premium Background with EPL Gradient */}
      <div className="absolute inset-0 epl-gradient opacity-90"></div>
      <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent"></div>
      
      {/* Animated Background Elements */}
      <div className="absolute inset-0 opacity-10">
        <div className="absolute top-8 left-1/4 w-2 h-2 bg-white rounded-full animate-float"></div>
        <div className="absolute top-16 right-1/3 w-1 h-1 bg-oddsy-secondary rounded-full animate-bounce delay-300"></div>
        <div className="absolute bottom-12 left-1/3 w-3 h-3 bg-white/50 rounded-full animate-pulse delay-700"></div>
        <div className="absolute bottom-8 right-1/4 w-1 h-1 bg-oddsy-accent rounded-full animate-float delay-500"></div>
      </div>
      
      <div className="relative z-10 container mx-auto px-4 py-12">
        <div className="text-center space-y-8">
          {/* Main Logo & Version */}
          <div className="space-y-2">
            <div className="text-2xl font-bold text-white">
              <span className="text-gradient-animate">⚽ Oddsy v3.0</span>
            </div>
            <div className="text-oddsy-secondary font-medium">
              {APP_CONFIG.MODEL_VERSION} • {APP_CONFIG.PIPELINE_VERSION}
            </div>
          </div>
          
          {/* Performance Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 max-w-2xl mx-auto">
            <div className="glass-effect-subtle rounded-lg p-4">
              <div className="text-lg font-bold text-oddsy-secondary">{APP_CONFIG.MODEL_ACCURACY}</div>
              <div className="text-sm text-white/90">{APP_CONFIG.MODEL_VERSION}</div>
            </div>
            <div className="glass-effect-subtle rounded-lg p-4">
              <div className="text-lg font-bold text-oddsy-accent">{APP_CONFIG.MODEL_ACCURACY}</div>
              <div className="text-sm text-white/90">Real EPL Accuracy</div>
            </div>
            <div className="glass-effect-subtle rounded-lg p-4">
              <div className="text-lg font-bold text-white">{gameweekData?.totalPredictions || 110}+</div>
              <div className="text-sm text-white/90">Predictions Generated</div>
            </div>
          </div>
          
          {/* Trust Indicators */}
          <div className="flex flex-wrap justify-center gap-4">
            <div className="badge-premium text-white border-white/30">
              <span className="text-oddsy-secondary">🎯</span>
              <span>Validated on {gameweekData?.validatedMatches || 100} Real EPL Matches</span>
            </div>
            <div className="badge-premium text-white border-white/30">
              <span className="text-oddsy-secondary">🔒</span>
              <span>Anti-Data Leakage Protection</span>
            </div>
            <div className="badge-premium text-white border-white/30">
              <span className="text-oddsy-secondary">⚡</span>
              <span>Real-time Updates</span>
            </div>
          </div>
          
          {/* Quick Links */}
          <div className="flex flex-wrap justify-center gap-6 text-sm">
            <a href="/matchday" className="text-white/80 hover:text-oddsy-secondary transition-colors duration-200 hover:underline">
              Latest Predictions
            </a>
            <a href="/pipeline" className="text-white/80 hover:text-oddsy-secondary transition-colors duration-200 hover:underline">
              Pipeline Status
            </a>
            <a href="/models" className="text-white/80 hover:text-oddsy-secondary transition-colors duration-200 hover:underline">
              Model Performance
            </a>
          </div>
          
          {/* Disclaimer */}
          <div className="border-t border-white/20 pt-6 space-y-2">
            <div className="text-sm text-white/80">
              Predictions powered by Enhanced Baseline v3.0 • Pipeline v1.0
            </div>
            <div className="text-xs text-white/60">
              AI-generated predictions • Not financial advice • For entertainment only
            </div>
            <div className="text-xs text-white/50">
              © 2024 Oddsy • Built with Next.js & Tailwind CSS
            </div>
          </div>
        </div>
      </div>
    </footer>
  )
}