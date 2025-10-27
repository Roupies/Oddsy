import { Metadata } from 'next'
import { PerformanceDashboard } from '@/components/performance/performance-dashboard'

export const metadata: Metadata = {
  title: 'Performance Dashboard | Oddsy',
  description: 'Analyse des performances de prédiction et comparaison avec le marché',
}

export default function PerformancePage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      <div className="container mx-auto px-4 py-8">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">
            Performance Dashboard
          </h1>
          <p className="text-slate-300 text-lg">
            Analyse détaillée des performances de prédiction et comparaison avec les marchés
          </p>
        </div>
        
        <PerformanceDashboard />
      </div>
    </div>
  )
}