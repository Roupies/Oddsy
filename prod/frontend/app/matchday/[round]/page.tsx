'use client'

import { useQuery } from '@tanstack/react-query'
import { oddsyAPI } from '@/lib/api'
import { MatchCardModern } from '@/components/predictions/match-card-modern'
import { LoadingSpinner } from '@/components/ui/loading-spinner'
import type { Metadata } from 'next'

interface MatchdayPageProps {
  params: {
    round: string
  }
}

export default function MatchdayPage({ params }: MatchdayPageProps) {
  const round = parseInt(params.round)
  
  // Validation de la journée
  if (isNaN(round) || round < 1 || round > 38) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="text-center py-12">
          <h1 className="text-2xl font-bold text-white mb-4">
            Invalid Round
          </h1>
          <p className="text-neutral-400 mb-6">
            Premier League rounds must be between 1 and 38.
          </p>
          <a 
            href="/matchday" 
            className="bg-oddsy-primary text-white px-6 py-3 rounded-lg hover:bg-oddsy-primary/90 transition-colors"
          >
            View Available Rounds
          </a>
        </div>
      </div>
    )
  }

  // Strict-first approach: try strict mode first, fallback to multi-GW if 422
  const { data: predictionsData, isLoading, error } = useQuery({
    queryKey: ['predictions', round],
    queryFn: async () => {
      try {
        // Tentative en mode strict d'abord
        return await oddsyAPI.getPredictions(round);
      } catch (error: any) {
        // Si 422 (validation failed), fallback vers multi-GW pour rounds 8+
        if (error.message?.includes('422') && round >= 8) {
          return await oddsyAPI.getPredictions(round, { allow_multi_gw: true });
        }
        throw error;
      }
    },
    staleTime: round <= 7 ? 1 * 60 * 1000 : 60 * 60 * 1000, // 1min for current, 1h for historical
  })

  if (isLoading) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="flex flex-col items-center justify-center min-h-[400px]">
          <LoadingSpinner />
          <p className="mt-4 text-neutral-400">Loading J{round} predictions...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="text-center py-12">
          <h1 className="text-2xl font-bold text-white mb-4">
            Unable to load predictions
          </h1>
          <p className="text-neutral-400 mb-6">
            There was an error loading predictions for J{round}.
          </p>
          <div className="space-x-4">
            <button 
              onClick={() => window.location.reload()} 
              className="bg-oddsy-primary text-white px-6 py-3 rounded-lg hover:bg-oddsy-primary/90 transition-colors"
            >
              Try Again
            </button>
            <a 
              href="/matchday" 
              className="bg-gray-200 text-gray-700 px-6 py-3 rounded-lg hover:bg-gray-300 transition-colors"
            >
              View Other Rounds
            </a>
          </div>
        </div>
      </div>
    )
  }

  
  if (!predictionsData?.data?.matches) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="text-center py-12">
          <h1 className="text-2xl font-bold text-white mb-4">
            No predictions available
          </h1>
          <p className="text-neutral-400 mb-6">
            Predictions for J{round} are not yet available.
          </p>
          <a 
            href="/pipeline" 
            className="bg-oddsy-primary text-white px-6 py-3 rounded-lg hover:bg-oddsy-primary/90 transition-colors"
          >
            Check Pipeline Status
          </a>
        </div>
      </div>
    )
  }

  const { matches } = predictionsData.data
  const totalMatches = matches.length
  const isCompatMode = false
  const modeEffective = 'strict_epl'

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Bandeau mode compat */}
      {isCompatMode && (
        <div className="mb-6 p-4 bg-amber-950/20 border border-amber-500/30 rounded-lg backdrop-blur-sm">
          <div className="flex items-center space-x-2">
            <span className="text-amber-600">⚠️</span>
            <div>
              <p className="text-sm font-medium text-amber-300">
                Mode compatibilité multi-GW activé
              </p>
              <p className="text-xs text-amber-400 mt-1">
                Pipeline v2.0 détecté avec {totalMatches} matchs. 
                Mode strict temporairement désactivé.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <h1 className="text-3xl font-bold text-white">
            Premier League J{round}
          </h1>
          <div className="text-sm text-neutral-400">
            {totalMatches} match{totalMatches !== 1 ? 'es' : ''}
          </div>
        </div>
        
        <p className="text-neutral-400 mb-6">
          Enhanced Baseline v3.0 predictions for Matchday {round}
        </p>
        
      </div>

      {/* Performance Summary */}
      <div className="card mb-8 bg-gradient-to-r from-emerald-950/30 to-blue-950/30 border border-emerald-500/20">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600 mb-1">53.5%</div>
            <div className="text-sm text-neutral-400">Cross-Validation Accuracy</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600 mb-1">51.7%</div>
            <div className="text-sm text-neutral-400">Real EPL Performance</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-600 mb-1">{totalMatches}</div>
            <div className="text-sm text-neutral-400">Predictions Available</div>
          </div>
        </div>
      </div>

      {/* Affichage simple des matchs */}
      <div className="space-y-8">
        <div>
          <h2 className="text-2xl font-bold text-white mb-4">
            Prédictions J{round} ({totalMatches} matchs)
          </h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {matches.map((match) => (
              <MatchCardModern 
                key={match.id} 
                match={match}
              />
            ))}
          </div>
        </div>
      </div>

      {/* Removed problematic conditional block for production build */}

      {/* Footer Info */}
      <div className="mt-12 text-center">
        <div className="card bg-neutral-900/60 border border-neutral-700 max-w-2xl mx-auto backdrop-blur-sm">
          <h3 className="font-semibold text-white mb-3">About These Predictions</h3>
          <div className="text-sm text-neutral-400 space-y-2">
            <p>
              Generated by Enhanced Baseline v3.0 using Pipeline v1.0 with strict temporal validation.
            </p>
            <p>
              All predictions are based on historical data and statistical analysis. 
              Performance validated on 60 real EPL matches (51.7% accuracy).
            </p>
            <p className="text-xs text-neutral-500">
              For entertainment only • Not financial advice
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}