'use client'

import { usePredictions } from '@/hooks/use-predictions'
import { MatchCardModern } from '@/components/predictions/match-card-modern'
import { LoadingSpinner } from '@/components/ui/loading-spinner'
import { Button } from '@/components/ui/button'
import { Calendar, ArrowRight, AlertTriangle } from 'lucide-react'
import Link from 'next/link'

interface UpcomingMatchesProps {
  maxMatches?: number
  round?: number
}

export function UpcomingMatches({ maxMatches = 3, round = 7 }: UpcomingMatchesProps) {
  const { data, isLoading, error, refetch } = usePredictions(round)
  
  if (isLoading) {
    return (
      <div className="card p-6">
        <div className="flex items-center justify-center py-8">
          <LoadingSpinner />
          <span className="ml-2 text-gray-600">Loading upcoming matches...</span>
        </div>
      </div>
    )
  }
  
  if (error) {
    return (
      <div className="card p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-gray-900">Upcoming Matches</h2>
        </div>
        
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center space-x-2 text-red-800">
            <AlertTriangle className="h-5 w-5" />
            <span className="font-medium">Failed to load matches</span>
          </div>
          <p className="text-sm text-red-600 mt-1">{error.message}</p>
          <Button 
            onClick={() => refetch()}
            variant="outline" 
            size="sm" 
            className="mt-3"
          >
            Try Again
          </Button>
        </div>
      </div>
    )
  }
  
  if (!data?.data?.matches || data.data.matches.length === 0) {
    return (
      <div className="card p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-gray-900">Upcoming Matches</h2>
          <Link href={`/matchday/j${round}`}>
            <Button variant="outline" size="sm" className="gap-2">
              <Calendar className="h-4 w-4" />
              View All
            </Button>
          </Link>
        </div>
        
        <div className="text-center py-8 text-gray-500">
          <Calendar className="h-12 w-12 mx-auto mb-4 text-gray-300" />
          <p className="font-medium">No upcoming matches</p>
          <p className="text-sm">Predictions for J{round} are not yet available</p>
        </div>
      </div>
    )
  }
  
  const matches = data.data.matches.slice(0, maxMatches)
  const totalMatches = data.data.matches.length
  
  return (
    <div className="card p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <Calendar className="h-6 w-6 text-oddsy-primary" />
          <h2 className="text-xl font-bold text-gray-900">
            Upcoming Matches - J{round}
          </h2>
        </div>
        
        <Link href={`/matchday/j${round}`}>
          <Button variant="outline" size="sm" className="gap-2">
            View All {totalMatches}
            <ArrowRight className="h-4 w-4" />
          </Button>
        </Link>
      </div>
      
      {/* Match Cards */}
      <div className="space-y-4">
        {matches.map((match) => (
          <MatchCardModern
            key={match.id}
            match={match}
          />
        ))}
      </div>
      
      {/* Summary Stats */}
      {totalMatches > maxMatches && (
        <div className="mt-6 pt-4 border-t border-gray-200">
          <div className="flex items-center justify-between text-sm text-gray-600">
            <span>Showing {maxMatches} of {totalMatches} matches</span>
            <span>
              Avg confidence: {
                (matches.reduce((sum, m) => sum + m.ensemble.confidence, 0) / matches.length * 100).toFixed(1)
              }%
            </span>
          </div>
        </div>
      )}
      
      {/* Quick Insights */}
      <div className="mt-4 grid grid-cols-1 sm:grid-cols-3 gap-3">
        <div className="bg-green-50 rounded-lg p-3">
          <div className="text-lg font-bold text-green-600">
            {matches.filter(m => m.ensemble.confidence > 0.6).length}
          </div>
          <div className="text-xs text-green-700">High Confidence</div>
        </div>
        
        <div className="bg-blue-50 rounded-lg p-3">
          <div className="text-lg font-bold text-blue-600">
            {matches.filter(m => m.disagreement > 0.2).length}
          </div>
          <div className="text-xs text-blue-700">High Disagreement</div>
        </div>
        
        <div className="bg-purple-50 rounded-lg p-3">
          <div className="text-lg font-bold text-purple-600">
            {matches.filter(m => m.ensemble.prediction === 'DRAW').length}
          </div>
          <div className="text-xs text-purple-700">Draw Predictions</div>
        </div>
      </div>
      
      {/* Pipeline Info */}
      {data.data.meta && (
        <div className="mt-4 pt-4 border-t border-gray-200">
          <div className="flex items-center justify-between text-xs text-gray-500">
            <span>Pipeline: {data.data.meta.pipeline_version}</span>
            <span>Generated: {new Date(data.data.meta.generated_at).toLocaleString()}</span>
          </div>
        </div>
      )}
    </div>
  )
}