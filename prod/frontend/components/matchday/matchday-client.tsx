'use client'

import { useState, useEffect } from 'react'
import { usePredictions, useInvalidatePredictions } from '@/hooks/use-predictions'
import { MatchCard } from '@/components/predictions/match-card'
import { LoadingSpinner } from '@/components/ui/loading-spinner'
import { Button } from '@/components/ui/button'
import { RefreshCw, AlertTriangle, CheckCircle } from 'lucide-react'
import { useToastHelpers } from '@/components/ui/toast'
import { DegradedModeBanner } from '@/components/ui/degraded-mode-banner'
import type { APIResponse, RoundPredictions } from '@/lib/types'

interface MatchdayClientProps {
  round: number
  initialData?: APIResponse<RoundPredictions> & { degradedMode?: any }
  error?: string | null
}

export function MatchdayClient({ round, initialData, error: serverError }: MatchdayClientProps) {
  const [showModels, setShowModels] = useState(true)
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid')
  const { success, error: toastError, apiError, apiSuccess } = useToastHelpers()
  
  // Client-side query with server data as initial state
  const { data, isLoading, error: clientError, refetch, isFetching } = usePredictions(round)
  const { invalidateRound } = useInvalidatePredictions()
  
  // Use server data if available, fallback to client data
  const currentData = data || initialData
  const currentError = clientError || serverError
  
  // Auto-refresh for current round
  useEffect(() => {
    if (round <= 7) {
      const interval = setInterval(() => {
        invalidateRound(round)
      }, 30000) // Refresh every 30s for current round
      
      return () => clearInterval(interval)
    }
  }, [round, invalidateRound])
  
  const handleRefresh = async () => {
    try {
      invalidateRound(round)
      await refetch()
      apiSuccess(`refreshed predictions for J${round}`)
    } catch (error) {
      apiError(`refresh predictions for J${round}`, handleRefresh)
    }
  }
  
  // Loading state
  if (isLoading && !currentData) {
    return (
      <div className="flex items-center justify-center py-12">
        <LoadingSpinner />
        <span className="ml-3 text-gray-600">Loading predictions...</span>
      </div>
    )
  }
  
  // Error state
  if (currentError && !currentData) {
    return (
      <div className="card bg-red-50 border-red-200">
        <div className="flex items-center space-x-3 text-red-800">
          <AlertTriangle className="h-6 w-6" />
          <div>
            <h3 className="font-semibold">Failed to load predictions</h3>
            <p className="text-sm">{currentError}</p>
            <Button 
              onClick={handleRefresh}
              variant="outline" 
              size="sm" 
              className="mt-2"
            >
              <RefreshCw className="h-4 w-4 mr-2" />
              Retry
            </Button>
          </div>
        </div>
      </div>
    )
  }
  
  // No data state
  if (!currentData?.data) {
    return (
      <div className="card bg-yellow-50 border-yellow-200">
        <div className="flex items-center space-x-3 text-yellow-800">
          <AlertTriangle className="h-6 w-6" />
          <div>
            <h3 className="font-semibold">No predictions available</h3>
            <p className="text-sm">Predictions for J{round} are not yet available</p>
          </div>
        </div>
      </div>
    )
  }
  
  const predictions = currentData.data
  const degradedMode = (currentData as any)?.degradedMode
  
  return (
    <div className="space-y-6">
      {/* Degraded Mode Banner */}
      {degradedMode?.active && (
        <DegradedModeBanner
          mode={degradedMode.mode}
          metadata={degradedMode.metadata}
          onRetry={handleRefresh}
          showRetry={true}
          dismissible={true}
        />
      )}
      
      {/* Controls */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <input
              type="checkbox"
              id="show-models"
              checked={showModels}
              onChange={(e) => setShowModels(e.target.checked)}
              className="rounded border-gray-300"
            />
            <label htmlFor="show-models" className="text-sm text-gray-700">
              Show individual models
            </label>
          </div>
          
          <div className="flex rounded-lg bg-gray-100 p-1">
            <button
              onClick={() => setViewMode('grid')}
              className={`px-3 py-1 text-sm rounded-md transition-colors ${
                viewMode === 'grid' 
                  ? 'bg-white text-gray-900 shadow-sm' 
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              Grid
            </button>
            <button
              onClick={() => setViewMode('list')}
              className={`px-3 py-1 text-sm rounded-md transition-colors ${
                viewMode === 'list' 
                  ? 'bg-white text-gray-900 shadow-sm' 
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              List
            </button>
          </div>
        </div>
        
        <div className="flex items-center space-x-3">
          {round <= 7 && (
            <div className="flex items-center space-x-2 text-sm text-green-600">
              <CheckCircle className="h-4 w-4" />
              <span>Auto-refresh: 30s</span>
            </div>
          )}
          
          <Button
            onClick={handleRefresh}
            variant="outline"
            size="sm"
            disabled={isFetching}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${isFetching ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>
      </div>
      
      {/* Stats Summary */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="card p-4">
          <div className="text-2xl font-bold text-gray-900">
            {predictions.matches.length}
          </div>
          <div className="text-sm text-gray-600">Total Matches</div>
        </div>
        
        <div className="card p-4">
          <div className="text-2xl font-bold text-green-600">
            {predictions.matches.filter(m => m.ensemble.confidence > 0.6).length}
          </div>
          <div className="text-sm text-gray-600">High Confidence</div>
        </div>
        
        <div className="card p-4">
          <div className="text-2xl font-bold text-blue-600">
            {predictions.matches.filter(m => m.disagreement > 0.2).length}
          </div>
          <div className="text-sm text-gray-600">High Disagreement</div>
        </div>
        
        <div className="card p-4">
          <div className="text-2xl font-bold text-oddsy-primary">
            {(predictions.matches.reduce((sum, m) => sum + m.ensemble.confidence, 0) / predictions.matches.length * 100).toFixed(1)}%
          </div>
          <div className="text-sm text-gray-600">Avg Confidence</div>
        </div>
      </div>
      
      {/* Match Predictions */}
      <div className={
        viewMode === 'grid' 
          ? 'grid grid-cols-1 lg:grid-cols-2 gap-6'
          : 'space-y-4'
      }>
        {predictions.matches.map((match) => (
          <MatchCard
            key={match.id}
            match={match}
            showModels={showModels}
            compact={viewMode === 'list'}
          />
        ))}
      </div>
      
      {/* Pipeline Info */}
      {predictions.meta && (
        <div className="card bg-gray-50 p-4">
          <h3 className="font-semibold text-gray-900 mb-3">Pipeline Information</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <div>
              <span className="text-gray-600">Pipeline:</span>
              <span className="ml-2 font-medium">{predictions.meta.pipeline_version}</span>
            </div>
            <div>
              <span className="text-gray-600">Generated:</span>
              <span className="ml-2 font-medium">
                {new Date(predictions.meta.generated_at).toLocaleString()}
              </span>
            </div>
            <div>
              <span className="text-gray-600">API Version:</span>
              <span className="ml-2 font-medium">v{predictions.meta.api_version}</span>
            </div>
          </div>
          
          {predictions.meta.git_sha && (
            <div className="mt-3 text-xs text-gray-500 font-mono">
              Build: {predictions.meta.git_sha}
            </div>
          )}
        </div>
      )}
    </div>
  )
}