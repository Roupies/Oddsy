'use client'

import { useQuery, useQueryClient } from '@tanstack/react-query'
import { oddsyAPI } from '@/lib/api'
import type { APIResponse, RoundPredictions } from '@/lib/types'
import { CACHE_KEYS, shouldRevalidate } from '@/lib/types'

/**
 * Hook pour récupérer prédictions d'une journée
 */
export function usePredictions(round: number, season: string = '2025-26') {
  return useQuery({
    queryKey: CACHE_KEYS.predictions(season, round),
    queryFn: () => oddsyAPI.getPredictions(round),
    enabled: round >= 1 && round <= 38,
    staleTime: shouldRevalidate(round) ? 30000 : 300000, // 30s current, 5min historical
    refetchOnWindowFocus: shouldRevalidate(round),
    retry: (failureCount, error) => {
      // Don't retry on 404 (round not available)
      if (error instanceof Error && error.message.includes('404')) {
        return false
      }
      return failureCount < 2
    }
  })
}

/**
 * Hook pour récupérer journées disponibles
 */
export function useAvailableRounds() {
  return useQuery({
    queryKey: CACHE_KEYS.available_rounds(),
    queryFn: () => oddsyAPI.getAvailableRounds(),
    staleTime: 5 * 60 * 1000, // 5 minutes
    refetchOnWindowFocus: false
  })
}

/**
 * Hook pour précharger une journée
 */
export function usePrefetchPredictions() {
  const queryClient = useQueryClient()

  return (round: number, season: string = '2025-26') => {
    queryClient.prefetchQuery({
      queryKey: CACHE_KEYS.predictions(season, round),
      queryFn: () => oddsyAPI.getPredictions(round),
      staleTime: shouldRevalidate(round) ? 30000 : 300000
    })
  }
}

/**
 * Hook pour invalider cache prédictions
 */
export function useInvalidatePredictions() {
  const queryClient = useQueryClient()

  return {
    invalidateRound: (round: number, season: string = '2025-26') => {
      queryClient.invalidateQueries({
        queryKey: CACHE_KEYS.predictions(season, round)
      })
    },
    invalidateAll: () => {
      queryClient.invalidateQueries({
        queryKey: ['predictions']
      })
    }
  }
}