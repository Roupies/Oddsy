/**
 * Predictions React Query Hooks for Oddsy
 * ========================================
 * 
 * This module provides React Query hooks for fetching and managing
 * football match predictions from the Oddsy API. Includes caching,
 * revalidation strategies, and error handling.
 */

'use client'

// React Query hooks for data fetching and caching
import { useQuery, useQueryClient } from '@tanstack/react-query'

// Oddsy API client for backend communication
import { oddsyAPI } from '@/lib/api'

// TypeScript types for API responses
import type { APIResponse, RoundPredictions } from '@/lib/types'

// Cache management utilities
import { CACHE_KEYS, shouldRevalidate } from '@/lib/types'

/**
 * Hook to fetch predictions for a specific gameweek
 * 
 * Provides caching and revalidation strategies based on whether
 * the round is current/future (frequent updates) or historical (stable).
 * 
 * @param round - Gameweek number (1-38)
 * @param season - Season identifier (default: '2025-26')
 * @returns React Query result with predictions data
 */
export function usePredictions(round: number, season: string = '2025-26') {
  return useQuery({
    queryKey: CACHE_KEYS.predictions(season, round),
    queryFn: () => oddsyAPI.getPredictions(round),
    enabled: round >= 1 && round <= 38,
    staleTime: shouldRevalidate(round) ? 30000 : 300000, // 30s current, 5min historical
    refetchOnWindowFocus: shouldRevalidate(round),        // Refetch current rounds on focus
    retry: (failureCount, error) => {
      // Don't retry on 404 (round not available yet)
      if (error instanceof Error && error.message.includes('404')) {
        return false
      }
      return failureCount < 2  // Max 2 retries for other errors
    }
  })
}

/**
 * Hook to fetch list of available gameweeks
 * 
 * Returns metadata about all gameweeks with available predictions,
 * including file sizes, generation timestamps, and status.
 * 
 * @returns React Query result with available rounds data
 */
export function useAvailableRounds() {
  return useQuery({
    queryKey: CACHE_KEYS.available_rounds(),
    queryFn: () => oddsyAPI.getAvailableRounds(),
    staleTime: 5 * 60 * 1000, // 5 minutes cache time
    refetchOnWindowFocus: false   // Don't refetch on window focus (stable data)
  })
}

/**
 * Hook to prefetch predictions for a gameweek
 * 
 * Useful for preloading predictions when user hovers over navigation
 * or when preparing to show adjacent gameweeks.
 * 
 * @returns Function to trigger prefetch for specific round
 */
export function usePrefetchPredictions() {
  const queryClient = useQueryClient()

  /**
   * Prefetch predictions for a specific round
   * 
   * @param round - Gameweek number to prefetch
   * @param season - Season identifier
   */
  return (round: number, season: string = '2025-26') => {
    queryClient.prefetchQuery({
      queryKey: CACHE_KEYS.predictions(season, round),
      queryFn: () => oddsyAPI.getPredictions(round),
      staleTime: shouldRevalidate(round) ? 30000 : 300000  // Same cache strategy
    })
  }
}

/**
 * Hook to manually invalidate prediction cache
 * 
 * Useful for forcing refetch when predictions are updated
 * or when switching between different data sources.
 * 
 * @returns Object with invalidation functions
 */
export function useInvalidatePredictions() {
  const queryClient = useQueryClient()

  return {
    /**
     * Invalidate cache for a specific round
     * 
     * @param round - Gameweek number to invalidate
     * @param season - Season identifier
     */
    invalidateRound: (round: number, season: string = '2025-26') => {
      queryClient.invalidateQueries({
        queryKey: CACHE_KEYS.predictions(season, round)
      })
    },
    /**
     * Invalidate all prediction caches
     * Use sparingly as this will refetch all data
     */
    invalidateAll: () => {
      queryClient.invalidateQueries({
        queryKey: ['predictions']
      })
    }
  }
}