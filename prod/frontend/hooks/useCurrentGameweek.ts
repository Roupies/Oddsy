'use client'

import { useQuery } from '@tanstack/react-query'
import { oddsyAPI } from '@/lib/api'
import { APP_CONFIG, getDynamicMetrics } from '@/config/appConstants'

interface GameweekData {
  currentGameweek: number
  nextGameweek: number
  totalPredictions: number
  validatedMatches: number
  currentGameweekLabel: string
  nextGameweekLabel: string
  upcomingMatches: Array<{
    id: string
    homeTeam: string
    awayTeam: string
    stadium: string
    homeWinProb: number
    drawProb: number
    awayWinProb: number
    confidence: 'high' | 'medium' | 'low'
    prediction: 'home' | 'draw' | 'away'
    kickoff: string
  }>
}

const mapConfidence = (prob: number): 'high' | 'medium' | 'low' => {
  if (prob >= 60) return 'high'
  if (prob >= 45) return 'medium'
  return 'low'
}

const getPredictionType = (homeProb: number, drawProb: number, awayProb: number): 'home' | 'draw' | 'away' => {
  const maxProb = Math.max(homeProb, drawProb, awayProb)
  if (maxProb === homeProb) return 'home'
  if (maxProb === awayProb) return 'away'
  return 'draw'
}

const formatKickoff = (kickoffUtc: string): string => {
  try {
    // Parse ISO 8601 UTC datetime
    const date = new Date(kickoffUtc)
    const dayName = date.toLocaleDateString('en-US', { weekday: 'long' })
    const time = date.toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit',
      hour12: false 
    })
    return `${dayName} ${time}`
  } catch {
    return 'TBD'
  }
}

export const useCurrentGameweek = (): {
  data: GameweekData | undefined
  isLoading: boolean
  error: Error | null
} => {
  const { data: predictionsData, isLoading, error } = useQuery({
    queryKey: ['currentGameweek'],
    queryFn: async () => {
      // Use API v5 for better reliability
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
      const latestResponse = await fetch(`${apiUrl}/api/v5/gameweeks/latest`)
      if (!latestResponse.ok) {
        throw new Error('Failed to fetch latest gameweek')
      }
      const latestData = await latestResponse.json()
      const latestGameweek = latestData.data.latest_gameweek
      
      // Fetch both predictions and fixtures in parallel
      const [predResponse, fixturesResponse] = await Promise.all([
        fetch(`${apiUrl}/api/v5/gameweeks/${latestGameweek}/predictions`),
        fetch(`${apiUrl}/api/v5/gameweeks/${latestGameweek}/fixtures`)
      ])
      
      if (!predResponse.ok) {
        throw new Error('Failed to fetch predictions')
      }
      
      const predictions = await predResponse.json()
      let fixtures = null
      
      // Fixtures are optional - don't fail if they're not available
      if (fixturesResponse.ok) {
        fixtures = await fixturesResponse.json()
      }
      
      return { predictions, fixtures }
    },
    staleTime: APP_CONFIG.PREDICTIONS_CACHE_TIME,
    retry: 2,
  })

  const gameweekData: GameweekData | undefined = predictionsData ? (() => {
    // Extract gameweek from v5 API response
    const currentGameweek = predictionsData.predictions?.gameweek || 11
    const metrics = getDynamicMetrics(currentGameweek)
    
    // Get fixtures data for kickoff times
    const fixtures = predictionsData.fixtures?.data?.fixtures || []
    const fixtureMap = new Map()
    fixtures.forEach((fixture: any) => {
      const key = `${fixture.home_team}_vs_${fixture.away_team}`
      fixtureMap.set(key, fixture)
    })
    
    // Convert v5 predictions format to component format
    const predictions = predictionsData.predictions?.predictions || {}
    const upcomingMatches = Object.entries(predictions).slice(0, 3).map(([matchKey, match]: [string, any], index: number) => {
      const matchInfo = match.match_info
      const probs = match.probabilities
      
      // Try to find corresponding fixture for real kickoff time
      const fixture = fixtureMap.get(matchKey)
      const kickoffTime = fixture?.kickoff_utc || new Date().toISOString()
      
      return {
        id: matchKey,
        homeTeam: matchInfo.home,
        awayTeam: matchInfo.away,
        stadium: fixture?.location || `${matchInfo.home} Stadium`,
        homeWinProb: Math.round(probs.home * 100),
        drawProb: Math.round(probs.draw * 100),
        awayWinProb: Math.round(probs.away * 100),
        confidence: mapConfidence(Math.max(probs.home * 100, probs.draw * 100, probs.away * 100)),
        prediction: match.prediction === 'H' ? 'home' : match.prediction === 'A' ? 'away' : 'draw',
        kickoff: formatKickoff(kickoffTime)
      }
    })

    return {
      currentGameweek,
      nextGameweek: currentGameweek + 1,
      ...metrics,
      upcomingMatches
    }
  })() : undefined

  return {
    data: gameweekData,
    isLoading,
    error: error as Error | null
  }
}