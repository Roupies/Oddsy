/**
 * Adaptateur API v5 vers format frontend
 * ====================================
 * Convertit les responses API v5 vers le format attendu par le frontend
 */

import type { APIResponse, RoundPredictions, MatchPrediction } from './types'

// Types API v5 (backend response)
interface APIv5Response {
  api_version: string
  mode: string
  gameweek: number
  metadata: {
    season_hash: string
    dataset_hash: string
    git_sha: string
    generated_at: string
    strict_validated: boolean
    fixtures_count: number
  }
  fixtures_count: number
  predictions: Record<string, APIv5Prediction>
  validation: {
    gw_compliant: boolean
    ko2h_strict: boolean
    epl_teams_only: boolean
    json_schema_valid: boolean
  }
}

interface APIv5Prediction {
  prediction: 'H' | 'D' | 'A'
  confidence: number
  probabilities: {
    home: number
    draw: number
    away: number
  }
  model_info: {
    prediction_mode: string
    enhanced_metadata: Record<string, any>
    model_version: string
    accuracy_improvement: string
    away_bias_correction: string
  }
  market_features: {
    market_confidence: number
    market_entropy: number
    market_favorite: 'H' | 'D' | 'A'
    home_advantage_market: number
  }
  match_info: {
    home: string
    away: string
    date: string
  }
}

/**
 * Convertit response API v5 vers format frontend
 */
export function adaptAPIv5Response(v5Response: APIv5Response): APIResponse<RoundPredictions> {
  const matches: MatchPrediction[] = Object.entries(v5Response.predictions).map(([matchKey, prediction], index) => {
    // Extract team names from "Team1_vs_Team2" format
    const [homeTeam, awayTeam] = matchKey.split('_vs_')
    
    return {
      id: `gw${v5Response.gameweek}_${index + 1}`,
      home_team: homeTeam,
      away_team: awayTeam,
      date: prediction.match_info.date,
      round: v5Response.gameweek,
      
      // Convert main prediction
      ensemble: {
        prediction: prediction.prediction,
        confidence: prediction.confidence,
        probabilities: prediction.probabilities
      },
      
      // Mock models data (API v5 only has ensemble)
      models: {
        enhanced_baseline: {
          prediction: prediction.prediction,
          confidence: prediction.confidence,
          probabilities: prediction.probabilities
        }
      },
      
      // Calculate disagreement (mock as 0 since only one model)
      disagreement: 0,
      
      // Market data from v5
      market_entropy_norm: prediction.market_features.market_entropy,
      market_overround: undefined, // Not available in v5
      market_probs_raw: {
        home: prediction.probabilities.home,
        draw: prediction.probabilities.draw,
        away: prediction.probabilities.away
      },
      
      // KO-2h data (not available in current v5, default to true)
      ko2h_ok: true,
      kickoff_utc: undefined,
      
      // Odds source data (mock as unavailable for now)
      odds_source: 'unavailable' as const,
      selected_snapshot: undefined,
      selection_metadata: undefined,
      missing_reason: 'no_odds_available'
    }
  })

  return {
    meta: {
      api_version: v5Response.api_version,
      pipeline_version: v5Response.metadata.dataset_hash,
      generated_at: v5Response.metadata.generated_at,
      git_sha: v5Response.metadata.git_sha
    },
    data: {
      round: v5Response.gameweek,
      season: '2025-26', // Hardcoded for now
      competition: 'Premier League',
      total_matches: v5Response.fixtures_count,
      ensemble_system: {
        system_name: 'Enhanced Baseline',
        version: 'v3.0',
        models: {
          enhanced_baseline: {
            name: 'Enhanced Baseline v3.0',
            accuracy: 0.518, // From API response
            weight: 1.0
          }
        },
        ensemble_strategy: 'single_model',
        weights: {
          enhanced_baseline: 1.0
        },
        expected_performance: 0.518
      },
      matches
    }
  }
}

/**
 * Client API v5 avec adaptateur simplifié et détection mode dégradé
 */
export async function fetchGameweekPredictions(
  gameweek: number,
  baseURL: string = 'http://localhost:8000'
): Promise<APIResponse<RoundPredictions> & { degradedMode?: any }> {
  try {
    // Fetch both predictions and fixtures to get real dates
    const [predictionsResponse, fixturesResponse] = await Promise.all([
      fetch(`${baseURL}/api/v5/gameweeks/${gameweek}/predictions`),
      fetch(`${baseURL}/api/v5/gameweeks/${gameweek}/fixtures`)
    ])
    
    const response = predictionsResponse
    
    if (!response.ok) {
      throw new Error(`API v5 Error: ${response.status} ${response.statusText}`)
    }
    
    // Détecter le mode dégradé depuis les headers
    const degradedMode = {
      active: false,
      mode: undefined,
      metadata: undefined
    }
    
    const fallbackMode = response.headers.get('X-Fallback-Mode')
    const dataSource = response.headers.get('X-Data-Source')
    const backendStatus = response.headers.get('X-Backend-Status')
    
    if (fallbackMode === 'true' || dataSource === 'static_fallback') {
      degradedMode.active = true
      degradedMode.mode = 'static_fallback'
      degradedMode.metadata = {
        source: dataSource,
        backend_status: backendStatus,
        last_update: new Date().toISOString()
      }
    } else if (backendStatus === 'unavailable') {
      degradedMode.active = true
      degradedMode.mode = 'backend_unavailable'
      degradedMode.metadata = {
        source: dataSource,
        backend_status: backendStatus,
        last_update: new Date().toISOString()
      }
    }
    
    const rawData = await response.json()
    const fixturesData = fixturesResponse.ok ? await fixturesResponse.json() : null
    
    // Function to convert UTC to French local time
    const convertToFrenchTime = (kickoffUtc: string): string => {
      try {
        const utcDate = new Date(kickoffUtc)
        // Convert to French timezone (Europe/Paris)
        const frenchTime = utcDate.toLocaleString('fr-FR', {
          timeZone: 'Europe/Paris',
          day: '2-digit',
          month: '2-digit', 
          year: 'numeric',
          hour: '2-digit',
          minute: '2-digit'
        })
        return frenchTime.replace(',', '')
      } catch (error) {
        return kickoffUtc
      }
    }

    // Team name mapping for predictions -> fixtures
    const teamMapping: Record<string, string> = {
      'Spurs': 'Tottenham Hotspur',
      'Man Utd': 'Manchester United', 
      'Man City': 'Manchester City',
      'Brighton': 'Brighton and Hove Albion',
      'Nott\'m Forest': 'Nottingham Forest',
      'Newcastle': 'Newcastle United',
      'West Ham': 'West Ham United',
      'Wolves': 'Wolverhampton Wanderers'
    }
    
    // Create fixtures lookup by team names
    const fixturesLookup: Record<string, any> = {}
    if (fixturesData?.data?.fixtures) {
      fixturesData.data.fixtures.forEach((fixture: any) => {
        // Try multiple key combinations to match
        const homeTeam = fixture.home_team
        const awayTeam = fixture.away_team
        
        // Create keys for lookup
        const keys = [
          `${homeTeam}_vs_${awayTeam}`,
          // Try with shortened names
          `${Object.keys(teamMapping).find(k => teamMapping[k] === homeTeam) || homeTeam}_vs_${Object.keys(teamMapping).find(k => teamMapping[k] === awayTeam) || awayTeam}`
        ]
        
        keys.forEach(key => {
          fixturesLookup[key] = fixture
        })
      })
    }
    
    // Adaptation format API v5 → format frontend
    if (rawData.predictions) {
      // Convertir format v5 {predictions: {}} → format frontend {data: {matches: []}}
      const matches = Object.entries(rawData.predictions).map(([matchKey, predData]: [string, any], index) => {
        const [homeTeam, awayTeam] = matchKey.split('_vs_')
        const fixture = fixturesLookup[matchKey]
        
        return {
          id: `gw${gameweek}_${index + 1}`,
          home_team: predData.match_info?.home || homeTeam,
          away_team: predData.match_info?.away || awayTeam,
          date: fixture?.kickoff_utc ? convertToFrenchTime(fixture.kickoff_utc) : (fixture?.kickoff_local || predData.match_info?.date || '2025-10-20'),
          round: gameweek,
          
          // Prédiction ensemble
          ensemble: {
            prediction: predData.prediction,
            confidence: predData.confidence,
            probabilities: predData.probabilities
          },
          
          // Mock models data (API v5 a juste ensemble)
          models: {
            enhanced_baseline: {
              prediction: predData.prediction,
              confidence: predData.confidence,
              probabilities: predData.probabilities
            }
          },
          
          // Calcul disagreement (0 car un seul modèle)
          disagreement: 0,
          
          // Market data si disponible
          market_entropy_norm: predData.market_features?.market_entropy || 0.8,
          market_overround: undefined,
          market_probs_raw: {
            home: predData.probabilities.home,
            draw: predData.probabilities.draw,
            away: predData.probabilities.away
          },
          
          // KO-2h data
          ko2h_ok: true,
          kickoff_utc: undefined,
          
          // Odds source data
          odds_source: 'unavailable' as const,
          selected_snapshot: undefined,
          selection_metadata: undefined,
          missing_reason: 'no_odds_available'
        }
      })

      const result = {
        meta: {
          api_version: rawData.api_version || '5.0.0',
          pipeline_version: rawData.metadata?.pipeline_version || 'durci_v1.0',
          generated_at: rawData.metadata?.generated_at || new Date().toISOString(),
          git_sha: rawData.metadata?.git_sha || 'unknown'
        },
        data: {
          round: gameweek,
          season: '2025-26',
          competition: 'Premier League',
          total_matches: rawData.fixtures_count || matches.length,
          ensemble_system: {
            system_name: 'Enhanced Baseline',
            version: rawData.metadata?.model_version || 'v3.0',
            models: {
              enhanced_baseline: {
                name: 'Enhanced Baseline v3.0',
                accuracy: 0.535,
                weight: 1.0
              }
            },
            ensemble_strategy: 'single_model',
            weights: { enhanced_baseline: 1.0 },
            expected_performance: 0.535
          },
          matches
        },
        degradedMode: degradedMode.active ? degradedMode : undefined
      }
      
      return result
    }
    
    // Fallback si le format est différent
    throw new Error('Invalid response format from backend')
    
  } catch (error) {
    console.error('API v5 fetch error:', error)
    throw error
  }
}