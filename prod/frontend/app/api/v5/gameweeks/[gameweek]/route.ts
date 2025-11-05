import { NextRequest, NextResponse } from 'next/server'
import { readFileSync } from 'fs'
import { join } from 'path'

// Backend API URL for primary data source
const BACKEND_API_URL = process.env.BACKEND_API_URL || 'http://localhost:8000'

async function fetchFromBackend(gameweek: number): Promise<any> {
  try {
    const backendUrl = `${BACKEND_API_URL}/api/gameweeks/${gameweek}/predictions`
    console.log(`🔄 Attempting to fetch from backend: ${backendUrl}`)
    
    const response = await fetch(backendUrl, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
        'User-Agent': 'Oddsy-Frontend-Fallback/1.0'
      },
      // Timeout après 5 secondes
      signal: AbortSignal.timeout(5000)
    })
    
    if (response.ok) {
      const data = await response.json()
      console.log(`✅ Backend fetch successful for GW${gameweek}`)
      return { data, source: 'backend' }
    } else {
      console.warn(`⚠️ Backend returned ${response.status}: ${response.statusText}`)
      return null
    }
  } catch (error) {
    console.warn(`⚠️ Backend fetch failed for GW${gameweek}:`, error.message)
    return null
  }
}

function readStaticFallback(gameweek: number): any {
  try {
    // Essayer de lire le fichier de prédictions depuis prediction/[gameweek]/prediction.json
    const predictionPath = join(process.cwd(), '..', 'prediction', gameweek.toString(), 'prediction.json')
    
    const predictionData = readFileSync(predictionPath, 'utf-8')
    const predictions = JSON.parse(predictionData)
    
    console.log(`📁 Static fallback successful for GW${gameweek}`)
    
    // Ajouter les métadonnées de mode dégradé
    return {
      ...predictions,
      _fallback_mode: {
        source: 'static_files',
        mode: 'degraded',
        backend_unavailable: true,
        file_path: predictionPath,
        warning: 'Données statiques - Le backend principal est indisponible'
      }
    }
  } catch (error) {
    console.error(`❌ Static fallback failed for GW${gameweek}:`, error)
    return null
  }
}

export async function GET(
  request: NextRequest,
  { params }: { params: { gameweek: string } }
) {
  try {
    const gameweek = parseInt(params.gameweek)
    
    // Validation gameweek
    if (isNaN(gameweek) || gameweek < 1 || gameweek > 38) {
      return NextResponse.json(
        { error: 'Invalid gameweek. Must be between 1 and 38.' },
        { status: 400 }
      )
    }
    
    // Stratégie hybride : Backend d'abord, puis fallback statique
    let result = await fetchFromBackend(gameweek)
    
    if (result) {
      // Backend disponible - retourner les données fraîches
      return NextResponse.json(result.data, {
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'GET',
          'Access-Control-Allow-Headers': 'Content-Type',
          'X-Data-Source': 'backend',
          'X-Fallback-Mode': 'false',
          'Cache-Control': 'public, max-age=300', // 5 minutes cache pour données backend
        },
      })
    }
    
    // Backend indisponible - essayer le fallback statique
    const staticData = readStaticFallback(gameweek)
    
    if (staticData) {
      // Mode dégradé avec bannière
      return NextResponse.json(staticData, {
        status: 200,
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'GET',
          'Access-Control-Allow-Headers': 'Content-Type',
          'X-Data-Source': 'static_fallback',
          'X-Fallback-Mode': 'true',
          'X-Backend-Status': 'unavailable',
          'Cache-Control': 'public, max-age=60', // Cache court pour données statiques
          'Warning': '199 - "Données statiques - Backend indisponible"',
        },
      })
    }
    
    // Aucune source disponible
    return NextResponse.json(
      { 
        error: `No predictions available for gameweek ${gameweek}`,
        details: 'Backend unavailable and no static fallback found',
        gameweek,
        sources_attempted: ['backend_api', 'static_files']
      },
      { 
        status: 404,
        headers: {
          'X-Data-Source': 'none',
          'X-Fallback-Mode': 'failed',
          'X-Backend-Status': 'unavailable'
        }
      }
    )
    
  } catch (error) {
    console.error('API Error:', error)
    return NextResponse.json(
      { 
        error: 'Internal server error',
        details: error.message 
      },
      { status: 500 }
    )
  }
}