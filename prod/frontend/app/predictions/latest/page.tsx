import { redirect } from 'next/navigation'

/**
 * /predictions/latest - Server Component avec redirect permanent
 * ============================================================
 * Résout automatiquement la GW courante via Pipeline Durci et redirige
 * ISR Configuration: revalidate courte + on-demand
 */

export const revalidate = 0 // No cache in dev, always fresh

async function getLatestGameweek(): Promise<number> {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
  
  try {
    const response = await fetch(`${apiUrl}/api/v5/gameweeks/latest`, {
      cache: 'no-store' // Toujours frais pour la découverte
    })
    
    if (!response.ok) {
      throw new Error(`Failed to fetch latest gameweek: ${response.status}`)
    }
    
    const data = await response.json()
    return data.data.latest_gameweek
  } catch (error) {
    console.error('Error fetching latest gameweek:', error)
    // Fallback vers J9 si API indisponible
    return 9
  }
}

export default async function PredictionsLatestPage() {
  const latestGw = await getLatestGameweek()
  
  // Redirect permanent (308) vers la GW courante
  redirect(`/predictions/${latestGw}`)
}