'use client'

import { useQuery } from '@tanstack/react-query'
import { oddsyAPI } from '@/lib/api'
import { useRouter } from 'next/navigation'
import { useEffect } from 'react'
import { LoadingSpinner } from '@/components/ui/loading-spinner'

export default function MatchdayRedirectPage() {
  const router = useRouter()
  
  // Fetch latest gameweek directly
  const { data: latestData, isLoading, error } = useQuery({
    queryKey: ['latest-gameweek'],
    queryFn: async () => {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/v5/gameweeks/latest`)
      if (!response.ok) throw new Error('Failed to fetch latest gameweek')
      return await response.json()
    },
    staleTime: 5 * 60 * 1000, // 5 minutes
  })

  useEffect(() => {
    if (latestData?.data?.latest_gameweek) {
      // Redirect to latest available matchday
      router.replace(`/matchday/${latestData.data.latest_gameweek}`)
    }
  }, [latestData, router])

  if (isLoading) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="flex flex-col items-center justify-center min-h-[400px]">
          <LoadingSpinner />
          <p className="mt-4 text-neutral-400">Finding latest predictions...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="text-center py-12">
          <h1 className="text-2xl font-bold text-white mb-4">
            Unable to load matchdays
          </h1>
          <p className="text-neutral-400 mb-6">
            There was an error loading available predictions.
          </p>
          <button 
            onClick={() => window.location.reload()} 
            className="bg-oddsy-primary text-white px-6 py-3 rounded-lg hover:bg-oddsy-primary/90 transition-colors"
          >
            Try Again
          </button>
        </div>
      </div>
    )
  }

  if (!latestData?.data?.latest_gameweek) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="text-center py-12">
          <h1 className="text-2xl font-bold text-white mb-4">
            No predictions available
          </h1>
          <p className="text-neutral-400 mb-6">
            No matchday predictions are currently available.
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

  return null
}