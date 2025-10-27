'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useQuery } from '@tanstack/react-query'
import { useState, useEffect } from 'react'
import { oddsyAPI } from '@/lib/api'
import { clsx } from 'clsx'

export function NavigationDynamic() {
  const pathname = usePathname()
  const [latestRound, setLatestRound] = useState(7) // Default fallback
  const [isClient, setIsClient] = useState(false)
  
  // Fetch latest available round for dynamic navigation
  const { data: availableData, isSuccess } = useQuery({
    queryKey: ['available-rounds'],
    queryFn: () => oddsyAPI.getAvailableRounds(),
    staleTime: 5 * 60 * 1000, // 5 minutes
  })

  // Update client state on hydration
  useEffect(() => {
    setIsClient(true)
  }, [])

  // Update latest round only on client side to prevent hydration mismatch
  useEffect(() => {
    if (isSuccess && availableData?.data?.latest_round) {
      setLatestRound(availableData.data.latest_round)
    }
  }, [isSuccess, availableData])

  // Prevent hydration mismatch by not rendering dynamic content on server
  if (!isClient) {
    return (
      <nav className="hidden md:flex space-x-8">
        <div className="h-8 w-32 bg-gray-200 rounded animate-pulse"></div>
        <div className="h-8 w-32 bg-gray-200 rounded animate-pulse"></div>
        <div className="h-8 w-32 bg-gray-200 rounded animate-pulse"></div>
      </nav>
    )
  }

  const navigation = [
    { name: 'Dashboard', href: '/' },
    { name: `Predictions J${latestRound}`, href: `/predictions/${latestRound}` },
    { name: 'Performance', href: '/performance' },
    { name: 'Models', href: '/models' },
    { name: 'Pipeline', href: '/pipeline' },
  ]

  return (
    <nav className="hidden md:flex space-x-8">
      {navigation.map((item) => (
        <Link
          key={item.name}
          href={item.href}
          className={clsx(
            'px-3 py-2 text-sm font-medium rounded-md transition-colors',
            pathname === item.href
              ? 'bg-oddsy-primary text-white'
              : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
          )}
        >
          {item.name}
        </Link>
      ))}
    </nav>
  )
}