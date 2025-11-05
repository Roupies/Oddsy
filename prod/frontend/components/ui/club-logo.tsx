'use client'

import Image from 'next/image'
import { useState } from 'react'
import { clsx } from 'clsx'

interface ClubLogoProps {
  clubName: string
  size?: 'sm' | 'md' | 'lg' | 'xl'
  className?: string
  showFallback?: boolean
}

const LOGO_SIZES = {
  sm: 'w-10 h-10 text-sm',
  md: 'w-12 h-12 text-base',
  lg: 'w-16 h-16 text-lg',
  xl: 'w-20 h-20 text-xl'
}

const TEAM_LOGO_MAP: Record<string, string> = {
  // Noms exacts de l'API v5
  'Arsenal': '/logos/Arsenal.svg',
  'Aston Villa': '/logos/Aston_Villa.svg',
  'Brighton and Hove Albion': '/logos/Brighton.svg',
  'Burnley': '/logos/Burnley.svg',
  'Chelsea': '/logos/Chelsea.svg',
  'Crystal Palace': '/logos/Crystal_Palace.svg',
  'Everton': '/logos/Everton.svg',
  'Fulham': '/logos/Fulham.svg',
  'Liverpool': '/logos/Liverpool.svg',
  'Manchester City': '/logos/Manchester_City.svg',
  'Manchester United': '/logos/Manchester_United.svg',
  'Newcastle United': '/logos/Newcastle.svg',
  'Nottingham Forest': '/logos/Nottingham_Forest.svg',
  'Tottenham Hotspur': '/logos/Tottenham.svg',
  'West Ham United': '/logos/West_Ham.svg',
  'Wolverhampton Wanderers': '/logos/Wolverhampton.svg',
  'Brentford': '/logos/Brentford.svg',
  'Bournemouth': '/logos/Bournemouth.svg',
  // Équipes qui pourraient apparaître occasionnellement
  'Leeds United': '/logos/Leeds_United.svg',
  'Sunderland': '/logos/Sunderland.svg',
  // Alias pour compatibilité
  'Brighton': '/logos/Brighton.svg',
  'Newcastle': '/logos/Newcastle.svg',
  'Tottenham': '/logos/Tottenham.svg',
  'West Ham': '/logos/West_Ham.svg',
  'Wolves': '/logos/Wolverhampton.svg',
  // Alias pour prédictions J9
  'Leeds': '/logos/Leeds_United.svg',
  'Man Utd': '/logos/Manchester_United.svg',
  'Man City': '/logos/Manchester_City.svg',
  'Nott\'m Forest': '/logos/Nottingham_Forest.svg',
  'Spurs': '/logos/Tottenham.svg',
  'Crystal Palace': '/logos/Crystal_Palace.svg'
}

const getTeamInitials = (teamName: string): string => {
  return teamName
    .split(' ')
    .map(word => word.charAt(0))
    .join('')
    .toUpperCase()
    .slice(0, 3)
}

const getTeamColors = (teamName: string): { bg: string; text: string } => {
  const colorMap: Record<string, { bg: string; text: string }> = {
    'Arsenal': { bg: 'bg-red-600', text: 'text-white' },
    'Aston Villa': { bg: 'bg-purple-800', text: 'text-white' },
    'Brighton': { bg: 'bg-blue-500', text: 'text-white' },
    'Burnley': { bg: 'bg-orange-600', text: 'text-white' },
    'Chelsea': { bg: 'bg-blue-600', text: 'text-white' },
    'Crystal Palace': { bg: 'bg-blue-800', text: 'text-white' },
    'Everton': { bg: 'bg-blue-700', text: 'text-white' },
    'Fulham': { bg: 'bg-gray-800', text: 'text-white' },
    'Liverpool': { bg: 'bg-red-700', text: 'text-white' },
    'Luton': { bg: 'bg-orange-500', text: 'text-white' },
    'Manchester City': { bg: 'bg-sky-500', text: 'text-white' },
    'Manchester United': { bg: 'bg-red-600', text: 'text-white' },
    'Newcastle': { bg: 'bg-gray-900', text: 'text-white' },
    'Nottingham Forest': { bg: 'bg-red-800', text: 'text-white' },
    'Sheffield United': { bg: 'bg-red-700', text: 'text-white' },
    'Tottenham': { bg: 'bg-gray-100', text: 'text-gray-900' },
    'West Ham': { bg: 'bg-purple-900', text: 'text-white' },
    'Wolves': { bg: 'bg-orange-600', text: 'text-white' },
    'Brentford': { bg: 'bg-red-600', text: 'text-white' },
    'Bournemouth': { bg: 'bg-red-700', text: 'text-white' },
    // Alias pour prédictions J9
    'Leeds': { bg: 'bg-yellow-400', text: 'text-blue-900' },
    'Man Utd': { bg: 'bg-red-600', text: 'text-white' },
    'Man City': { bg: 'bg-sky-500', text: 'text-white' },
    'Nott\'m Forest': { bg: 'bg-red-800', text: 'text-white' },
    'Spurs': { bg: 'bg-gray-100', text: 'text-gray-900' },
    'Sunderland': { bg: 'bg-red-600', text: 'text-white' }
  }
  
  return colorMap[teamName] || { bg: 'bg-gray-500', text: 'text-white' }
}

export function ClubLogo({ 
  clubName, 
  size = 'md', 
  className,
  showFallback = true 
}: ClubLogoProps) {
  const [imageError, setImageError] = useState(false)
  const logoPath = TEAM_LOGO_MAP[clubName]
  const initials = getTeamInitials(clubName)
  const colors = getTeamColors(clubName)
  
  const sizeClasses = LOGO_SIZES[size]
  
  // Si on a un logo et pas d'erreur, afficher l'image
  if (logoPath && !imageError) {
    return (
      <div className={clsx(
        'relative flex-shrink-0 rounded-full p-0.5 drop-shadow-sm',
        sizeClasses,
        className
      )}>
        <div className="relative w-full h-full rounded-full overflow-hidden">
          <Image
            src={logoPath}
            alt={`${clubName} logo`}
            fill
            className="object-contain object-center p-0.5"
            onError={() => setImageError(true)}
            sizes="(max-width: 768px) 40px, 60px"
            priority={false}
          />
        </div>
      </div>
    )
  }
  
  // Fallback avec initiales et couleurs d'équipe
  if (showFallback) {
    return (
      <div className={clsx(
        'rounded-full flex items-center justify-center flex-shrink-0 font-bold drop-shadow-sm',
        sizeClasses,
        colors.bg,
        colors.text,
        className
      )}>
        {initials}
      </div>
    )
  }
  
  return null
}