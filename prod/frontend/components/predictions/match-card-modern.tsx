'use client'

import { ClubLogo } from '@/components/ui/club-logo'
import { StadiumBackground } from '@/components/ui/stadium-background'
import { getConfidenceColors, getProgressBarColor, formatConfidence } from '@/lib/confidence-colors'
import { useState, useEffect } from 'react'
import { clsx } from 'clsx'

// Fonction pour raccourcir les noms d'équipes
const shortenTeamName = (teamName: string): string => {
  const shortNames: Record<string, string> = {
    'Brighton and Hove Albion': 'Brighton',
    'Wolverhampton Wanderers': 'Wolves',
    'West Ham United': 'West Ham',
    'Manchester United': 'Man United',
    'Manchester City': 'Man City',
    'Tottenham Hotspur': 'Tottenham',
    'Newcastle United': 'Newcastle',
    'Nottingham Forest': 'Forest'
  }
  return shortNames[teamName] || teamName
}

// Fonction pour tronquer intelligemment les noms longs
const truncateTeamName = (teamName: string, maxLength: number = 14): string => {
  if (!teamName) return ''
  const shortName = shortenTeamName(teamName)
  if (shortName.length <= maxLength) return shortName
  return shortName.substring(0, maxLength - 1) + '…'
}

interface TeamHeaderProps {
  team: string
  align: 'left' | 'right'
}

function TeamHeader({ team, align }: TeamHeaderProps) {
  const isLeft = align === 'left'
  
  return (
    <div className={clsx(
      'flex items-center gap-3',
      isLeft ? 'flex-row' : 'flex-row-reverse'
    )}>
      <ClubLogo clubName={team} size="md" />
      <div className={clsx(
        'flex flex-col',
        isLeft ? 'items-start' : 'items-end'
      )}>
        <span className="text-sm md:text-base font-bold text-white drop-shadow-lg truncate max-w-[120px]">
          {truncateTeamName(team, 14)}
        </span>
        <span className="text-xs text-white/80 font-medium">
          {isLeft ? 'HOME' : 'AWAY'}
        </span>
      </div>
    </div>
  )
}

interface StatPillProps {
  label: string
  value: string
  color: 'green' | 'yellow' | 'blue'
  strong?: boolean
  delay?: number
}

function StatPill({ label, value, color, strong, delay = 0 }: StatPillProps) {
  const colorMap = {
    green: 'bg-white/18 text-white border-white/25 backdrop-blur-[4px]',
    yellow: 'bg-white/18 text-white border-white/25 backdrop-blur-[4px]',
    blue: 'bg-white/18 text-white border-white/25 backdrop-blur-[4px]'
  }
  
  return (
    <div 
      className={clsx(
        'flex flex-col items-center px-2 py-1.5 rounded-xl border flex-1',
        'transition-all duration-300 hover:scale-105',
        'animate-slideUp opacity-0',
        colorMap[color],
        strong ? 'font-bold text-sm' : 'font-semibold text-xs'
      )}
      style={{ 
        animationDelay: `${delay}ms`,
        animationFillMode: 'forwards'
      }}
    >
      <span 
        className="text-xs font-medium opacity-90"
        style={{ textShadow: '0 1px 2px rgba(0,0,0,0.35)' }}
      >
        {label}
      </span>
      <span 
        className="leading-tight text-center font-semibold"
        style={{ textShadow: '0 1px 2px rgba(0,0,0,0.35)' }}
      >
        {value}
      </span>
    </div>
  )
}

export function MatchCardModern({ match }: { match: any }) {
  const [isHovered, setIsHovered] = useState(false)
  const [progressWidth, setProgressWidth] = useState(0)
  
  const getPredictionResult = (prediction: string) => {
    const homeTeam = shortenTeamName(match.home_team || 'Home')
    const awayTeam = shortenTeamName(match.away_team || 'Away')
    
    switch (prediction) {
      case 'H': return { type: 'team', team: homeTeam, clubName: match.home_team }
      case 'D': return { type: 'draw' }
      case 'A': return { type: 'team', team: awayTeam, clubName: match.away_team }
      default: return { type: 'uncertain' }
    }
  }

  const getPredictionColor = (prediction: string) => {
    switch (prediction) {
      case 'H': return { badge: 'bg-green-500', text: 'text-white' }
      case 'D': return { badge: 'bg-yellow-500', text: 'text-white' }
      case 'A': return { badge: 'bg-blue-500', text: 'text-white' }
      default: return { badge: 'bg-gray-500', text: 'text-white' }
    }
  }

  const confidence = match.ensemble?.confidence || 0
  const confidenceColors = getConfidenceColors(confidence)
  const confidencePercentage = confidence * 100
  const predictionColors = getPredictionColor(match.ensemble?.prediction || 'H')
  const predictionResult = getPredictionResult(match.ensemble?.prediction || 'H')
  
  // Animation progressive de la barre de confiance
  useEffect(() => {
    const timer = setTimeout(() => {
      setProgressWidth(Math.max(confidencePercentage, 5))
    }, 800)
    return () => clearTimeout(timer)
  }, [confidencePercentage])
  

  return (
    <div 
      className={clsx(
        'relative rounded-2xl overflow-hidden shadow-2xl mx-auto cursor-pointer group',
        'w-full max-w-lg md:max-w-xl lg:max-w-2xl h-[420px] md:h-[480px]',
        'border border-gray-200 transition-all duration-500',
        'hover:shadow-3xl'
      )}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Background stade - plein écran */}
      <div className="absolute inset-0 overflow-hidden rounded-2xl">
        <StadiumBackground 
          homeTeam={match.home_team} 
          overlayOpacity={isHovered ? 0.03 : 0.05}
        />
      </div>
      
      {/* Gradient adaptatif pour contraste */}
      <div className="absolute inset-0 bg-gradient-to-b from-black/35 via-transparent via-60% to-black/50" />
      
      {/* Bordure intérieure subtile */}
      <div className="absolute inset-0 rounded-2xl ring-1 ring-inset ring-white/10" />
      
      {/* Gradient de contraste contextuel en bas */}
      <div className="absolute bottom-0 h-28 w-full bg-gradient-to-t from-black/25 to-transparent pointer-events-none rounded-2xl" />
      
      {/* Header équipes flottant en haut */}
      <div className="absolute z-20 w-full top-0 px-6 md:px-8 py-5 flex items-center justify-between">
        <TeamHeader team={match.home_team} align="left" />
        
        <div className="flex flex-col items-center">
          <span className="text-lg md:text-xl font-bold text-white/90 drop-shadow-lg">VS</span>
          <div className="mt-1 bg-black/40 backdrop-blur-sm rounded-full px-3 py-1">
            <span className="text-xs text-white font-medium">
              GW{match.round} • {match.date}
            </span>
          </div>
        </div>
        
        <TeamHeader team={match.away_team} align="right" />
      </div>

      {/* Bloc stats flottant en bas avec glassmorphism premium */}
      <div className="absolute bottom-3 md:bottom-4 left-0 right-0 flex justify-center">
        <div className={clsx(
          'w-[94%] md:w-[78%]',
          'bg-white/18 bg-gradient-to-b from-white/22 via-white/12 to-white/8',
          'backdrop-blur-[6px] border border-white/35',
          'rounded-[20px] px-6 md:px-8 py-5 md:py-6 flex flex-col items-center',
          'shadow-[0_10px_40px_-10px_rgba(0,0,0,0.35)]',
          'transition-all duration-300 animate-fadeUp',
          isHovered && '-translate-y-0.5 backdrop-blur-[7px]'
        )}
        style={{
          boxShadow: isHovered 
            ? '0 25px 50px -12px rgba(0, 0, 0, 0.15), 0 8px 25px -8px rgba(0, 0, 0, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.1)'
            : '0 20px 40px -12px rgba(0, 0, 0, 0.12), 0 8px 20px -8px rgba(0, 0, 0, 0.06), inset 0 1px 0 rgba(255, 255, 255, 0.05)'
        }}>
        
        {/* Prédiction principale */}
        <div className={clsx(
          'px-3 md:px-4 py-1.5 md:py-2 rounded-full font-bold text-sm md:text-base',
          'tracking-tight transition-all duration-300 mb-2 md:mb-3',
          'hover:scale-[1.02] flex items-center gap-1.5 md:gap-2',
          'cursor-pointer hover:brightness-110 ring-1 ring-white/30',
          'shadow-md hover:shadow-lg',
          predictionColors.badge,
          predictionColors.text
        )}>
          {predictionResult.type === 'team' ? (
            <>
              <ClubLogo clubName={predictionResult.clubName} size="sm" />
              <span>{predictionResult.team} Win</span>
            </>
          ) : predictionResult.type === 'draw' ? (
            <span>🤝 Draw</span>
          ) : (
            <span>🤔 Uncertain</span>
          )}
        </div>

        {/* Barre de confiance premium avec glow */}
        <div className="w-full mb-3">
          <div className="flex items-center justify-between mb-1">
            <span 
              className="text-xs font-semibold text-white"
              style={{ textShadow: '0 1px 2px rgba(0,0,0,0.35)' }}
            >
              {confidenceColors.label}
            </span>
            <span 
              className={clsx(
                'text-xs font-bold',
                confidence >= 0.6 ? 'text-white' : 
                confidence >= 0.45 ? 'text-white' : 
                confidence >= 0.30 ? 'text-white/90' : 'text-white/80'
              )}
              style={{ textShadow: '0 1px 2px rgba(0,0,0,0.35)' }}
            >
              {formatConfidence(confidence)}
            </span>
          </div>
          <div className="w-full h-1.5 bg-white/25 rounded-full overflow-hidden">
            <div 
              className="h-1.5 rounded-full transition-[width] duration-700 ease-out"
              style={{ 
                width: `${progressWidth}%`,
                minWidth: progressWidth > 0 ? '12px' : '0px',
                backgroundColor: confidence >= 0.6 ? '#10b981' : confidence >= 0.45 ? '#3b82f6' : confidence >= 0.30 ? '#60a5fa' : '#93c5fd',
                boxShadow: '0 2px 8px rgba(0,0,0,0.25)'
              }}
            />
          </div>
        </div>

        {/* Probabilités en pills stylisées avec animation staggerée */}
        <div className="flex justify-between items-center gap-2 w-full">
          <StatPill 
            label="Home" 
            value={`${((match.ensemble?.probabilities?.home || 0) * 100).toFixed(1)}%`}
            color="green" 
            strong={match.ensemble?.prediction === 'H'}
            delay={100}
          />
          <StatPill 
            label="Draw" 
            value={`${((match.ensemble?.probabilities?.draw || 0) * 100).toFixed(1)}%`}
            color="yellow"
            strong={match.ensemble?.prediction === 'D'}
            delay={200}
          />
          <StatPill 
            label="Away" 
            value={`${((match.ensemble?.probabilities?.away || 0) * 100).toFixed(1)}%`}
            color="blue"
            strong={match.ensemble?.prediction === 'A'}
            delay={300}
          />
        </div>

        </div>
      </div>
    </div>
  )
}