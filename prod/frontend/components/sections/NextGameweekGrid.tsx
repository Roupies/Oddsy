'use client'

import React from 'react'
import Link from 'next/link'
import { motion } from 'framer-motion'
import { useCurrentGameweek } from '@/hooks/useCurrentGameweek'
import { APP_CONFIG, getStadiumImage } from '@/config/appConstants'
import { LoadingSpinner } from '@/components/ui/loading-spinner'

interface MatchPrediction {
  homeTeam: string
  awayTeam: string
  stadium: string
  stadiumImage: string
  homeWinProb: number
  drawProb: number
  awayWinProb: number
  confidence: 'high' | 'medium' | 'low'
  prediction: 'home' | 'draw' | 'away'
  kickoff: string
}

const nextGameweekData: MatchPrediction[] = [
  {
    homeTeam: "Arsenal",
    awayTeam: "Liverpool",
    stadium: "Emirates Stadium",
    stadiumImage: "/images/Stades/Arsenal.avif",
    homeWinProb: 45,
    drawProb: 28,
    awayWinProb: 27,
    confidence: 'high',
    prediction: 'home',
    kickoff: "Sunday 15:00"
  },
  {
    homeTeam: "Manchester City",
    awayTeam: "Chelsea",
    stadium: "Etihad Stadium", 
    stadiumImage: "/images/Stades/Manchester_City.jpg",
    homeWinProb: 67,
    drawProb: 21,
    awayWinProb: 12,
    confidence: 'high',
    prediction: 'home',
    kickoff: "Saturday 17:30"
  },
  {
    homeTeam: "Newcastle",
    awayTeam: "Tottenham",
    stadium: "St. James' Park",
    stadiumImage: "/images/Stades/Newcastle.jpg",
    homeWinProb: 38,
    drawProb: 31,
    awayWinProb: 31,
    confidence: 'medium',
    prediction: 'draw',
    kickoff: "Sunday 14:00"
  }
]

const PredictionCard: React.FC<{ match: MatchPrediction; index: number }> = ({ match, index }) => {
  const getPredictionColor = (pred: string) => {
    switch (pred) {
      case 'home': return 'text-emerald-500'
      case 'away': return 'text-blue-400'
      case 'draw': return 'text-yellow-400'
      default: return 'text-neutral-400'
    }
  }

  const getConfidenceIcon = (conf: string) => {
    switch (conf) {
      case 'high': return '🎯'
      case 'medium': return '⚖️'
      case 'low': return '🤔'
      default: return '📊'
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay: index * 0.1 }}
      viewport={{ once: true }}
      className="group relative overflow-hidden rounded-lg bg-neutral-900 border border-neutral-800 hover:border-neutral-700 transition-all duration-300 hover:-translate-y-1"
    >
      {/* Stadium background */}
      <div className="relative h-48 overflow-hidden">
        <div 
          className="stadium-bg w-full h-full bg-cover bg-center transition-all duration-700 group-hover:scale-105"
          style={{ backgroundImage: `url('${match.stadiumImage}')` }}
        />
        <div className="absolute inset-0 bg-gradient-to-t from-neutral-900 via-neutral-900/60 to-transparent" />
        
        {/* Kickoff time */}
        <div className="absolute top-4 right-4 bg-black/60 backdrop-blur-sm rounded px-3 py-1">
          <span className="text-white text-xs font-mono">{match.kickoff}</span>
        </div>

        {/* Confidence indicator */}
        <div className="absolute top-4 left-4 bg-black/60 backdrop-blur-sm rounded-full px-3 py-1">
          <span className="text-white text-xs">
            {getConfidenceIcon(match.confidence)} {match.confidence.toUpperCase()}
          </span>
        </div>
      </div>

      {/* Match info */}
      <div className="p-6">
        {/* Teams */}
        <div className="mb-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-white font-semibold">{match.homeTeam}</span>
            <span className="text-neutral-500 text-sm">vs</span>
            <span className="text-white font-semibold">{match.awayTeam}</span>
          </div>
          <div className="text-neutral-400 text-sm text-center">
            {match.stadium}
          </div>
        </div>

        {/* Prediction */}
        <div className="mb-4">
          <div className="text-center mb-2">
            <span className="text-xs text-neutral-500 uppercase tracking-wider">Prediction</span>
          </div>
          <div className={`text-xl font-bold text-center ${getPredictionColor(match.prediction)}`}>
            {match.prediction === 'home' ? match.homeTeam : 
             match.prediction === 'away' ? match.awayTeam : 'Draw'}
          </div>
        </div>

        {/* Probabilities */}
        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <span className="text-xs text-neutral-400">Home</span>
            <div className="flex-1 mx-3 bg-neutral-800 rounded-full h-1">
              <div 
                className="h-1 bg-emerald-500 rounded-full transition-all duration-1000"
                style={{ width: `${match.homeWinProb}%` }}
              />
            </div>
            <span className="text-xs text-white font-mono">{match.homeWinProb}%</span>
          </div>
          
          <div className="flex justify-between items-center">
            <span className="text-xs text-neutral-400">Draw</span>
            <div className="flex-1 mx-3 bg-neutral-800 rounded-full h-1">
              <div 
                className="h-1 bg-yellow-400 rounded-full transition-all duration-1000"
                style={{ width: `${match.drawProb}%` }}
              />
            </div>
            <span className="text-xs text-white font-mono">{match.drawProb}%</span>
          </div>
          
          <div className="flex justify-between items-center">
            <span className="text-xs text-neutral-400">Away</span>
            <div className="flex-1 mx-3 bg-neutral-800 rounded-full h-1">
              <div 
                className="h-1 bg-blue-400 rounded-full transition-all duration-1000"
                style={{ width: `${match.awayWinProb}%` }}
              />
            </div>
            <span className="text-xs text-white font-mono">{match.awayWinProb}%</span>
          </div>
        </div>
      </div>
    </motion.div>
  )
}

export const NextGameweekGrid: React.FC = () => {
  const { data: gameweekData, isLoading, error } = useCurrentGameweek()

  const currentGameweek = gameweekData?.currentGameweek || 11
  
  if (isLoading) {
    return (
      <section id="predictions" className="bg-neutral-950 text-white py-12 md:py-16 px-4 md:px-8 lg:px-16">
        <div className="max-w-7xl mx-auto">
          <div className="flex flex-col items-center justify-center min-h-[400px]">
            <LoadingSpinner />
            <p className="mt-4 text-neutral-400">Loading gameweek predictions...</p>
          </div>
        </div>
      </section>
    )
  }

  // Show error state if no data available
  if (!gameweekData?.upcomingMatches?.length) {
    return (
      <section id="predictions" className="bg-neutral-950 text-white py-12 md:py-16 px-4 md:px-8 lg:px-16">
        <div className="max-w-7xl mx-auto">
          <div className="flex flex-col items-center justify-center min-h-[400px]">
            <div className="text-center">
              <div className="text-6xl mb-4">⚠️</div>
              <h3 className="text-xl font-semibold text-white mb-2">Predictions Unavailable</h3>
              <p className="text-neutral-400 mb-4">
                {error ? 'Failed to load prediction data.' : 'No upcoming matches available for display.'}
              </p>
              <p className="text-sm text-neutral-500">
                Please check back later or contact support if this persists.
              </p>
            </div>
          </div>
        </div>
      </section>
    )
  }

  const displayMatches = gameweekData.upcomingMatches.map(match => ({
    ...match,
    stadiumImage: getStadiumImage(match.homeTeam)
  }))

  return (
    <section id="predictions" className="bg-neutral-950 text-white py-12 md:py-16 px-4 md:px-8 lg:px-16">
      <div className="max-w-7xl mx-auto">
        {/* Section header */}
        <motion.div 
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="mb-12"
        >
          <div className="flex items-center gap-4 mb-6">
            <div className="w-2 h-16 bg-emerald-500" />
            <span className="text-neutral-500 text-sm font-mono uppercase tracking-wider">
              {APP_CONFIG.MODEL_VERSION}
            </span>
          </div>
          
          <h2 className="text-[clamp(3rem,6vw,5rem)] font-black tracking-tighter leading-[0.9] mb-4">
            NEXT
            <br />
            GAMEWEEK
          </h2>
          <div className="w-20 h-1 bg-emerald-500 mb-6" />
          
          <p className="text-neutral-400 text-lg md:text-xl leading-relaxed max-w-2xl">
            Live predictions powered by machine learning. Real-time analysis of form, 
            venue advantage, and tactical patterns across the Premier League.
          </p>
        </motion.div>

        {/* Error state */}
        {error && (
          <div className="mb-8 p-4 bg-amber-950/20 border border-amber-500/30 rounded-lg">
            <p className="text-amber-300 text-sm">
              ⚠️ Using cached predictions. Live data temporarily unavailable.
            </p>
          </div>
        )}

        {/* Predictions grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {displayMatches.map((match, index) => (
            <PredictionCard 
              key={`${match.homeTeam}-${match.awayTeam}`}
              match={match}
              index={index}
            />
          ))}
        </div>

        {/* Bottom CTA */}
        <motion.div 
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="mt-12 text-center"
        >
          <Link href={`/matchday/${currentGameweek}`}>
            <button className="group bg-emerald-500 hover:bg-emerald-400 text-neutral-950 font-bold px-8 py-4 rounded transition-all duration-300 hover:-translate-y-0.5">
              <span className="flex items-center gap-2">
                View All GW{currentGameweek} Predictions
                <svg className="w-5 h-5 transition-transform group-hover:translate-x-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                </svg>
              </span>
            </button>
          </Link>
        </motion.div>
      </div>
    </section>
  )
}