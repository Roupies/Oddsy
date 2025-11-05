'use client'

import { useEffect, useState } from 'react'
import { clsx } from 'clsx'
import { PredictionOutcome } from '@/lib/types'

interface ConfidenceGaugeProps {
  confidence: number
  prediction: PredictionOutcome
  className?: string
  size?: 'sm' | 'md' | 'lg'
  animated?: boolean
}

export function ConfidenceGauge({ 
  confidence, 
  prediction, 
  className = '',
  size = 'md',
  animated = true 
}: ConfidenceGaugeProps) {
  const [displayConfidence, setDisplayConfidence] = useState(0)
  
  // Animation count-up effect
  useEffect(() => {
    if (!animated) {
      setDisplayConfidence(confidence)
      return
    }
    
    const duration = 800 // 800ms animation
    const steps = 50
    const increment = confidence / steps
    const stepDuration = duration / steps
    
    let current = 0
    const timer = setInterval(() => {
      current += increment
      if (current >= confidence) {
        setDisplayConfidence(confidence)
        clearInterval(timer)
      } else {
        setDisplayConfidence(current)
      }
    }, stepDuration)
    
    return () => clearInterval(timer)
  }, [confidence, animated])
  
  // Déterminer la couleur selon le niveau de confiance
  const getConfidenceColor = (conf: number) => {
    if (conf >= 0.7) return 'epl-green' // Vert EPL pour haute confiance
    if (conf >= 0.5) return 'yellow-500' // Jaune pour confiance moyenne
    return 'red-500' // Rouge pour faible confiance
  }
  
  // Déterminer la couleur selon la prédiction
  const getPredictionColor = (pred: PredictionOutcome) => {
    switch (pred) {
      case 'H': return 'blue-600' // Domicile
      case 'D': return 'gray-600' // Nul  
      case 'A': return 'red-600'  // Extérieur
      default: return 'gray-500'
    }
  }
  
  const confidenceColor = getConfidenceColor(confidence)
  const predictionColor = getPredictionColor(prediction)
  const percentage = Math.round(displayConfidence * 100)
  
  // Tailles selon prop size
  const sizes = {
    sm: {
      height: 'h-2',
      text: 'text-xs',
      badge: 'text-xs px-2 py-1'
    },
    md: {
      height: 'h-3',
      text: 'text-sm',
      badge: 'text-sm px-3 py-1'
    },
    lg: {
      height: 'h-4',
      text: 'text-base',
      badge: 'text-base px-4 py-2'
    }
  }
  
  const sizeConfig = sizes[size]
  
  return (
    <div className={clsx('space-y-2', className)}>
      {/* Header avec pourcentage et badge prédiction */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <span className={clsx('font-mono font-bold', sizeConfig.text, `text-${confidenceColor}`)}>
            {percentage}%
          </span>
          <span className={clsx('text-gray-500', sizeConfig.text)}>
            confidence
          </span>
        </div>
        
        <div className={clsx(
          'rounded-full font-semibold',
          sizeConfig.badge,
          `bg-${predictionColor}`,
          'text-white'
        )}>
          {prediction === 'H' ? 'HOME' : prediction === 'D' ? 'DRAW' : 'AWAY'}
        </div>
      </div>
      
      {/* Progress bar */}
      <div className={clsx(
        'w-full bg-gray-200 rounded-full overflow-hidden',
        sizeConfig.height
      )}>
        <div 
          className={clsx(
            `bg-${confidenceColor}`,
            sizeConfig.height,
            'rounded-full transition-all duration-700 ease-out',
            animated && 'animate-pulse-slow'
          )}
          style={{ 
            width: `${displayConfidence * 100}%`,
            transition: animated ? 'width 0.8s ease-out' : 'none'
          }}
        />
      </div>
      
      {/* Labels indicatifs */}
      <div className="flex justify-between text-xs text-gray-400">
        <span>Low</span>
        <span>Medium</span>
        <span>High</span>
      </div>
    </div>
  )
}