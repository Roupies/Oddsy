'use client'

import { useEffect, useState } from 'react'
import { clsx } from 'clsx'
import { Clock, CheckCircle, AlertTriangle, XCircle } from 'lucide-react'

interface Ko2hBadgeProps {
  ko2hOk?: boolean
  kickoffUtc?: string // Format ISO string "2025-10-15T15:00:00Z"
  snapshotUtc?: string // Timestamp du snapshot odds
  className?: string
  size?: 'sm' | 'md' | 'lg'
  showCountdown?: boolean
}

interface TimeRemaining {
  hours: number
  minutes: number
  total_minutes: number
}

export function Ko2hBadge({ 
  ko2hOk, 
  kickoffUtc, 
  snapshotUtc,
  className = '',
  size = 'md',
  showCountdown = true
}: Ko2hBadgeProps) {
  const [timeToKickoff, setTimeToKickoff] = useState<TimeRemaining | null>(null)
  const [currentTime, setCurrentTime] = useState(new Date())
  
  // Mise à jour du temps en temps réel
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentTime(new Date())
    }, 1000) // Update every second
    
    return () => clearInterval(interval)
  }, [])
  
  // Calculer le temps restant jusqu'au kickoff
  useEffect(() => {
    if (!kickoffUtc) {
      setTimeToKickoff(null)
      return
    }
    
    const kickoff = new Date(kickoffUtc)
    const now = currentTime
    const diffMs = kickoff.getTime() - now.getTime()
    const totalMinutes = Math.floor(diffMs / (1000 * 60))
    
    if (totalMinutes < 0) {
      setTimeToKickoff({ hours: 0, minutes: 0, total_minutes: 0 })
    } else {
      const hours = Math.floor(totalMinutes / 60)
      const minutes = totalMinutes % 60
      setTimeToKickoff({ hours, minutes, total_minutes: totalMinutes })
    }
  }, [kickoffUtc, currentTime])
  
  // Déterminer le statut et la couleur
  const getStatus = () => {
    if (!timeToKickoff) {
      return {
        status: 'unknown',
        color: 'gray',
        icon: Clock,
        label: 'Unknown',
        message: 'Kickoff time not available'
      }
    }
    
    const { total_minutes } = timeToKickoff
    
    // Déjà joué
    if (total_minutes <= 0) {
      return {
        status: 'finished',
        color: 'gray',
        icon: CheckCircle,
        label: 'Finished',
        message: 'Match completed'
      }
    }
    
    // Plus de 2h avant coup d'envoi
    if (total_minutes > 120) {
      return {
        status: 'pending',
        color: 'blue',
        icon: Clock,
        label: 'Pending',
        message: `${timeToKickoff.hours}h ${timeToKickoff.minutes}m to kickoff`
      }
    }
    
    // Moins de 2h - vérifier ko2hOk
    if (ko2hOk) {
      return {
        status: 'ready',
        color: 'green',
        icon: CheckCircle,
        label: 'Ready',
        message: `KO-2h validated • ${timeToKickoff.hours}h ${timeToKickoff.minutes}m left`
      }
    } else {
      return {
        status: 'violation',
        color: 'red',
        icon: XCircle,
        label: 'Violation',
        message: `KO-2h deadline passed • ${timeToKickoff.hours}h ${timeToKickoff.minutes}m left`
      }
    }
  }
  
  const statusInfo = getStatus()
  const Icon = statusInfo.icon
  
  // Configuration tailles
  const sizeConfig = {
    sm: {
      padding: 'px-2 py-1',
      text: 'text-xs',
      icon: 'w-3 h-3'
    },
    md: {
      padding: 'px-3 py-1.5',
      text: 'text-sm',
      icon: 'w-4 h-4'
    },
    lg: {
      padding: 'px-4 py-2',
      text: 'text-base',
      icon: 'w-5 h-5'
    }
  }
  
  const config = sizeConfig[size]
  
  // Couleurs selon statut
  const colorClasses = {
    green: 'bg-green-100 text-green-800 border-green-200',
    blue: 'bg-blue-100 text-blue-800 border-blue-200',
    red: 'bg-red-100 text-red-800 border-red-200',
    gray: 'bg-gray-100 text-gray-800 border-gray-200'
  }
  
  return (
    <div className={clsx(
      'inline-flex items-center gap-2 rounded-full border font-medium',
      config.padding,
      config.text,
      colorClasses[statusInfo.color as keyof typeof colorClasses],
      className
    )}>
      <Icon className={clsx(config.icon, statusInfo.status === 'ready' && 'animate-pulse')} />
      
      <span>{statusInfo.label}</span>
      
      {showCountdown && timeToKickoff && timeToKickoff.total_minutes > 0 && (
        <span className="font-mono">
          {timeToKickoff.hours}h {timeToKickoff.minutes}m
        </span>
      )}
      
      {/* Indicateur snapshot si disponible */}
      {snapshotUtc && timeToKickoff && timeToKickoff.total_minutes > 120 && (
        <div className="flex items-center gap-1 ml-1 pl-2 border-l border-current/20">
          <div className="w-1.5 h-1.5 bg-current rounded-full animate-pulse" />
          <span className="text-xs opacity-75">Live</span>
        </div>
      )}
    </div>
  )
}

// Hook utilitaire pour le statut KO-2h
export function useKo2hCountdown(kickoffUtc?: string) {
  const [countdown, setCountdown] = useState<TimeRemaining | null>(null)
  
  useEffect(() => {
    if (!kickoffUtc) return
    
    const updateCountdown = () => {
      const kickoff = new Date(kickoffUtc)
      const now = new Date()
      const diffMs = kickoff.getTime() - now.getTime()
      const totalMinutes = Math.floor(diffMs / (1000 * 60))
      
      if (totalMinutes < 0) {
        setCountdown({ hours: 0, minutes: 0, total_minutes: 0 })
      } else {
        const hours = Math.floor(totalMinutes / 60)
        const minutes = totalMinutes % 60
        setCountdown({ hours, minutes, total_minutes: totalMinutes })
      }
    }
    
    updateCountdown()
    const interval = setInterval(updateCountdown, 1000)
    
    return () => clearInterval(interval)
  }, [kickoffUtc])
  
  return countdown
}