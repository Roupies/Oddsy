import { Ko2hStatus } from '@/lib/types'
import { clsx } from 'clsx'

interface Ko2hStatusIndicatorProps {
  ko2h_ok: boolean
  missing_reason?: string
  minutes_to_kickoff?: number
  compact?: boolean
  className?: string
}

// Configuration des status avec couleurs et icônes
const STATUS_CONFIG = {
  ok: {
    color: 'bg-green-100 text-green-800 border-green-200',
    icon: '✅',
    label: 'KO-2h OK'
  },
  warning: {
    color: 'bg-yellow-100 text-yellow-800 border-yellow-200', 
    icon: '⚠️',
    label: 'KO-2h Warning'
  },
  violation: {
    color: 'bg-red-100 text-red-800 border-red-200',
    icon: '❌', 
    label: 'KO-2h Violation'
  }
}

export function Ko2hStatusIndicator({ 
  ko2h_ok, 
  missing_reason, 
  minutes_to_kickoff,
  compact = false, 
  className 
}: Ko2hStatusIndicatorProps) {
  
  // Déterminer le status basé sur les inputs
  let status: keyof typeof STATUS_CONFIG = 'ok'
  let message = ''
  
  if (!ko2h_ok) {
    status = 'violation'
    message = missing_reason || 'KO-2h constraint not respected'
  } else if (minutes_to_kickoff !== undefined && minutes_to_kickoff < 150) {
    // Warning si moins de 2h30 avant kickoff (proche de la limite)
    status = 'warning'
    message = `Proche limite: ${Math.round(minutes_to_kickoff)}min avant kickoff`
  } else {
    message = minutes_to_kickoff 
      ? `✅ OK - ${Math.round(minutes_to_kickoff)}min avant kickoff`
      : '✅ KO-2h constraint respecté'
  }
  
  const config = STATUS_CONFIG[status]
  
  if (compact) {
    return (
      <div 
        className={clsx(
          'inline-flex items-center gap-1 px-2 py-1 rounded-md text-xs font-medium border',
          config.color,
          className
        )}
        title={message}
      >
        <span>{config.icon}</span>
        <span>KO-2h</span>
      </div>
    )
  }
  
  return (
    <div className={clsx(
      'inline-flex items-center gap-2 px-3 py-2 rounded-lg border',
      config.color,
      className
    )}>
      <span className="text-lg">{config.icon}</span>
      <div className="flex flex-col">
        <span className="font-semibold text-sm">{config.label}</span>
        <span className="text-xs opacity-75">{message}</span>
      </div>
    </div>
  )
}

// Composant bandeau KO-2h pour MatchCard  
export function Ko2hBanner({
  ko2h_ok,
  missing_reason,
  minutes_to_kickoff,
  className
}: Ko2hStatusIndicatorProps) {
  
  let status: keyof typeof STATUS_CONFIG = 'ok'
  let bannerMessage = ''
  
  if (!ko2h_ok) {
    status = 'violation'
    bannerMessage = `🚨 ${missing_reason || 'Données KO-2h non conformes'}`
  } else if (minutes_to_kickoff !== undefined && minutes_to_kickoff < 150) {
    status = 'warning'
    bannerMessage = `⚠️ Proche limite KO-2h (${Math.round(minutes_to_kickoff)}min)`
  } else {
    status = 'ok'
    bannerMessage = minutes_to_kickoff 
      ? `✅ Données validées (${Math.round(minutes_to_kickoff)}min avant KO)`
      : '✅ Contrainte KO-2h respectée'
  }
  
  const config = STATUS_CONFIG[status]
  
  return (
    <div className={clsx(
      'w-full px-3 py-2 rounded-t-lg border-b text-center text-sm font-medium',
      config.color,
      className
    )}>
      {bannerMessage}
    </div>
  )
}

// Hook utilitaire pour calculer le status KO-2h
export function useKo2hStatus(
  ko2h_ok: boolean,
  kickoff_utc?: string,
  missing_reason?: string
): Ko2hStatus {
  let minutes_to_kickoff: number | undefined
  
  if (kickoff_utc) {
    try {
      const kickoffDate = new Date(kickoff_utc)
      const now = new Date()
      minutes_to_kickoff = Math.max(0, (kickoffDate.getTime() - now.getTime()) / (1000 * 60))
    } catch (e) {
      // Erreur parsing date
    }
  }
  
  return {
    ok: ko2h_ok,
    reason: missing_reason,
    minutes_to_kickoff
  }
}