import { OddsSnapshot, BookmakerId } from '@/lib/types'
import { formatDistanceToNow } from 'date-fns'
import { fr } from 'date-fns/locale'
import { clsx } from 'clsx'

interface OddsSnapshotBadgeProps {
  snapshot: OddsSnapshot
  compact?: boolean
  className?: string
}

// Configuration des bookmakers avec couleurs et labels
const BOOKMAKER_CONFIG: Record<BookmakerId, {
  label: string
  color: string
  tier: 'tier1' | 'tier2' | 'tier3'
}> = {
  bet365: { label: 'Bet365', color: 'bg-green-100 text-green-800 border-green-200', tier: 'tier1' },
  pinnacle: { label: 'Pinnacle', color: 'bg-blue-100 text-blue-800 border-blue-200', tier: 'tier1' },
  betfair: { label: 'Betfair', color: 'bg-purple-100 text-purple-800 border-purple-200', tier: 'tier2' },
  william_hill: { label: 'W.Hill', color: 'bg-orange-100 text-orange-800 border-orange-200', tier: 'tier2' },
  ladbrokes: { label: 'Ladbrokes', color: 'bg-red-100 text-red-800 border-red-200', tier: 'tier3' },
  unibet: { label: 'Unibet', color: 'bg-yellow-100 text-yellow-800 border-yellow-200', tier: 'tier3' }
}

// Configuration confidence avec couleurs
const CONFIDENCE_CONFIG = {
  high: { color: 'ring-green-500', icon: '🟢' },
  medium: { color: 'ring-yellow-500', icon: '🟡' },
  low: { color: 'ring-red-500', icon: '🔴' }
}

export function OddsSnapshotBadge({ snapshot, compact = false, className }: OddsSnapshotBadgeProps) {
  const bookmakerConfig = BOOKMAKER_CONFIG[snapshot.bookmaker]
  const confidenceConfig = CONFIDENCE_CONFIG[snapshot.market_confidence]
  
  // Formater timestamp en local
  const snapshotDate = new Date(snapshot.snapshot_utc)
  const timeAgo = formatDistanceToNow(snapshotDate, { locale: fr, addSuffix: true })
  
  // Formater overround en pourcentage
  const overroundPct = ((snapshot.overround - 1) * 100).toFixed(1)
  
  if (compact) {
    return (
      <div className={clsx(
        'inline-flex items-center gap-1 px-2 py-1 rounded-md text-xs font-medium border',
        bookmakerConfig.color,
        `ring-1 ${confidenceConfig.color}`,
        className
      )}>
        <span>{bookmakerConfig.label}</span>
        <span className="text-gray-500">•</span>
        <span>{overroundPct}%</span>
        <span title={`Market confidence: ${snapshot.market_confidence}`}>
          {confidenceConfig.icon}
        </span>
      </div>
    )
  }
  
  return (
    <div className={clsx(
      'inline-flex flex-col gap-1 px-3 py-2 rounded-lg border',
      bookmakerConfig.color,
      `ring-2 ${confidenceConfig.color}`,
      className
    )}>
      {/* Header avec bookmaker et confidence */}
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5">
          <span className="font-semibold text-sm">
            {bookmakerConfig.label}
          </span>
          <span className="text-xs px-1.5 py-0.5 rounded bg-white/50">
            {bookmakerConfig.tier}
          </span>
        </div>
        <div className="flex items-center gap-1" title={`Market confidence: ${snapshot.market_confidence}`}>
          <span className="text-xs">{snapshot.market_confidence}</span>
          <span>{confidenceConfig.icon}</span>
        </div>
      </div>
      
      {/* Détails snapshot */}
      <div className="flex items-center justify-between gap-2 text-xs">
        <div className="flex items-center gap-1 text-gray-600">
          <span>📸</span>
          <span title={snapshot.snapshot_utc}>{timeAgo}</span>
        </div>
        <div className="flex items-center gap-1 font-medium">
          <span>📊</span>
          <span title={`Overround: ${snapshot.overround.toFixed(4)}`}>
            {overroundPct}%
          </span>
        </div>
      </div>
    </div>
  )
}

// Composant pour afficher quand snapshot manquant
export function MissingSnapshotBadge({ 
  reason, 
  compact = false, 
  className 
}: { 
  reason?: string
  compact?: boolean
  className?: string 
}) {
  if (compact) {
    return (
      <div className={clsx(
        'inline-flex items-center gap-1 px-2 py-1 rounded-md text-xs font-medium',
        'bg-gray-100 text-gray-600 border border-gray-200',
        className
      )}>
        <span>❌</span>
        <span>No odds</span>
      </div>
    )
  }
  
  return (
    <div className={clsx(
      'inline-flex flex-col gap-1 px-3 py-2 rounded-lg border',
      'bg-gray-50 text-gray-600 border-gray-200',
      className
    )}>
      <div className="flex items-center gap-1.5">
        <span>❌</span>
        <span className="font-semibold text-sm">Odds indisponibles</span>
      </div>
      {reason && (
        <div className="text-xs text-gray-500">
          {reason}
        </div>
      )}
    </div>
  )
}