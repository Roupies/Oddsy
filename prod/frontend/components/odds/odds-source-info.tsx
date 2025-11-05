import { OddsSource, MarketProbabilities, OddsSelectionMetadata } from '@/lib/types'
import { clsx } from 'clsx'
import * as Tooltip from '@radix-ui/react-tooltip'

interface OddsSourceInfoProps {
  odds_source: OddsSource
  market_probs_raw?: MarketProbabilities
  selection_metadata?: OddsSelectionMetadata
  missing_reason?: string
  className?: string
}

// Configuration sources avec icônes et couleurs
const SOURCE_CONFIG = {
  real: {
    icon: '🎯',
    label: 'Vraies odds',
    color: 'text-green-600',
    description: 'Données réelles validées'
  },
  unavailable: {
    icon: '❌',
    label: 'Indisponible', 
    color: 'text-red-600',
    description: 'Odds non disponibles'
  }
}

export function OddsSourceInfo({
  odds_source,
  market_probs_raw,
  selection_metadata,
  missing_reason,
  className
}: OddsSourceInfoProps) {
  
  const config = SOURCE_CONFIG[odds_source]
  
  // Contenu du tooltip
  const tooltipContent = (
    <div className="max-w-sm p-3 space-y-2">
      <div className="font-semibold text-sm flex items-center gap-1">
        <span>{config.icon}</span>
        <span>{config.label}</span>
      </div>
      
      <div className="text-xs text-gray-600">
        {config.description}
      </div>
      
      {/* Raison si indisponible */}
      {odds_source === 'unavailable' && missing_reason && (
        <div className="text-xs p-2 bg-red-50 rounded border border-red-200">
          <span className="font-medium text-red-800">Raison:</span>
          <div className="text-red-700 mt-1">{missing_reason}</div>
        </div>
      )}
      
      {/* Probabilités marché si disponibles */}
      {market_probs_raw && (
        <div className="text-xs p-2 bg-blue-50 rounded border border-blue-200">
          <div className="font-medium text-blue-800 mb-1">Probabilités marché:</div>
          <div className="grid grid-cols-3 gap-1 text-blue-700">
            <div>Home: {(market_probs_raw.home * 100).toFixed(1)}%</div>
            <div>Draw: {(market_probs_raw.draw * 100).toFixed(1)}%</div>
            <div>Away: {(market_probs_raw.away * 100).toFixed(1)}%</div>
          </div>
        </div>
      )}
      
      {/* Métadonnées sélection */}
      {selection_metadata && (
        <div className="text-xs p-2 bg-gray-50 rounded border border-gray-200">
          <div className="font-medium text-gray-800 mb-1">Métadonnées sélection:</div>
          <div className="space-y-1 text-gray-700">
            <div>Tier utilisé: <span className="font-medium">{selection_metadata.tier_used}</span></div>
            <div>Snapshots disponibles: <span className="font-medium">{selection_metadata.snapshots_available}</span></div>
            {selection_metadata.ko2h_cutoff && (
              <div>Cutoff KO-2h: <span className="font-mono text-xs">{new Date(selection_metadata.ko2h_cutoff).toLocaleString()}</span></div>
            )}
          </div>
        </div>
      )}
    </div>
  )
  
  return (
    <Tooltip.Provider>
      <Tooltip.Root>
        <Tooltip.Trigger asChild>
          <button 
            className={clsx(
              'inline-flex items-center gap-1 text-sm hover:opacity-75 transition-opacity',
              config.color,
              className
            )}
          >
            <span>{config.icon}</span>
            <span className="underline decoration-dotted">{config.label}</span>
            <span className="text-xs text-gray-400">ℹ️</span>
          </button>
        </Tooltip.Trigger>
        
        <Tooltip.Portal>
          <Tooltip.Content
            className="bg-white border border-gray-200 rounded-lg shadow-lg z-50"
            sideOffset={5}
          >
            {tooltipContent}
            <Tooltip.Arrow className="fill-white" />
          </Tooltip.Content>
        </Tooltip.Portal>
      </Tooltip.Root>
    </Tooltip.Provider>
  )
}

// Composant simple sans tooltip pour affichage inline
export function OddsSourceBadge({
  odds_source,
  missing_reason,
  compact = false,
  className
}: Pick<OddsSourceInfoProps, 'odds_source' | 'missing_reason' | 'className'> & { compact?: boolean }) {
  
  const config = SOURCE_CONFIG[odds_source]
  
  if (compact) {
    return (
      <span 
        className={clsx(
          'inline-flex items-center gap-1 text-xs',
          config.color,
          className
        )}
        title={missing_reason || config.description}
      >
        <span>{config.icon}</span>
        <span>{config.label}</span>
      </span>
    )
  }
  
  return (
    <div 
      className={clsx(
        'inline-flex items-center gap-1.5 px-2 py-1 rounded-md text-xs font-medium',
        odds_source === 'real' 
          ? 'bg-green-50 text-green-700 border border-green-200'
          : 'bg-red-50 text-red-700 border border-red-200',
        className
      )}
      title={missing_reason || config.description}
    >
      <span>{config.icon}</span>
      <span>{config.label}</span>
    </div>
  )
}

// Composant pour afficher le fallback bookmaker
export function BookmakerFallbackIndicator({
  tier_used,
  expected_tier = 'tier1',
  className
}: {
  tier_used?: 'tier1' | 'tier2' | 'tier3'
  expected_tier?: 'tier1' | 'tier2' | 'tier3'
  className?: string
}) {
  
  if (!tier_used || tier_used === expected_tier) {
    return null // Pas de fallback
  }
  
  const tierLabels = {
    tier1: 'Premier choix',
    tier2: 'Fallback',
    tier3: 'Fallback secondaire'
  }
  
  const tierColors = {
    tier1: 'bg-green-50 text-green-700 border-green-200',
    tier2: 'bg-yellow-50 text-yellow-700 border-yellow-200', 
    tier3: 'bg-orange-50 text-orange-700 border-orange-200'
  }
  
  return (
    <div className={clsx(
      'inline-flex items-center gap-1 px-2 py-1 rounded-md text-xs font-medium border',
      tierColors[tier_used],
      className
    )}>
      <span>🔄</span>
      <span>{tierLabels[tier_used]} utilisé</span>
    </div>
  )
}