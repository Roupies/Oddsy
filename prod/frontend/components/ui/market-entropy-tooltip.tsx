'use client'

import { useState } from 'react'
import { clsx } from 'clsx'
import { Info, TrendingUp, TrendingDown, Minus } from 'lucide-react'

interface MarketEntropyTooltipProps {
  entropy?: number
  overround?: number
  bookmakerName?: string
  marketProbsRaw?: {
    home: number
    draw: number
    away: number
  }
  className?: string
}

export function MarketEntropyTooltip({ 
  entropy, 
  overround, 
  bookmakerName = 'Unknown',
  marketProbsRaw,
  className = ''
}: MarketEntropyTooltipProps) {
  const [isVisible, setIsVisible] = useState(false)

  if (!entropy && !overround && !marketProbsRaw) {
    return null
  }

  const getEntropyLevel = (entropyValue: number) => {
    if (entropyValue >= 1.5) return { level: 'high', color: 'green', icon: TrendingUp, label: 'High Confidence' }
    if (entropyValue >= 1.0) return { level: 'medium', color: 'yellow', icon: Minus, label: 'Medium Confidence' }
    return { level: 'low', color: 'red', icon: TrendingDown, label: 'Low Confidence' }
  }

  const formatPercentage = (value: number) => `${(value * 100).toFixed(1)}%`
  const formatEntropy = (value: number) => value.toFixed(3)

  const entropyInfo = entropy ? getEntropyLevel(entropy) : null
  const EntropyIcon = entropyInfo?.icon || Info

  return (
    <div className="relative inline-block">
      {/* Trigger */}
      <button
        onMouseEnter={() => setIsVisible(true)}
        onMouseLeave={() => setIsVisible(false)}
        className={clsx(
          'inline-flex items-center space-x-1 text-xs text-gray-500 hover:text-gray-700 transition-colors',
          'focus:outline-none focus:ring-2 focus:ring-oddsy-primary/20 rounded',
          className
        )}
        aria-label="Market entropy information"
      >
        <Info className="w-3 h-3" />
        <span>Market Info</span>
      </button>

      {/* Tooltip */}
      {isVisible && (
        <div className={clsx(
          'absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 z-50',
          'bg-white border border-gray-200 rounded-lg shadow-lg p-4 min-w-64',
          'animate-in fade-in-0 zoom-in-95 duration-200'
        )}>
          {/* Flèche */}
          <div className="absolute top-full left-1/2 transform -translate-x-1/2">
            <div className="border-4 border-transparent border-t-white"></div>
            <div className="absolute top-[-5px] left-[-4px] border-4 border-transparent border-t-gray-200"></div>
          </div>

          {/* Contenu */}
          <div className="space-y-3">
            {/* Header */}
            <div className="flex items-center space-x-2 pb-2 border-b border-gray-100">
              <Info className="w-4 h-4 text-oddsy-primary" />
              <h4 className="font-semibold text-sm text-gray-900">Market Analysis</h4>
            </div>

            {/* Bookmaker */}
            <div className="flex justify-between items-center">
              <span className="text-xs text-gray-600">Source:</span>
              <span className="text-xs font-medium text-gray-900 bg-gray-100 px-2 py-1 rounded">
                {bookmakerName}
              </span>
            </div>

            {/* Market Probabilities */}
            {marketProbsRaw && (
              <div className="space-y-2">
                <h5 className="text-xs font-medium text-gray-700">Implied Probabilities</h5>
                <div className="grid grid-cols-3 gap-2 text-xs">
                  <div className="text-center p-2 bg-blue-50 rounded">
                    <div className="font-mono font-bold text-blue-700">
                      {formatPercentage(marketProbsRaw.home)}
                    </div>
                    <div className="text-blue-600">Home</div>
                  </div>
                  <div className="text-center p-2 bg-gray-50 rounded">
                    <div className="font-mono font-bold text-gray-700">
                      {formatPercentage(marketProbsRaw.draw)}
                    </div>
                    <div className="text-gray-600">Draw</div>
                  </div>
                  <div className="text-center p-2 bg-red-50 rounded">
                    <div className="font-mono font-bold text-red-700">
                      {formatPercentage(marketProbsRaw.away)}
                    </div>
                    <div className="text-red-600">Away</div>
                  </div>
                </div>
              </div>
            )}

            {/* Entropy */}
            {entropy && entropyInfo && (
              <div className="flex justify-between items-center">
                <div className="flex items-center space-x-1">
                  <span className="text-xs text-gray-600">Market Entropy:</span>
                  <EntropyIcon className={clsx('w-3 h-3', `text-${entropyInfo.color}-500`)} />
                </div>
                <div className="text-right">
                  <div className="text-xs font-mono font-bold">{formatEntropy(entropy)}</div>
                  <div className={clsx('text-xs', `text-${entropyInfo.color}-600`)}>
                    {entropyInfo.label}
                  </div>
                </div>
              </div>
            )}

            {/* Overround */}
            {overround && (
              <div className="flex justify-between items-center">
                <span className="text-xs text-gray-600">Overround:</span>
                <div className="text-right">
                  <div className="text-xs font-mono font-bold">{formatPercentage(overround)}</div>
                  <div className="text-xs text-gray-500">Bookmaker margin</div>
                </div>
              </div>
            )}

            {/* Explanation */}
            <div className="pt-2 border-t border-gray-100">
              <p className="text-xs text-gray-500 leading-relaxed">
                Market entropy measures prediction uncertainty. Higher values indicate more confident market consensus.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

interface SimpleMarketInfoProps {
  entropy?: number
  source?: string
  className?: string
}

export function SimpleMarketInfo({ entropy, source, className = '' }: SimpleMarketInfoProps) {
  if (!entropy && !source) return null

  const entropyInfo = entropy ? getEntropyLevel(entropy) : null
  const EntropyIcon = entropyInfo?.icon || Info

  return (
    <div className={clsx('inline-flex items-center space-x-1 text-xs text-gray-500', className)}>
      {entropy && entropyInfo && (
        <>
          <EntropyIcon className={clsx('w-3 h-3', `text-${entropyInfo.color}-500`)} />
          <span className="font-mono">{entropy.toFixed(2)}</span>
        </>
      )}
      {source && (
        <span className="bg-gray-100 px-1.5 py-0.5 rounded text-xs">
          {source}
        </span>
      )}
    </div>
  )
}

function getEntropyLevel(entropyValue: number) {
  if (entropyValue >= 1.5) return { level: 'high', color: 'green', icon: TrendingUp, label: 'High Confidence' }
  if (entropyValue >= 1.0) return { level: 'medium', color: 'yellow', icon: Minus, label: 'Medium Confidence' }
  return { level: 'low', color: 'red', icon: TrendingDown, label: 'Low Confidence' }
}