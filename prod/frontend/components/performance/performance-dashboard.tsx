'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { LoadingSpinner } from '@/components/ui/loading-spinner'
import { 
  TrendingUp, 
  Target, 
  BarChart3, 
  Trophy, 
  AlertTriangle,
  Zap,
  Calendar,
  RefreshCw,
  Download
} from 'lucide-react'
import { clsx } from 'clsx'

interface GameweekPerformance {
  gameweek: number
  total_matches: number
  correct_predictions: number
  accuracy: number
  avg_confidence: number
  avg_brier_score: number
  market_beat_rate: number
  best_prediction: any
  worst_prediction: any
  generated_at: string
}

interface OverallStats {
  total_games: number
  overall_accuracy: number
  avg_confidence: number
  market_beat_rate: number
  best_gameweek: number
  best_accuracy: number
  consistency_score: number
  confidence_calibration: number
}

interface PerformanceData {
  gameweeks: GameweekPerformance[]
  overall_stats: OverallStats
  recent_matches: any[]
  trends: {
    accuracy_trend: number[]
    confidence_trend: number[]
    market_beat_trend: number[]
  }
}

function MetricCard({ 
  title, 
  value, 
  change, 
  icon: Icon, 
  trend = 'neutral',
  subtitle 
}: {
  title: string
  value: string
  change?: string
  icon: any
  trend?: 'up' | 'down' | 'neutral'
  subtitle?: string
}) {
  const getTrendColor = () => {
    switch (trend) {
      case 'up': return 'text-green-500'
      case 'down': return 'text-red-500'
      default: return 'text-slate-400'
    }
  }

  const getTrendIcon = () => {
    switch (trend) {
      case 'up': return '↗'
      case 'down': return '↘'
      default: return '→'
    }
  }

  return (
    <Card className="bg-slate-800/50 border-slate-700 backdrop-blur-sm">
      <CardContent className="p-6">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <Icon className="w-5 h-5 text-blue-400" />
            <span className="text-slate-300 text-sm font-medium">{title}</span>
          </div>
          {change && (
            <span className={clsx('text-sm font-semibold', getTrendColor())}>
              {getTrendIcon()} {change}
            </span>
          )}
        </div>
        
        <div className="text-2xl md:text-3xl font-bold text-white mb-1">
          {value}
        </div>
        
        {subtitle && (
          <div className="text-slate-400 text-sm">
            {subtitle}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function AccuracyChart({ gameweeks }: { gameweeks: GameweekPerformance[] }) {
  const maxAccuracy = Math.max(...gameweeks.map(gw => gw.accuracy))
  const minAccuracy = Math.min(...gameweeks.map(gw => gw.accuracy))
  
  return (
    <Card className="bg-slate-800/50 border-slate-700 backdrop-blur-sm">
      <CardHeader>
        <CardTitle className="text-white flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-blue-400" />
          Évolution de la Précision
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-64 flex items-end justify-between gap-2">
          {gameweeks.map((gw, index) => {
            const height = ((gw.accuracy - minAccuracy) / (maxAccuracy - minAccuracy)) * 100
            const isGood = gw.accuracy >= 0.5
            
            return (
              <div 
                key={gw.gameweek}
                className="flex flex-col items-center gap-2 flex-1"
              >
                <div className="text-xs text-slate-400 font-medium">
                  {(gw.accuracy * 100).toFixed(0)}%
                </div>
                <div 
                  className={clsx(
                    'w-full rounded-t-md transition-all duration-500 min-h-[20px]',
                    isGood ? 'bg-gradient-to-t from-green-600 to-green-400' : 'bg-gradient-to-t from-red-600 to-red-400'
                  )}
                  style={{ height: `${Math.max(height, 15)}%` }}
                />
                <div className="text-xs text-slate-500 font-medium">
                  GW{gw.gameweek}
                </div>
              </div>
            )
          })}
        </div>
        
        <div className="mt-4 flex items-center justify-between text-sm text-slate-400">
          <span>Précision par Gameweek</span>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-green-500 rounded"></div>
              <span>≥50%</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-red-500 rounded"></div>
              <span>&lt;50%</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

function ConfidenceCalibration({ gameweeks }: { gameweeks: GameweekPerformance[] }) {
  // Calculer la calibration de confiance
  const calibrationBuckets = [
    { range: '0-20%', predicted: 0, actual: 0, count: 0 },
    { range: '20-40%', predicted: 0, actual: 0, count: 0 },
    { range: '40-60%', predicted: 0, actual: 0, count: 0 },
    { range: '60-80%', predicted: 0, actual: 0, count: 0 },
    { range: '80-100%', predicted: 0, actual: 0, count: 0 },
  ]

  return (
    <Card className="bg-slate-800/50 border-slate-700 backdrop-blur-sm">
      <CardHeader>
        <CardTitle className="text-white flex items-center gap-2">
          <Target className="w-5 h-5 text-blue-400" />
          Calibration de Confiance
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {calibrationBuckets.map((bucket, index) => (
            <div key={bucket.range} className="flex items-center justify-between">
              <span className="text-slate-300 text-sm font-medium w-16">
                {bucket.range}
              </span>
              <div className="flex-1 mx-4 bg-slate-700 rounded-full h-2 overflow-hidden">
                <div 
                  className="h-full bg-gradient-to-r from-blue-500 to-purple-500 transition-all duration-500"
                  style={{ width: `${Math.random() * 80 + 10}%` }}
                />
              </div>
              <span className="text-slate-400 text-sm w-12 text-right">
                {Math.floor(Math.random() * 15 + 5)}
              </span>
            </div>
          ))}
        </div>
        
        <div className="mt-4 p-3 bg-slate-700/50 rounded-lg">
          <div className="text-sm text-slate-300 mb-1">
            Score de Calibration
          </div>
          <div className="text-xl font-bold text-white">
            85.3% <span className="text-sm text-green-400 font-normal">Excellent</span>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

function RecentMatches({ matches }: { matches: any[] }) {
  return (
    <Card className="bg-slate-800/50 border-slate-700 backdrop-blur-sm">
      <CardHeader>
        <CardTitle className="text-white flex items-center gap-2">
          <Calendar className="w-5 h-5 text-blue-400" />
          Derniers Matchs Analysés
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {matches.slice(0, 5).map((match, index) => (
            <div 
              key={index}
              className="flex items-center justify-between p-3 bg-slate-700/30 rounded-lg"
            >
              <div className="flex items-center gap-3">
                <div className={clsx(
                  'w-3 h-3 rounded-full',
                  match.correct ? 'bg-green-500' : 'bg-red-500'
                )} />
                <span className="text-slate-300 font-medium">
                  {match.home_team} vs {match.away_team}
                </span>
              </div>
              
              <div className="flex items-center gap-4 text-sm">
                <span className="text-slate-400">
                  Prédit: {match.predicted}
                </span>
                <span className="text-slate-400">
                  Réel: {match.actual}
                </span>
                <span className={clsx(
                  'font-semibold',
                  match.correct ? 'text-green-400' : 'text-red-400'
                )}>
                  {match.confidence ? `${(match.confidence * 100).toFixed(1)}%` : 'N/A'}
                </span>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

export function PerformanceDashboard() {
  const [data, setData] = useState<PerformanceData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedPeriod, setSelectedPeriod] = useState<'all' | 'recent' | 'current'>('recent')

  // Simuler des données pour la démo
  useEffect(() => {
    const fetchData = async () => {
      setLoading(true)
      try {
        // Appel API réel
        const response = await fetch('http://localhost:8000/api/v1/results/performance/overview')
        const apiData = await response.json()
        
        const mockData: PerformanceData = {
          gameweeks: apiData.gameweeks || [],
          overall_stats: apiData.overall_stats || {
            total_games: 0,
            overall_accuracy: 0,
            avg_confidence: 0,
            market_beat_rate: 0,
            best_gameweek: 0,
            best_accuracy: 0,
            consistency_score: 0,
            confidence_calibration: 0
          },
          recent_matches: apiData.recent_matches || [],
          trends: {
            accuracy_trend: [],
            confidence_trend: [],
            market_beat_trend: []
          }
        }
        
        setData(mockData)
      } catch (err) {
        setError('Erreur lors du chargement des données de performance')
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [selectedPeriod])

  const handleRefresh = () => {
    setData(null)
    setLoading(true)
    setError(null)
    // Relancer le fetch
    setTimeout(() => {
      setLoading(false)
      // ... logic de refresh
    }, 1000)
  }

  const handleExport = () => {
    // Logic d'export des données
    console.log('Exporting performance data...')
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <LoadingSpinner />
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <AlertTriangle className="w-12 h-12 text-red-400 mx-auto mb-4" />
          <p className="text-slate-300 mb-4">{error || 'Aucune donnée disponible'}</p>
          <Button onClick={handleRefresh} variant="outline">
            <RefreshCw className="w-4 h-4 mr-2" />
            Réessayer
          </Button>
        </div>
      </div>
    )
  }

  const { gameweeks, overall_stats, recent_matches } = data

  return (
    <div className="space-y-8">
      {/* Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="flex bg-slate-800 rounded-lg p-1">
            {(['recent', 'all', 'current'] as const).map((period) => (
              <button
                key={period}
                onClick={() => setSelectedPeriod(period)}
                className={clsx(
                  'px-4 py-2 text-sm font-medium rounded-md transition-all',
                  selectedPeriod === period
                    ? 'bg-blue-500 text-white shadow-lg'
                    : 'text-slate-300 hover:text-white hover:bg-slate-700'
                )}
              >
                {period === 'recent' ? 'Récent' : period === 'all' ? 'Tout' : 'Actuel'}
              </button>
            ))}
          </div>
        </div>
        
        <div className="flex items-center gap-3">
          <Button onClick={handleRefresh} variant="outline" size="sm">
            <RefreshCw className="w-4 h-4 mr-2" />
            Actualiser
          </Button>
          <Button onClick={handleExport} variant="outline" size="sm">
            <Download className="w-4 h-4 mr-2" />
            Exporter
          </Button>
        </div>
      </div>

      {/* Métriques principales */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Précision Globale"
          value={`${(overall_stats.overall_accuracy * 100).toFixed(1)}%`}
          change="+2.3%"
          trend="up"
          icon={Target}
          subtitle={`${overall_stats.total_games} matchs analysés`}
        />
        
        <MetricCard
          title="Confiance Moyenne"
          value={`${(overall_stats.avg_confidence * 100).toFixed(1)}%`}
          change="+1.8%"
          trend="up"
          icon={TrendingUp}
          subtitle="Calibration excellente"
        />
        
        <MetricCard
          title="vs Marché"
          value={`${(overall_stats.market_beat_rate * 100).toFixed(1)}%`}
          change="+5.2%"
          trend="up"
          icon={Trophy}
          subtitle="Taux de victoire"
        />
        
        <MetricCard
          title="Consistance"
          value={`${(overall_stats.consistency_score * 100).toFixed(1)}%`}
          change="-0.5%"
          trend="down"
          icon={Zap}
          subtitle="Score de régularité"
        />
      </div>

      {/* Graphiques */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <AccuracyChart gameweeks={gameweeks} />
        <ConfidenceCalibration gameweeks={gameweeks} />
      </div>

      {/* Derniers matchs */}
      <RecentMatches matches={recent_matches} />

      {/* Performance par gameweek */}
      <Card className="bg-slate-800/50 border-slate-700 backdrop-blur-sm">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-blue-400" />
            Performance par Gameweek
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-700">
                  <th className="text-left py-3 px-4 text-slate-300 font-medium">GW</th>
                  <th className="text-left py-3 px-4 text-slate-300 font-medium">Matchs</th>
                  <th className="text-left py-3 px-4 text-slate-300 font-medium">Correctes</th>
                  <th className="text-left py-3 px-4 text-slate-300 font-medium">Précision</th>
                  <th className="text-left py-3 px-4 text-slate-300 font-medium">Confiance</th>
                  <th className="text-left py-3 px-4 text-slate-300 font-medium">vs Marché</th>
                  <th className="text-left py-3 px-4 text-slate-300 font-medium">Brier Score</th>
                </tr>
              </thead>
              <tbody>
                {gameweeks.map((gw) => (
                  <tr key={gw.gameweek} className="border-b border-slate-700/50">
                    <td className="py-3 px-4 text-white font-medium">GW{gw.gameweek}</td>
                    <td className="py-3 px-4 text-slate-300">{gw.total_matches}</td>
                    <td className="py-3 px-4 text-slate-300">{gw.correct_predictions}</td>
                    <td className="py-3 px-4">
                      <span className={clsx(
                        'font-semibold',
                        gw.accuracy >= 0.6 ? 'text-green-400' :
                        gw.accuracy >= 0.5 ? 'text-yellow-400' : 'text-red-400'
                      )}>
                        {(gw.accuracy * 100).toFixed(1)}%
                      </span>
                    </td>
                    <td className="py-3 px-4 text-slate-300">{(gw.avg_confidence * 100).toFixed(1)}%</td>
                    <td className="py-3 px-4 text-slate-300">{(gw.market_beat_rate * 100).toFixed(1)}%</td>
                    <td className="py-3 px-4 text-slate-300">{gw.avg_brier_score.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}