'use client'

import { useAvailableRounds } from '@/hooks/use-predictions'
import { AlertTriangle, TrendingUp, Target, Clock } from 'lucide-react'
import Link from 'next/link'

export function DashboardOverview() {
  const { data: availableRounds, isLoading, error } = useAvailableRounds()
  
  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-oddsy-primary"></div>
        <span className="ml-2 text-gray-600">Loading overview...</span>
      </div>
    )
  }
  
  if (error) {
    return (
      <div className="card bg-red-50 border-red-200 p-4">
        <div className="flex items-center space-x-2 text-red-800">
          <AlertTriangle className="h-5 w-5" />
          <span className="font-medium">Failed to load overview</span>
        </div>
        <p className="text-sm text-red-600 mt-1">{error.message}</p>
      </div>
    )
  }
  
  // Mock data for demonstration - in real app would come from API
  const stats = {
    currentRound: 7,
    totalRounds: availableRounds?.rounds?.length || 0,
    lastUpdated: new Date().toISOString(),
    accuracy: {
      baseline: 53.5,
      cascade: 50.0,
      ensemble: 55.2
    },
    recentPredictions: 10,
    confidenceAvg: 67.8
  }
  
  return (
    <div className="space-y-6">
      {/* Quick Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="card p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Current Round</p>
              <p className="text-2xl font-bold text-oddsy-primary">J{stats.currentRound}</p>
            </div>
            <Clock className="h-8 w-8 text-gray-400" />
          </div>
        </div>
        
        <div className="card p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Available Rounds</p>
              <p className="text-2xl font-bold text-gray-900">{stats.totalRounds}</p>
            </div>
            <TrendingUp className="h-8 w-8 text-gray-400" />
          </div>
        </div>
        
        <div className="card p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Ensemble Accuracy</p>
              <p className="text-2xl font-bold text-green-600">{stats.accuracy.ensemble}%</p>
            </div>
            <Target className="h-8 w-8 text-gray-400" />
          </div>
        </div>
        
        <div className="card p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Avg Confidence</p>
              <p className="text-2xl font-bold text-blue-600">{stats.confidenceAvg}%</p>
            </div>
            <TrendingUp className="h-8 w-8 text-gray-400" />
          </div>
        </div>
      </div>
      
      {/* Model Performance Comparison */}
      <div className="card p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Model Performance</h3>
        <div className="space-y-4">
          {/* Enhanced Baseline */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-3 h-3 bg-green-500 rounded-full"></div>
              <span className="font-medium">Enhanced Baseline v3.0</span>
              <span className="badge bg-green-100 text-green-800">Champion</span>
            </div>
            <div className="text-right">
              <div className="font-bold text-green-600">{stats.accuracy.baseline}%</div>
              <div className="text-xs text-gray-500">CV Accuracy</div>
            </div>
          </div>
          
          {/* Cascade Champion */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
              <span className="font-medium">Cascade Champion v2.1</span>
              <span className="badge bg-blue-100 text-blue-800">Specialist</span>
            </div>
            <div className="text-right">
              <div className="font-bold text-blue-600">{stats.accuracy.cascade}%</div>
              <div className="text-xs text-gray-500">EPL 2025-26</div>
            </div>
          </div>
          
          {/* Ensemble */}
          <div className="flex items-center justify-between pt-2 border-t">
            <div className="flex items-center space-x-3">
              <div className="w-3 h-3 bg-oddsy-primary rounded-full"></div>
              <span className="font-medium">Dual Champions Ensemble</span>
              <span className="badge bg-oddsy-primary text-white">Production</span>
            </div>
            <div className="text-right">
              <div className="font-bold text-oddsy-primary">{stats.accuracy.ensemble}%</div>
              <div className="text-xs text-gray-500">Combined</div>
            </div>
          </div>
        </div>
      </div>
      
      {/* Quick Actions */}
      <div className="card p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <Link href="/matchday/7" className="button-card">
            <div className="flex items-center space-x-3">
              <div className="text-2xl">⚽</div>
              <div>
                <div className="font-medium">View GW7 Predictions</div>
                <div className="text-sm text-gray-600">Latest round predictions</div>
              </div>
            </div>
          </Link>
          
          <Link href="/models" className="button-card">
            <div className="flex items-center space-x-3">
              <div className="text-2xl">📊</div>
              <div>
                <div className="font-medium">Model Performance</div>
                <div className="text-sm text-gray-600">Detailed metrics</div>
              </div>
            </div>
          </Link>
          
          <Link href="/pipeline" className="button-card">
            <div className="flex items-center space-x-3">
              <div className="text-2xl">⚙️</div>
              <div>
                <div className="font-medium">Pipeline Status</div>
                <div className="text-sm text-gray-600">System health</div>
              </div>
            </div>
          </Link>
          
          <div className="button-card opacity-50 cursor-not-allowed">
            <div className="flex items-center space-x-3">
              <div className="text-2xl">🔄</div>
              <div>
                <div className="font-medium">Trigger Refresh</div>
                <div className="text-sm text-gray-600">Manual update</div>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      {/* Last Updated */}
      <div className="text-center text-sm text-gray-500">
        Last updated: {new Date(stats.lastUpdated).toLocaleString()}
      </div>
    </div>
  )
}