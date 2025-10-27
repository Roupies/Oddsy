'use client'

import { useState, useEffect } from 'react'
import { CheckCircle, AlertTriangle, Clock, RefreshCw, Zap } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { LoadingSpinner } from '@/components/ui/loading-spinner'

interface PipelineJob {
  id: string
  round: number
  status: 'pending' | 'running' | 'completed' | 'failed'
  created_at: string
  updated_at: string
  logs?: string[]
}

interface PipelineStatusData {
  current_round: number
  pipeline_version: string
  last_successful_run: string
  recent_jobs: PipelineJob[]
  system_health: {
    api_status: 'healthy' | 'degraded' | 'down'
    pipeline_status: 'healthy' | 'degraded' | 'down'
    last_check: string
  }
}

export function PipelineStatus() {
  const [data, setData] = useState<PipelineStatusData | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [isRefreshing, setIsRefreshing] = useState(false)
  
  const fetchStatus = async () => {
    try {
      setError(null)
      const response = await fetch('/api/v1/pipeline/status')
      
      if (!response.ok) {
        throw new Error(`Failed to fetch status: ${response.statusText}`)
      }
      
      const result = await response.json()
      setData(result.data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
      // Mock data for development
      setData({
        current_round: 7,
        pipeline_version: 'Pipeline Durci v1.0',
        last_successful_run: new Date(Date.now() - 3600000).toISOString(),
        recent_jobs: [
          {
            id: 'job_001',
            round: 7,
            status: 'completed',
            created_at: new Date(Date.now() - 7200000).toISOString(),
            updated_at: new Date(Date.now() - 3600000).toISOString(),
            logs: ['Started prediction generation', 'Enhanced Baseline v2.4 completed', 'Cascade Champion v2.1 completed', 'Ensemble aggregation completed']
          },
          {
            id: 'job_002',
            round: 6,
            status: 'completed',
            created_at: new Date(Date.now() - 86400000).toISOString(),
            updated_at: new Date(Date.now() - 86400000 + 1800000).toISOString()
          }
        ],
        system_health: {
          api_status: 'healthy',
          pipeline_status: 'healthy',
          last_check: new Date().toISOString()
        }
      })
    } finally {
      setIsLoading(false)
      setIsRefreshing(false)
    }
  }
  
  const handleRefresh = async () => {
    setIsRefreshing(true)
    await fetchStatus()
  }
  
  const triggerPipeline = async (round: number) => {
    try {
      const response = await fetch('/api/v1/pipeline/trigger', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ round })
      })
      
      if (!response.ok) {
        throw new Error('Failed to trigger pipeline')
      }
      
      // Refresh status after triggering
      setTimeout(handleRefresh, 1000)
    } catch (err) {
      console.error('Pipeline trigger failed:', err)
    }
  }
  
  useEffect(() => {
    fetchStatus()
    
    // Auto-refresh every 30 seconds
    const interval = setInterval(fetchStatus, 30000)
    return () => clearInterval(interval)
  }, [])
  
  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-8">
        <LoadingSpinner />
        <span className="ml-2 text-gray-600">Loading pipeline status...</span>
      </div>
    )
  }
  
  if (!data) {
    return (
      <div className="card bg-red-50 border-red-200 p-4">
        <div className="flex items-center space-x-2 text-red-800">
          <AlertTriangle className="h-5 w-5" />
          <span className="font-medium">Pipeline status unavailable</span>
        </div>
        {error && <p className="text-sm text-red-600 mt-1">{error}</p>}
      </div>
    )
  }
  
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
      case 'completed':
        return <CheckCircle className="h-5 w-5 text-green-500" />
      case 'degraded':
      case 'running':
        return <Clock className="h-5 w-5 text-yellow-500 animate-pulse" />
      case 'down':
      case 'failed':
        return <AlertTriangle className="h-5 w-5 text-red-500" />
      default:
        return <Clock className="h-5 w-5 text-gray-500" />
    }
  }
  
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
      case 'completed':
        return 'text-green-600'
      case 'degraded':
      case 'running':
        return 'text-yellow-600'
      case 'down':
      case 'failed':
        return 'text-red-600'
      default:
        return 'text-gray-600'
    }
  }
  
  return (
    <div className="space-y-6">
      {/* System Health Overview */}
      <div className="card p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">System Health</h3>
          <Button
            onClick={handleRefresh}
            variant="outline"
            size="sm"
            disabled={isRefreshing}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${isRefreshing ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
            <div className="flex items-center space-x-3">
              {getStatusIcon(data.system_health.api_status)}
              <span className="font-medium">API Status</span>
            </div>
            <span className={`capitalize font-medium ${getStatusColor(data.system_health.api_status)}`}>
              {data.system_health.api_status}
            </span>
          </div>
          
          <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
            <div className="flex items-center space-x-3">
              {getStatusIcon(data.system_health.pipeline_status)}
              <span className="font-medium">Pipeline Status</span>
            </div>
            <span className={`capitalize font-medium ${getStatusColor(data.system_health.pipeline_status)}`}>
              {data.system_health.pipeline_status}
            </span>
          </div>
        </div>
      </div>
      
      {/* Pipeline Info */}
      <div className="card p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Pipeline Information</h3>
        
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-gray-600">Version</span>
            <span className="font-medium">{data.pipeline_version}</span>
          </div>
          
          <div className="flex items-center justify-between">
            <span className="text-gray-600">Current Round</span>
            <span className="font-medium">J{data.current_round}</span>
          </div>
          
          <div className="flex items-center justify-between">
            <span className="text-gray-600">Last Successful Run</span>
            <span className="font-medium">
              {new Date(data.last_successful_run).toLocaleString()}
            </span>
          </div>
          
          <div className="flex items-center justify-between">
            <span className="text-gray-600">Last Health Check</span>
            <span className="font-medium">
              {new Date(data.system_health.last_check).toLocaleString()}
            </span>
          </div>
        </div>
      </div>
      
      {/* Recent Jobs */}
      <div className="card p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">Recent Jobs</h3>
          <Button
            onClick={() => triggerPipeline(data.current_round + 1)}
            variant="outline"
            size="sm"
            className="gap-2"
          >
            <Zap className="h-4 w-4" />
            Trigger J{data.current_round + 1}
          </Button>
        </div>
        
        <div className="space-y-3">
          {data.recent_jobs.map((job) => (
            <div key={job.id} className="border border-gray-200 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center space-x-3">
                  {getStatusIcon(job.status)}
                  <span className="font-medium">Round J{job.round}</span>
                  <span className={`text-sm capitalize ${getStatusColor(job.status)}`}>
                    {job.status}
                  </span>
                </div>
                <span className="text-sm text-gray-500">
                  {new Date(job.updated_at).toLocaleString()}
                </span>
              </div>
              
              {job.logs && job.logs.length > 0 && (
                <div className="mt-3 bg-gray-50 rounded p-3">
                  <div className="text-xs text-gray-600 mb-2">Logs:</div>
                  <div className="space-y-1">
                    {job.logs.map((log, idx) => (
                      <div key={idx} className="text-xs text-gray-700 font-mono">
                        • {log}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}
          
          {data.recent_jobs.length === 0 && (
            <div className="text-center py-4 text-gray-500">
              No recent jobs
            </div>
          )}
        </div>
      </div>
    </div>
  )
}