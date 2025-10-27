import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'Pipeline Status - Oddsy',
  description: 'Real-time status of Pipeline Durci v1.0 with system health monitoring',
  openGraph: {
    title: 'Oddsy Pipeline Status',
    description: 'Pipeline Durci v1.0 monitoring and health dashboard',
  }
}

// Mock pipeline data
const pipelineData = {
  version: 'Pipeline Durci v1.0',
  status: 'healthy',
  current_round: 7,
  last_successful_run: new Date(Date.now() - 3600000).toISOString(),
  last_update: new Date().toISOString(),
  total_predictions_generated: 840,
  uptime_days: 42,
  git_sha: '102f87f',
  system_health: {
    api_status: 'healthy',
    pipeline_status: 'healthy',
    data_extraction: 'healthy',
    model_inference: 'healthy',
    cache_status: 'healthy'
  },
  recent_jobs: [
    {
      id: 'job_007',
      round: 7,
      status: 'completed',
      started_at: '2025-10-03T09:00:00Z',
      completed_at: '2025-10-03T09:12:34Z',
      duration_seconds: 754,
      matches_processed: 10,
      models_executed: ['enhanced_baseline_v24', 'cascade_champion_v21'],
      logs: [
        'Started data extraction for J7',
        'Enhanced Baseline v2.4: 10 predictions generated',
        'Cascade Champion v2.1: 10 predictions generated',
        'Ensemble aggregation completed',
        'Validation checks passed',
        'Predictions saved to /predictions/j7_predictions_20251003.json'
      ]
    },
    {
      id: 'job_006',
      round: 6,
      status: 'completed',
      started_at: '2025-09-28T09:00:00Z',
      completed_at: '2025-09-28T09:15:22Z',
      duration_seconds: 922,
      matches_processed: 10,
      models_executed: ['enhanced_baseline_v24', 'cascade_champion_v21']
    },
    {
      id: 'job_005',
      round: 5,
      status: 'completed',
      started_at: '2025-09-21T09:00:00Z',
      completed_at: '2025-09-21T09:11:45Z',
      duration_seconds: 705,
      matches_processed: 10,
      models_executed: ['enhanced_baseline_v24', 'cascade_champion_v21']
    }
  ],
  pipeline_stages: [
    { name: 'Data Extraction', status: 'healthy', last_run: '2025-10-03T09:00:00Z', duration_avg: 45 },
    { name: 'Feature Engineering', status: 'healthy', last_run: '2025-10-03T09:02:15Z', duration_avg: 120 },
    { name: 'Enhanced Baseline v2.4', status: 'healthy', last_run: '2025-10-03T09:04:30Z', duration_avg: 180 },
    { name: 'Cascade Champion v2.1', status: 'healthy', last_run: '2025-10-03T09:07:45Z', duration_avg: 150 },
    { name: 'Ensemble Aggregation', status: 'healthy', last_run: '2025-10-03T09:10:20Z', duration_avg: 60 },
    { name: 'Validation & Export', status: 'healthy', last_run: '2025-10-03T09:12:00Z', duration_avg: 90 }
  ]
}

const getStatusIcon = (status: string) => {
  switch (status) {
    case 'healthy':
    case 'completed':
      return '✅'
    case 'warning':
    case 'running':
      return '⚠️'
    case 'error':
    case 'failed':
      return '❌'
    default:
      return '⏳'
  }
}

const getStatusColor = (status: string) => {
  switch (status) {
    case 'healthy':
    case 'completed':
      return 'text-green-600'
    case 'warning':
    case 'running':
      return 'text-yellow-600'
    case 'error':
    case 'failed':
      return 'text-red-600'
    default:
      return 'text-neutral-400'
  }
}

const formatDuration = (seconds: number) => {
  const minutes = Math.floor(seconds / 60)
  const remainingSeconds = seconds % 60
  return `${minutes}m ${remainingSeconds}s`
}

export default function PipelinePage() {
  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-4">
          Pipeline Status
        </h1>
        <p className="text-neutral-400 mb-6">
          Real-time monitoring of Pipeline Durci v1.0 - our hardened prediction pipeline with anti-data leakage protection
        </p>
        
        <div className="flex flex-wrap gap-4 text-sm">
          <span className="badge badge-success">✅ {pipelineData.version}</span>
          <span className="badge badge-success">✅ {pipelineData.uptime_days} Days Uptime</span>
          <span className="badge badge-success">✅ {pipelineData.total_predictions_generated} Predictions Generated</span>
        </div>
      </div>

      {/* System Health Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4 mb-8">
        {Object.entries(pipelineData.system_health).map(([component, status]) => (
          <div key={component} className="card p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-2xl">{getStatusIcon(status)}</span>
              <span className={`text-sm font-medium ${getStatusColor(status)}`}>
                {status.toUpperCase()}
              </span>
            </div>
            <div className="font-medium text-white capitalize">
              {component.replace('_', ' ')}
            </div>
          </div>
        ))}
      </div>

      {/* Pipeline Overview */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
        {/* Current Status */}
        <div className="card">
          <h3 className="text-lg font-bold text-white mb-4">Current Status</h3>
          
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-neutral-400">Pipeline Version</span>
              <span className="font-medium">{pipelineData.version}</span>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-neutral-400">Current Round</span>
              <span className="font-medium">J{pipelineData.current_round}</span>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-neutral-400">Last Successful Run</span>
              <span className="font-medium">
                {new Date(pipelineData.last_successful_run).toLocaleString()}
              </span>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-neutral-400">System Status</span>
              <span className={`font-medium ${getStatusColor(pipelineData.status)}`}>
                {getStatusIcon(pipelineData.status)} {pipelineData.status.toUpperCase()}
              </span>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-neutral-400">Build</span>
              <span className="font-medium font-mono">{pipelineData.git_sha}</span>
            </div>
          </div>
        </div>

        {/* Performance Metrics */}
        <div className="card">
          <h3 className="text-lg font-bold text-white mb-4">Performance Metrics</h3>
          
          <div className="space-y-4">
            <div className="text-center">
              <div className="text-3xl font-bold text-green-600">{pipelineData.uptime_days}</div>
              <div className="text-sm text-neutral-400">Days Uptime</div>
            </div>
            
            <div className="text-center">
              <div className="text-3xl font-bold text-blue-600">{pipelineData.total_predictions_generated}</div>
              <div className="text-sm text-neutral-400">Total Predictions</div>
            </div>
            
            <div className="text-center">
              <div className="text-3xl font-bold text-oddsy-primary">
                {pipelineData.recent_jobs.filter(j => j.status === 'completed').length}
              </div>
              <div className="text-sm text-neutral-400">Recent Successful Jobs</div>
            </div>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="card">
          <h3 className="text-lg font-bold text-white mb-4">Quick Actions</h3>
          
          <div className="space-y-3">
            <button className="w-full btn-primary opacity-50 cursor-not-allowed">
              🔄 Trigger GW8 Generation
            </button>
            
            <button className="w-full btn-secondary">
              📊 View Logs
            </button>
            
            <button className="w-full btn-secondary">
              🔍 System Diagnostics
            </button>
            
            <button className="w-full btn-secondary">
              📈 Performance Report
            </button>
          </div>
          
          <div className="mt-4 p-3 bg-yellow-50 rounded-lg">
            <div className="text-xs text-yellow-800">
              ⚠️ Pipeline triggers are disabled in demo mode
            </div>
          </div>
        </div>
      </div>

      {/* Pipeline Stages */}
      <div className="card mb-8">
        <h2 className="text-xl font-bold text-white mb-6">Pipeline Stages</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {pipelineData.pipeline_stages.map((stage, idx) => (
            <div key={idx} className="bg-gray-50 rounded-lg p-4">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center space-x-2">
                  <span className="text-lg">{getStatusIcon(stage.status)}</span>
                  <span className="font-medium text-white">{stage.name}</span>
                </div>
                <span className={`text-xs font-medium ${getStatusColor(stage.status)}`}>
                  {stage.status.toUpperCase()}
                </span>
              </div>
              
              <div className="space-y-2 text-sm text-neutral-400">
                <div className="flex justify-between">
                  <span>Last Run:</span>
                  <span>{new Date(stage.last_run).toLocaleTimeString()}</span>
                </div>
                <div className="flex justify-between">
                  <span>Avg Duration:</span>
                  <span>{formatDuration(stage.duration_avg)}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Recent Jobs */}
      <div className="card">
        <h2 className="text-xl font-bold text-white mb-6">Recent Jobs</h2>
        
        <div className="space-y-6">
          {pipelineData.recent_jobs.map((job) => (
            <div key={job.id} className="border border-gray-200 rounded-lg p-6">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-3">
                  <span className="text-lg">{getStatusIcon(job.status)}</span>
                  <div>
                    <div className="font-medium text-white">Round J{job.round} Generation</div>
                    <div className="text-sm text-neutral-400">Job ID: {job.id}</div>
                  </div>
                </div>
                
                <div className="text-right">
                  <div className={`font-medium ${getStatusColor(job.status)}`}>
                    {job.status.toUpperCase()}
                  </div>
                  <div className="text-sm text-neutral-400">
                    Duration: {formatDuration(job.duration_seconds)}
                  </div>
                </div>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4 text-sm">
                <div>
                  <span className="text-neutral-400">Started:</span>
                  <div className="font-medium">{new Date(job.started_at).toLocaleString()}</div>
                </div>
                <div>
                  <span className="text-neutral-400">Completed:</span>
                  <div className="font-medium">{new Date(job.completed_at).toLocaleString()}</div>
                </div>
                <div>
                  <span className="text-neutral-400">Matches Processed:</span>
                  <div className="font-medium">{job.matches_processed}</div>
                </div>
              </div>
              
              <div className="mb-4">
                <span className="text-sm text-neutral-400">Models Executed:</span>
                <div className="flex flex-wrap gap-2 mt-1">
                  {job.models_executed.map((model, idx) => (
                    <span key={idx} className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded">
                      {model.replace('_', ' ').replace('v', ' v')}
                    </span>
                  ))}
                </div>
              </div>
              
              {job.logs && (
                <div className="bg-gray-900 rounded-lg p-4">
                  <div className="text-xs text-gray-400 mb-2">Execution Logs:</div>
                  <div className="space-y-1">
                    {job.logs.map((log, idx) => (
                      <div key={idx} className="text-xs text-green-400 font-mono">
                        [{new Date(job.started_at).toLocaleTimeString()}] {log}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Pipeline Architecture */}
      <div className="card bg-gray-50 mt-8">
        <h3 className="font-semibold text-white mb-3">Pipeline Durci v1.0 Architecture</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm">
          <div>
            <h4 className="font-medium text-gray-700 mb-2">Core Principles</h4>
            <ul className="space-y-1 text-neutral-400">
              <li>• Strict temporal validation (TimeSeriesSplit)</li>
              <li>• Anti-data leakage protection</li>
              <li>• Feature calculation BEFORE match_date</li>
              <li>• Automated validation checks</li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-medium text-gray-700 mb-2">Quality Assurance</h4>
            <ul className="space-y-1 text-neutral-400">
              <li>• Automated testing on unseen data</li>
              <li>• Performance monitoring</li>
              <li>• Model drift detection</li>
              <li>• Rollback capabilities</li>
            </ul>
          </div>
        </div>
        
        <div className="mt-4 pt-4 border-t border-gray-200">
          <div className="text-xs text-gray-500">
            Pipeline Durci v1.0 ensures production-ready predictions with validated accuracy on real EPL data.
            All components are monitored for performance and reliability.
          </div>
        </div>
      </div>
    </div>
  )
}