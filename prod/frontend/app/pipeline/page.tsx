'use client'

import { motion } from 'framer-motion'
import { CardGlass } from '@/components/ui/card-glass'
import type { Metadata } from 'next'

const pipelineSteps = [
  { step: 1, title: "Data Collection", description: "Fetch fixtures, odds, and team statistics" },
  { step: 2, title: "Feature Engineering", description: "Process raw data into model-ready features" },
  { step: 3, title: "Model Inference", description: "Generate probability predictions" },
  { step: 4, title: "Publication", description: "Serve predictions via API" },
  { step: 5, title: "Evaluation", description: "Track performance after matches" }
]

const externalAPIs = [
  { 
    name: "Football Data API", 
    purpose: "Official EPL fixtures and results",
    data: "Match schedules, final scores, team lineup information"
  },
  { 
    name: "The Odds API", 
    purpose: "Real-time betting market data",
    data: "Pre-match odds, implied probabilities, market sentiment"
  },
  { 
    name: "API-Football", 
    purpose: "Advanced team statistics",
    data: "xG data, team performance metrics, historical records"
  }
]

export default function PipelinePage() {
  return (
    <div className="min-h-screen bg-neutral-950 text-white">
      {/* Hero Section */}
      <section className="relative py-20 px-4 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-emerald-500/10 via-transparent to-emerald-400/10" />
        
        <div className="container mx-auto relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center max-w-4xl mx-auto"
          >
            <h1 className="text-4xl md:text-6xl font-bold mb-6 text-gradient-animate">
              Prediction Pipeline
            </h1>
            <p className="text-xl md:text-2xl text-neutral-300 mb-8">
              Pipeline v1.0 - Automated data processing and prediction serving
            </p>
            <div className="text-lg text-neutral-400">
              From data collection to live predictions - how we deliver reliable football insights
            </div>
          </motion.div>
        </div>
      </section>

      {/* Pipeline Overview */}
      <section className="py-16 px-4">
        <div className="container mx-auto max-w-6xl">
          
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="mb-16"
          >
            <CardGlass variant="premium" className="p-8">
              <h2 className="text-3xl font-bold mb-6 text-emerald-400">Overview</h2>
              <p className="text-lg text-neutral-300 leading-relaxed mb-6">
                Oddsy's prediction pipeline automatically collects real Premier League data, generates calibrated predictions, 
                and serves them through our API. The system follows MLOps best practices with automated data ingestion, 
                feature engineering, model inference, publication, and post-match evaluation.
              </p>
              
              <div className="grid md:grid-cols-5 gap-4 mt-8">
                {pipelineSteps.map((step, index) => (
                  <motion.div
                    key={step.step}
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6, delay: index * 0.1 }}
                    viewport={{ once: true }}
                    className="text-center"
                  >
                    <div className="w-12 h-12 bg-oddsy-primary rounded-full flex items-center justify-center mx-auto mb-3 text-black font-bold">
                      {step.step}
                    </div>
                    <h3 className="font-semibold text-white mb-2">{step.title}</h3>
                    <p className="text-sm text-neutral-400">{step.description}</p>
                  </motion.div>
                ))}
              </div>
            </CardGlass>
          </motion.div>

          {/* Data Collection */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            viewport={{ once: true }}
            className="mb-16"
          >
            <CardGlass variant="premium" className="p-8">
              <h2 className="text-3xl font-bold mb-6 text-emerald-400">Data Collection (External APIs)</h2>
              <p className="text-lg text-neutral-300 leading-relaxed mb-8">
                Our pipeline integrates with multiple external APIs to ensure we have the most comprehensive and up-to-date 
                information about Premier League matches. All external data passes through validation checks before processing.
              </p>
              
              <div className="grid gap-6">
                {externalAPIs.map((api, index) => (
                  <motion.div
                    key={api.name}
                    initial={{ opacity: 0, x: -20 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.6, delay: index * 0.1 }}
                    viewport={{ once: true }}
                    className="bg-neutral-900/50 rounded-lg p-6 border border-neutral-800"
                  >
                    <div className="flex items-start justify-between mb-4">
                      <h3 className="text-xl font-semibold text-emerald-400">{api.name}</h3>
                      <div className="text-sm text-neutral-400 bg-neutral-800 px-3 py-1 rounded">
                        {api.purpose}
                      </div>
                    </div>
                    <p className="text-neutral-300">{api.data}</p>
                  </motion.div>
                ))}
              </div>
            </CardGlass>
          </motion.div>

          {/* Feature Engineering */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            viewport={{ once: true }}
            className="mb-16"
          >
            <CardGlass variant="premium" className="p-8">
              <h2 className="text-3xl font-bold mb-6 text-emerald-400">Feature Engineering Pipeline</h2>
              <div className="grid md:grid-cols-2 gap-8">
                <div>
                  <h3 className="text-xl font-semibold mb-4 text-emerald-400">Temporal Safety</h3>
                  <p className="text-neutral-300 leading-relaxed mb-4">
                    All features are computed using only data available before match kickoff to prevent data leakage. 
                    This ensures our model's predictions reflect real-world conditions.
                  </p>
                  <ul className="space-y-2 text-neutral-400">
                    <li>• Anti-data leakage validation</li>
                    <li>• Time-aware feature computation</li>
                    <li>• Historical data boundaries</li>
                  </ul>
                </div>
                <div>
                  <h3 className="text-xl font-semibold mb-4 text-emerald-400">Real-time Processing</h3>
                  <p className="text-neutral-300 leading-relaxed mb-4">
                    ELO ratings update after each completed match, market probabilities are corrected for overround, 
                    and contextual features are computed dynamically.
                  </p>
                  <ul className="space-y-2 text-neutral-400">
                    <li>• Dynamic ELO updates</li>
                    <li>• Market entropy calculation</li>
                    <li>• Form metrics aggregation</li>
                  </ul>
                </div>
              </div>
              
              <div className="mt-8 p-6 bg-neutral-900/30 rounded-lg border border-neutral-800">
                <h3 className="text-lg font-semibold mb-3 text-emerald-400">Serving Consistency</h3>
                <p className="text-neutral-300">
                  The serving code reproduces the same transformations as the training code (same scaling/aggregation windows) 
                  to avoid training/serving skew. This ensures predictions remain reliable in production.
                </p>
              </div>
            </CardGlass>
          </motion.div>

          {/* Prediction & Publication */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.6 }}
            viewport={{ once: true }}
            className="mb-16"
          >
            <CardGlass variant="premium" className="p-8">
              <h2 className="text-3xl font-bold mb-6 text-emerald-400">Prediction & Publication</h2>
              <div className="grid md:grid-cols-3 gap-8">
                <div>
                  <h3 className="text-xl font-semibold mb-4 text-emerald-400">Model Inference</h3>
                  <p className="text-neutral-300 leading-relaxed">
                    Enhanced Baseline v3.0 Random Forest generates probability distributions (p_H, p_D, p_A) for upcoming fixtures. 
                    Each prediction includes confidence intervals and calibration metrics.
                  </p>
                </div>
                <div>
                  <h3 className="text-xl font-semibold mb-4 text-emerald-400">API Serving</h3>
                  <p className="text-neutral-300 leading-relaxed">
                    Predictions are exposed via FastAPI at descriptive endpoints like <code className="bg-neutral-800 px-2 py-1 rounded text-emerald-400">/api/gameweeks/{'{round}'}/predictions</code>. 
                    Legacy endpoints ensure backward compatibility.
                  </p>
                </div>
                <div>
                  <h3 className="text-xl font-semibold mb-4 text-emerald-400">Frontend Integration</h3>
                  <p className="text-neutral-300 leading-relaxed">
                    Next.js App Router with incremental revalidation ensures fresh data delivery. 
                    Static fallbacks guarantee site availability even during backend maintenance.
                  </p>
                </div>
              </div>
            </CardGlass>
          </motion.div>

          {/* Architecture & Reliability */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.8 }}
            viewport={{ once: true }}
            className="mb-16"
          >
            <CardGlass variant="premium" className="p-8">
              <h2 className="text-3xl font-bold mb-6 text-emerald-400">Architecture & Reliability</h2>
              <div className="grid md:grid-cols-2 gap-8">
                <div>
                  <h3 className="text-xl font-semibold mb-4 text-emerald-400">Backend Infrastructure</h3>
                  <ul className="space-y-3 text-neutral-300">
                    <li className="flex items-start">
                      <span className="text-emerald-400 mr-2">•</span>
                      FastAPI with PostgreSQL database and structured logging
                    </li>
                    <li className="flex items-start">
                      <span className="text-emerald-400 mr-2">•</span>
                      Health monitoring at <code className="bg-neutral-800 px-2 py-1 rounded">/api/system/health</code>
                    </li>
                    <li className="flex items-start">
                      <span className="text-emerald-400 mr-2">•</span>
                      Rate limiting and proper CORS configuration
                    </li>
                    <li className="flex items-start">
                      <span className="text-emerald-400 mr-2">•</span>
                      Redis caching with different TTL for current vs historical data
                    </li>
                  </ul>
                </div>
                <div>
                  <h3 className="text-xl font-semibold mb-4 text-emerald-400">Reliability Features</h3>
                  <ul className="space-y-3 text-neutral-300">
                    <li className="flex items-start">
                      <span className="text-emerald-400 mr-2">•</span>
                      Health checks validate database, filesystem, and external API connectivity
                    </li>
                    <li className="flex items-start">
                      <span className="text-emerald-400 mr-2">•</span>
                      Fallback to static prediction files during backend unavailability
                    </li>
                    <li className="flex items-start">
                      <span className="text-emerald-400 mr-2">•</span>
                      Graceful error handling with proper HTTP status codes
                    </li>
                    <li className="flex items-start">
                      <span className="text-emerald-400 mr-2">•</span>
                      Automated retry logic for external API failures
                    </li>
                  </ul>
                </div>
              </div>
            </CardGlass>
          </motion.div>

          {/* Post-Match Evaluation */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 1.0 }}
            viewport={{ once: true }}
          >
            <CardGlass variant="premium" className="p-8">
              <h2 className="text-3xl font-bold mb-6 text-emerald-400">Post-Match Evaluation</h2>
              <div className="grid md:grid-cols-3 gap-8">
                <div>
                  <h3 className="text-xl font-semibold mb-4 text-emerald-400">Results Integration</h3>
                  <p className="text-neutral-300 leading-relaxed">
                    After each gameweek, match results are automatically ingested from Football Data API 
                    and matched to our predictions for comprehensive performance tracking.
                  </p>
                </div>
                <div>
                  <h3 className="text-xl font-semibold mb-4 text-emerald-400">Calibration Monitoring</h3>
                  <p className="text-neutral-300 leading-relaxed">
                    Brier score computation and reliability diagrams are updated weekly to ensure 
                    our probability predictions remain well-calibrated throughout the season.
                  </p>
                </div>
                <div>
                  <h3 className="text-xl font-semibold mb-4 text-emerald-400">Performance Dashboards</h3>
                  <p className="text-neutral-300 leading-relaxed">
                    Real-time accuracy metrics and prediction quality analysis help us continuously 
                    monitor and improve our model's performance.
                  </p>
                </div>
              </div>
              
              <div className="mt-8 p-6 bg-gradient-to-r from-emerald-500/10 to-emerald-400/10 rounded-lg border border-emerald-400/20">
                <h3 className="text-lg font-semibold mb-3 text-emerald-400">Current Status: Pipeline v1.0</h3>
                <div className="grid md:grid-cols-3 gap-6 text-sm">
                  <div>
                    <span className="text-neutral-400">Status:</span>
                    <div className="text-green-400 font-semibold">Production Ready ✓</div>
                  </div>
                  <div>
                    <span className="text-neutral-400">API Endpoints:</span>
                    <div className="text-white">System & Gameweeks</div>
                  </div>
                  <div>
                    <span className="text-neutral-400">External Dependencies:</span>
                    <div className="text-white">3 APIs with fallbacks</div>
                  </div>
                </div>
              </div>
            </CardGlass>
          </motion.div>

        </div>
      </section>
    </div>
  )
}