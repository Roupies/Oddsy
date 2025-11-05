'use client'

import { motion } from 'framer-motion'
import { CardGlass } from '@/components/ui/card-glass'
import { APP_CONFIG } from '@/config/appConstants'
import type { Metadata } from 'next'

const benchmarkComparison = [
  { name: 'Random Baseline', accuracy: 33.3, description: 'Pure random predictions' },
  { name: 'Majority Class', accuracy: 43.6, description: 'Always predict most frequent outcome' },
  { name: 'Bookmaker Favorites', accuracy: 48.2, description: 'Follow betting favorites' },
  { name: 'Enhanced Baseline v3.0', accuracy: 51.3, description: 'Our Random Forest production model', highlight: true }
]

export default function ModelsPage() {
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
              Our Prediction Model
            </h1>
            <p className="text-xl md:text-2xl text-neutral-300 mb-8">
              Enhanced Baseline v3.0 - Random Forest approach for Premier League outcome prediction
            </p>
            <div className="text-lg text-neutral-400">
              Delivering <span className="text-emerald-400 font-semibold">{APP_CONFIG.MODEL_ACCURACY}</span> accuracy through ensemble learning
            </div>
          </motion.div>
        </div>
      </section>

      {/* Main Content */}
      <section className="py-16 px-4">
        <div className="container mx-auto max-w-6xl">
          
          {/* What it is */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="mb-16"
          >
            <CardGlass variant="premium" className="p-8">
              <h2 className="text-3xl font-bold mb-6 text-emerald-400">What it is</h2>
              <p className="text-lg text-neutral-300 leading-relaxed mb-4">
                Enhanced Baseline v3.0 is Oddsy's production probability model that predicts Premier League match outcomes: Home, Draw, Away. 
                It combines real team performance metrics, market intelligence, and contextual factors to output calibrated probabilities for each fixture.
              </p>
              <p className="text-lg text-neutral-300 leading-relaxed">
                Rather than making binary predictions, our model outputs probability distributions (p_H, p_D, p_A) that can be evaluated for calibration quality and provide confidence levels for each prediction.
              </p>
            </CardGlass>
          </motion.div>

          {/* How it works */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            viewport={{ once: true }}
            className="mb-16"
          >
            <CardGlass variant="premium" className="p-8">
              <h2 className="text-3xl font-bold mb-6 text-emerald-400">How it works</h2>
              <div className="space-y-4">
                <p className="text-lg text-neutral-300 leading-relaxed">
                  <strong>The core algorithm is a Random Forest:</strong> Think of it as a panel of football experts, where each expert (decision tree) 
                  focuses on different aspects - one looks at recent form, another at historical head-to-head, another at market sentiment. 
                  The Random Forest combines all these "expert opinions" into a single, more reliable prediction.
                </p>
                <div className="grid md:grid-cols-3 gap-6 mt-8">
                  <div className="bg-neutral-900/50 rounded-lg p-6 border border-neutral-800">
                    <h3 className="text-xl font-semibold mb-3 text-emerald-400">Ensemble Learning</h3>
                    <p className="text-neutral-400">100+ decision trees vote together, reducing overfitting on football's inherent randomness</p>
                  </div>
                  <div className="bg-neutral-900/50 rounded-lg p-6 border border-neutral-800">
                    <h3 className="text-xl font-semibold mb-3 text-emerald-400">Probability Output</h3>
                    <p className="text-neutral-400">Outputs calibrated probability distributions suitable for Brier score evaluation</p>
                  </div>
                  <div className="bg-neutral-900/50 rounded-lg p-6 border border-neutral-800">
                    <h3 className="text-xl font-semibold mb-3 text-emerald-400">Feature Interactions</h3>
                    <p className="text-neutral-400">Automatically captures complex relationships between team strength and context</p>
                  </div>
                </div>
              </div>
            </CardGlass>
          </motion.div>

          {/* Features Used */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            viewport={{ once: true }}
            className="mb-16"
          >
            <CardGlass variant="premium" className="p-8">
              <h2 className="text-3xl font-bold mb-6 text-emerald-400">Features Used (Oddsy Specific)</h2>
              <div className="grid md:grid-cols-2 gap-8">
                <div>
                  <h3 className="text-xl font-semibold mb-4 text-emerald-400">Team Performance</h3>
                  <ul className="space-y-2 text-neutral-300">
                    <li>• Normalized ELO ratings</li>
                    <li>• Rolling xG efficiency over last 6 matches</li>
                    <li>• Recent form metrics</li>
                    <li>• Historical head-to-head records</li>
                  </ul>
                </div>
                <div>
                  <h3 className="text-xl font-semibold mb-4 text-emerald-400">Market Intelligence</h3>
                  <ul className="space-y-2 text-neutral-300">
                    <li>• Corrected probabilities from The Odds API</li>
                    <li>• Overround removal and entropy calculations</li>
                    <li>• Market sentiment indicators</li>
                    <li>• Betting pattern analysis</li>
                  </ul>
                </div>
                <div>
                  <h3 className="text-xl font-semibold mb-4 text-emerald-400">Contextual Factors</h3>
                  <ul className="space-y-2 text-neutral-300">
                    <li>• Home advantage coefficients</li>
                    <li>• Fixture congestion analysis</li>
                    <li>• Matchday timing effects</li>
                    <li>• Schedule density impacts</li>
                  </ul>
                </div>
                <div>
                  <h3 className="text-xl font-semibold mb-4 text-emerald-400">Pipeline Integration</h3>
                  <ul className="space-y-2 text-neutral-300">
                    <li>• Strict temporal validation</li>
                    <li>• Anti-data leakage safeguards</li>
                    <li>• Real-time feature computation</li>
                    <li>• Consistent serving transformations</li>
                  </ul>
                </div>
              </div>
            </CardGlass>
          </motion.div>

          {/* Performance Comparison */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.6 }}
            viewport={{ once: true }}
            className="mb-16"
          >
            <CardGlass variant="premium" className="p-8">
              <h2 className="text-3xl font-bold mb-6 text-emerald-400">Current Performance</h2>
              <div className="grid gap-4 mb-8">
                {benchmarkComparison.map((benchmark, index) => (
                  <div
                    key={benchmark.name}
                    className={`p-4 rounded-lg border transition-all duration-300 ${
                      benchmark.highlight 
                        ? 'bg-emerald-500/10 border-emerald-400' 
                        : 'bg-neutral-900/50 border-neutral-800'
                    }`}
                  >
                    <div className="flex justify-between items-center">
                      <div>
                        <h3 className={`font-semibold ${benchmark.highlight ? 'text-emerald-400' : 'text-white'}`}>
                          {benchmark.name}
                        </h3>
                        <p className="text-neutral-400 text-sm">{benchmark.description}</p>
                      </div>
                      <div className={`text-2xl font-bold ${benchmark.highlight ? 'text-emerald-400' : 'text-neutral-300'}`}>
                        {benchmark.accuracy}%
                      </div>
                    </div>
                  </div>
                ))}
              </div>
              
              <div className="grid md:grid-cols-2 gap-8">
                <div>
                  <h3 className="text-xl font-semibold mb-4 text-emerald-400">Validation Methodology</h3>
                  <p className="text-neutral-300 leading-relaxed">
                    Cross-validation on historical Premier League seasons using TimeSeriesSplit to prevent data leakage. 
                    Probability quality tracked with Brier score and reliability diagrams.
                  </p>
                </div>
                <div>
                  <h3 className="text-xl font-semibold mb-4 text-emerald-400">Calibration Quality</h3>
                  <p className="text-neutral-300 leading-relaxed">
                    We track prediction quality with Brier score - when we predict 60% home win probability, 
                    about 6 out of 10 similar situations should indeed be home wins. This keeps our probabilities honest.
                  </p>
                </div>
              </div>
            </CardGlass>
          </motion.div>

          {/* Roadmap */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.8 }}
            viewport={{ once: true }}
          >
            <CardGlass variant="premium" className="p-8">
              <h2 className="text-3xl font-bold mb-6 text-emerald-400">Limitations & Roadmap</h2>
              <div className="grid md:grid-cols-2 gap-8">
                <div>
                  <h3 className="text-xl font-semibold mb-4 text-emerald-400">Current Limitations</h3>
                  <ul className="space-y-2 text-neutral-300">
                    <li>• No real-time injury information</li>
                    <li>• Limited lineup prediction capabilities</li>
                    <li>• Weather conditions not yet integrated</li>
                    <li>• Player-level performance metrics pending</li>
                  </ul>
                </div>
                <div>
                  <h3 className="text-xl font-semibold mb-4 text-emerald-400">Future Enhancements</h3>
                  <ul className="space-y-2 text-neutral-300">
                    <li>• Pre-match team news integration</li>
                    <li>• Advanced player performance models</li>
                    <li>• Real-time market reaction analysis</li>
                    <li>• Enhanced feature engineering pipeline</li>
                  </ul>
                </div>
              </div>
              
              <div className="mt-8 p-6 bg-neutral-900/30 rounded-lg border border-neutral-800">
                <p className="text-neutral-300 leading-relaxed">
                  <strong>Production Status:</strong> Enhanced Baseline v3.0 is currently serving live predictions 
                  via Pipeline v1.0, with continuous monitoring of accuracy and calibration metrics.
                </p>
              </div>
            </CardGlass>
          </motion.div>

        </div>
      </section>
    </div>
  )
}