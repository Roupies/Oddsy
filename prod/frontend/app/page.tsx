'use client'

import { HeroSplitPremium } from '@/components/hero/HeroSplitPremium'
import { FortressAnalysisSection } from '@/components/sections/FortressAnalysisSection'
import { NextGameweekGrid } from '@/components/sections/NextGameweekGrid'
import { CardGlass } from '@/components/ui/card-glass'
import { ButtonPremium } from '@/components/ui/button-premium'
import { useCurrentGameweek } from '@/hooks/useCurrentGameweek'
import { APP_CONFIG } from '@/config/appConstants'
import Link from 'next/link'

export default function HomePage() {
  const { data: gameweekData } = useCurrentGameweek()
  
  return (
    <div className="min-h-screen -mt-20 relative">
      {/* Editorial Hero Section - Full viewport */}
      <HeroSplitPremium />
      
      {/* Stadium Stories Section - Reduced gap */}
      <div className="-mt-8">
        <FortressAnalysisSection />
      </div>
      
      {/* Next Gameweek Predictions - Reduced gap */}
      <div className="-mt-8">
        <NextGameweekGrid />
      </div>
      
      {/* Legacy Content Container */}
      <div className="container mx-auto px-4 py-16 space-y-16 relative">
        {/* Background Gradient */}
        <div className="absolute inset-0 bg-gradient-to-b from-oddsy-primary/5 via-transparent to-oddsy-secondary/5 pointer-events-none" />

        {/* Key Metrics - Premium Style */}
        <section id="model" className="relative z-10 -mt-32">
          <div className="text-center mb-12">
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
              Proven Performance
            </h2>
            <p className="text-xl text-neutral-400 max-w-2xl mx-auto">
              Our AI models deliver consistent results validated on real Premier League data
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <CardGlass variant="premium" hover="premium" className="text-center group">
              <div className="text-4xl md:text-5xl font-bold text-gradient-animate mb-3 group-hover:scale-110 transition-transform duration-300">
                {APP_CONFIG.MODEL_ACCURACY}
              </div>
              <div className="text-lg font-semibold text-white mb-2">{APP_CONFIG.MODEL_VERSION}</div>
              <div className="text-sm text-neutral-400">Cross-validation accuracy</div>
              <div className="mt-4 inline-flex items-center space-x-1 text-oddsy-secondary text-sm font-medium">
                <span>🏆</span>
                <span>Champion Model</span>
              </div>
            </CardGlass>
            
            <CardGlass variant="premium" hover="premium" className="text-center group">
              <div className="text-4xl md:text-5xl font-bold text-gradient-animate mb-3 group-hover:scale-110 transition-transform duration-300">
                {APP_CONFIG.MODEL_ACCURACY}
              </div>
              <div className="text-lg font-semibold text-white mb-2">Real EPL Accuracy</div>
              <div className="text-sm text-neutral-400">{APP_CONFIG.DATASET_SIZE}</div>
              <div className="mt-4 inline-flex items-center space-x-1 text-oddsy-accent text-sm font-medium">
                <span>⚽</span>
                <span>Real Data Tested</span>
              </div>
            </CardGlass>
            
            <CardGlass variant="premium" hover="premium" className="text-center group">
              <div className="text-4xl md:text-5xl font-bold text-gradient-animate mb-3 group-hover:scale-110 transition-transform duration-300">
                {gameweekData?.totalPredictions || 110}+
              </div>
              <div className="text-lg font-semibold text-white mb-2">Predictions Generated</div>
              <div className="text-sm text-neutral-400">{APP_CONFIG.PIPELINE_VERSION}</div>
              <div className="mt-4 inline-flex items-center space-x-1 text-oddsy-primary text-sm font-medium">
                <span>🚀</span>
                <span>Production Ready</span>
              </div>
            </CardGlass>
          </div>
        </section>

        {/* Features - Premium Grid */}
        <section className="relative z-10">
          <div className="text-center mb-12">
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
              Next-Generation AI Technology
            </h2>
            <p className="text-xl text-neutral-400 max-w-3xl mx-auto">
              Built with cutting-edge machine learning and rigorous validation processes
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            <CardGlass variant="epl" hover="glow" className="group">
              <div className="text-4xl mb-4 group-hover:animate-bounce-gentle">🏆</div>
              <h3 className="text-xl font-bold text-white mb-3">{APP_CONFIG.MODEL_VERSION}</h3>
              <p className="text-neutral-400 mb-4">
                Advanced feature engineering with calibration and ensemble methods for optimal accuracy
              </p>
              <div className="inline-flex items-center text-oddsy-secondary text-sm font-medium">
                <span className="w-2 h-2 bg-oddsy-secondary rounded-full mr-2 animate-pulse"></span>
                Champion Model
              </div>
            </CardGlass>
            
            <CardGlass variant="epl" hover="glow" className="group">
              <div className="text-4xl mb-4 group-hover:animate-bounce-gentle">⚡</div>
              <h3 className="text-xl font-bold text-white mb-3">{APP_CONFIG.PIPELINE_VERSION}</h3>
              <p className="text-neutral-400 mb-4">
                Hardened production pipeline with strict temporal validation and anti-leakage protection
              </p>
              <div className="inline-flex items-center text-oddsy-accent text-sm font-medium">
                <span className="w-2 h-2 bg-oddsy-accent rounded-full mr-2 animate-pulse"></span>
                Production Ready
              </div>
            </CardGlass>
            
            <CardGlass variant="epl" hover="glow" className="group">
              <div className="text-4xl mb-4 group-hover:animate-bounce-gentle">📊</div>
              <h3 className="text-xl font-bold text-white mb-3">Real-time Predictions</h3>
              <p className="text-neutral-400 mb-4">
                Live predictions with confidence intervals and comprehensive model analysis
              </p>
              <div className="inline-flex items-center text-oddsy-primary text-sm font-medium">
                <span className="w-2 h-2 bg-oddsy-primary rounded-full mr-2 animate-pulse"></span>
                Live Updates
              </div>
            </CardGlass>
            
            <CardGlass variant="epl" hover="glow" className="group">
              <div className="text-4xl mb-4 group-hover:animate-bounce-gentle">🎯</div>
              <h3 className="text-xl font-bold text-white mb-3">Validated Performance</h3>
              <p className="text-neutral-400 mb-4">
                Rigorously tested on {gameweekData?.validatedMatches || 100} real EPL matches with transparent accuracy reporting
              </p>
              <div className="inline-flex items-center text-oddsy-secondary text-sm font-medium">
                <span className="w-2 h-2 bg-oddsy-secondary rounded-full mr-2 animate-pulse"></span>
                Proven Results
              </div>
            </CardGlass>
            
            <CardGlass variant="epl" hover="glow" className="group">
              <div className="text-4xl mb-4 group-hover:animate-bounce-gentle">🔒</div>
              <h3 className="text-xl font-bold text-white mb-3">Anti-Data Leakage</h3>
              <p className="text-neutral-400 mb-4">
                Strict temporal validation ensures no future information contamination
              </p>
              <div className="inline-flex items-center text-oddsy-accent text-sm font-medium">
                <span className="w-2 h-2 bg-oddsy-accent rounded-full mr-2 animate-pulse"></span>
                Secure & Valid
              </div>
            </CardGlass>
            
            <CardGlass variant="epl" hover="glow" className="group">
              <div className="text-4xl mb-4 group-hover:animate-bounce-gentle">📈</div>
              <h3 className="text-xl font-bold text-white mb-3">Performance Monitoring</h3>
              <p className="text-neutral-400 mb-4">
                Continuous tracking with model comparison and disagreement analysis
              </p>
              <div className="inline-flex items-center text-oddsy-primary text-sm font-medium">
                <span className="w-2 h-2 bg-oddsy-primary rounded-full mr-2 animate-pulse"></span>
                Always Improving
              </div>
            </CardGlass>
          </div>
        </section>

        {/* Premium Call to Action */}
        <section className="relative z-10 overflow-hidden">
          <div className="epl-gradient rounded-3xl p-12 md:p-16 text-center text-white relative">
            {/* Animated Background Elements */}
            <div className="absolute inset-0 opacity-20">
              <div className="absolute top-10 left-10 w-4 h-4 bg-white rounded-full animate-float"></div>
              <div className="absolute top-20 right-16 w-2 h-2 bg-oddsy-secondary rounded-full animate-bounce delay-300"></div>
              <div className="absolute bottom-16 left-20 w-3 h-3 bg-oddsy-accent rounded-full animate-pulse delay-700"></div>
              <div className="absolute bottom-10 right-10 w-2 h-2 bg-white rounded-full animate-float delay-500"></div>
            </div>
            
            <div className="relative z-10">
              <h2 className="text-4xl md:text-5xl font-bold mb-6 leading-tight">
                Experience the Future of
                <span className="block text-oddsy-secondary">
                  Football Predictions
                </span>
              </h2>
              
              <p className="text-xl md:text-2xl opacity-90 mb-8 max-w-2xl mx-auto font-light">
                Join thousands using our AI-powered predictions validated on real Premier League data
              </p>
              
              {/* Trust Indicators */}
              <div className="flex flex-wrap justify-center gap-6 mb-10">
                <div className="inline-flex items-center space-x-2 bg-white/10 backdrop-blur-sm rounded-lg px-4 py-2">
                  <span className="text-oddsy-secondary">✨</span>
                  <span className="text-sm font-medium">{APP_CONFIG.MODEL_ACCURACY} Accuracy</span>
                </div>
                <div className="inline-flex items-center space-x-2 bg-white/10 backdrop-blur-sm rounded-lg px-4 py-2">
                  <span className="text-oddsy-secondary">🚀</span>
                  <span className="text-sm font-medium">{gameweekData?.totalPredictions || 110}+ Predictions</span>
                </div>
                <div className="inline-flex items-center space-x-2 bg-white/10 backdrop-blur-sm rounded-lg px-4 py-2">
                  <span className="text-oddsy-secondary">⚽</span>
                  <span className="text-sm font-medium">Real EPL Data</span>
                </div>
              </div>
              
              {/* CTA Buttons */}
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <Link href="/predictions/latest">
                  <ButtonPremium 
                    size="lg" 
                    className="bg-white text-oddsy-primary hover:bg-gray-100 hover:scale-105 shadow-2xl"
                  >
                    <span className="flex items-center space-x-2">
                      <span>View Latest Predictions</span>
                      <span className="text-xl">⚽</span>
                    </span>
                  </ButtonPremium>
                </Link>
                
                <Link href="/pipeline">
                  <ButtonPremium 
                    variant="ghost" 
                    size="lg"
                    className="border-2 border-white/30 hover:border-white hover:bg-white/10"
                  >
                    <span className="flex items-center space-x-2">
                      <span>Pipeline Status</span>
                      <span className="text-xl">📊</span>
                    </span>
                  </ButtonPremium>
                </Link>
              </div>
              
              {/* Bottom Notice */}
              <div className="mt-8 pt-8 border-t border-white/20">
                <p className="text-sm opacity-75">
                  Free to use • Updated in real-time • Validated on {gameweekData?.validatedMatches || 100} real EPL matches
                </p>
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  )
}