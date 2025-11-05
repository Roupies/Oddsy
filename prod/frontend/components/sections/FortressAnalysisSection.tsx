'use client'

import React from 'react'
import { motion } from 'framer-motion'
import { StadiumStory } from './StadiumStory'

const stadiumsData = [
  {
    stadium: "Anfield",
    team: "Liverpool FC",
    narrative: "Liverpool's fortress. The Kop roars, opponents crumble. Our models predict continued dominance at home with crowd psychology as a decisive factor.",
    imageUrl: "/images/Stades/Liverpool.webp",
    imagePosition: "left" as const
  },
  {
    stadium: "Etihad Stadium",
    team: "Manchester City",
    narrative: "City's machine. Clinical, relentless, predictable. Home advantage maximized under Guardiola's tactical precision and squad depth.",
    imageUrl: "/images/Stades/Manchester_City.jpg",
    imagePosition: "left" as const
  },
  {
    stadium: "Stamford Bridge",
    team: "Chelsea",
    narrative: "Chelsea's resilience. Compact, intense, unforgiving. The Bridge remains a test for any visiting side despite recent transitions.",
    imageUrl: "/images/Stades/Chelsea.webp",
    imagePosition: "left" as const
  }
]

export const FortressAnalysisSection: React.FC = () => {
  return (
    <section id="fortress" className="bg-neutral-950 text-white py-12 md:py-16 px-4 md:px-8 lg:px-16">
      <div className="max-w-7xl mx-auto">
        {/* Section header */}
        <motion.div 
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="mb-12 md:mb-16"
        >
          <div className="flex items-center gap-4 mb-6">
            <div className="w-2 h-16 bg-emerald-500" />
            <span className="text-neutral-500 text-sm font-mono uppercase tracking-wider">
              Machine Learning Analysis
            </span>
          </div>
          
          <h2 className="text-[clamp(3rem,6vw,5rem)] font-black tracking-tighter leading-[0.9] mb-4">
            FORTRESS
            <br />
            ANALYSIS
          </h2>
          <div className="w-20 h-1 bg-emerald-500 mb-6" />
          
          <p className="text-neutral-400 text-lg md:text-xl leading-relaxed max-w-2xl">
            Home advantage quantified through data science. Each stadium tells a story of tactical patterns, 
            crowd psychology, and predictive insights from our Enhanced Baseline v3.0 model.
          </p>
        </motion.div>

        {/* Stadium stories */}
        <div className="space-y-12 md:space-y-16">
          {stadiumsData.map((stadium, index) => (
            <StadiumStory
              key={stadium.stadium}
              {...stadium}
              delay={index * 0.2}
            />
          ))}
        </div>

        {/* Bottom insight */}
        <motion.div 
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="mt-12 pt-12 border-t border-neutral-800"
        >
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="text-center md:text-left">
              <div className="text-3xl font-black text-emerald-500 mb-2">67%</div>
              <div className="text-neutral-400 text-sm uppercase tracking-wider">
                Average home win rate across top 6
              </div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-black text-emerald-500 mb-2">2.2</div>
              <div className="text-neutral-400 text-sm uppercase tracking-wider">
                Average goals per home game
              </div>
            </div>
            <div className="text-center md:text-right">
              <div className="text-3xl font-black text-emerald-500 mb-2">53k</div>
              <div className="text-neutral-400 text-sm uppercase tracking-wider">
                Average stadium capacity
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  )
}