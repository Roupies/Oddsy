import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import { Providers } from '../lib/providers'
import { Header } from '../components/layout/header'
import { Footer } from '../components/layout/footer'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Oddsy - AI Premier League Predictions',
  description: 'AI-powered Premier League match predictions with Enhanced Baseline v2.4',
  keywords: 'Premier League, predictions, AI, football, machine learning, EPL',
  authors: [{ name: 'Oddsy Team' }],
  openGraph: {
    title: 'Oddsy - AI Premier League Predictions',
    description: 'Validated AI predictions for Premier League matches',
    type: 'website',
    siteName: 'Oddsy',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Oddsy - AI Premier League Predictions',
    description: 'Validated AI predictions for Premier League matches',
  },
  robots: {
    index: true,
    follow: true,
  },
  verification: {
    // Add Google Search Console verification if needed
  }
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="scroll-smooth">
      <body className={`${inter.className} bg-neutral-950`}>
        <Providers>
          <div className="min-h-screen flex flex-col bg-neutral-950">
            <Header />
            {/* Add top padding to account for fixed header */}
            <main className="flex-1 pt-20 bg-neutral-950">
              {children}
            </main>
            <Footer />
          </div>
        </Providers>
      </body>
    </html>
  )
}