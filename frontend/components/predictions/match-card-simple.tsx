import { ClubLogo } from '@/components/ui/club-logo'
import { StadiumBackground } from '@/components/ui/stadium-background'
import { getConfidenceColors, getProgressBarColor, formatConfidence } from '@/lib/confidence-colors'

// Mapping pour vérifier la présence d'un stade
const STADIUM_MAP: Record<string, boolean> = {
  'Liverpool': true,
  'Arsenal': true, 
  'Burnley': true,
  'Chelsea': true,
  'Manchester City': true,
  'Manchester United': true
}

// Fonction pour raccourcir les noms d'équipes
const shortenTeamName = (teamName: string): string => {
  const shortNames: Record<string, string> = {
    'Brighton and Hove Albion': 'Brighton',
    'Wolverhampton Wanderers': 'Wolves',
    'West Ham United': 'West Ham',
    'Manchester United': 'Man United',
    'Manchester City': 'Man City',
    'Tottenham Hotspur': 'Tottenham',
    'Newcastle United': 'Newcastle',
    'Nottingham Forest': 'Forest'
  }
  return shortNames[teamName] || teamName
}

// Fonction pour tronquer intelligemment les noms longs
const truncateTeamName = (teamName: string, maxLength: number = 12): string => {
  const shortName = shortenTeamName(teamName)
  if (shortName.length <= maxLength) return shortName
  return shortName.substring(0, maxLength - 1) + '…'
}

export function MatchCardSimple({ match }: { match: any }) {
  const getPredictionLabel = (prediction: string) => {
    const homeTeam = shortenTeamName(match.home_team)
    const awayTeam = shortenTeamName(match.away_team)
    
    switch (prediction) {
      case 'H': return `🏠 ${homeTeam} Win`
      case 'D': return '🤝 Draw'
      case 'A': return `✈️ ${awayTeam} Win`
      default: return '🤔 Uncertain'
    }
  }

  const getPredictionColor = (prediction: string) => {
    switch (prediction) {
      case 'H': return { badge: 'bg-green-100', text: 'text-green-700' }
      case 'D': return { badge: 'bg-yellow-100', text: 'text-yellow-700' }
      case 'A': return { badge: 'bg-blue-100', text: 'text-blue-700' }
      default: return { badge: 'bg-gray-100', text: 'text-gray-700' }
    }
  }

  const confidence = match.ensemble.confidence
  const confidenceColors = getConfidenceColors(confidence)
  const confidencePercentage = confidence * 100
  const predictionColors = getPredictionColor(match.ensemble.prediction)
  
  // Détection présence stade pour contraste dynamique
  const hasStadium = STADIUM_MAP[match.home_team] || false
  const textColorClass = hasStadium ? 'text-white drop-shadow-lg' : 'text-gray-900'
  const textColorSecondary = hasStadium ? 'text-white/80' : 'text-gray-600'
  
  return (
    <div className="
      relative overflow-hidden rounded-xl 
      transform transition-all duration-300 hover:scale-[1.01] hover:shadow-2xl 
      cursor-pointer group
      w-full max-w-lg md:max-w-xl lg:max-w-2xl h-80 md:h-96 lg:h-[420px]
      border border-gray-200 shadow-lg mx-auto
    ">
      {/* Stadium Background */}
      <StadiumBackground 
        homeTeam={match.home_team} 
        overlayOpacity={0.08}
      />
      
      {/* Content Overlay */}
      <div className="relative z-10 h-full flex flex-col">

        {/* Header avec équipes et logos - Plus d'espace */}
        <div className="p-6 md:p-8 pb-4 flex-1">
          <div className="flex flex-col sm:flex-row items-center justify-between mb-4 md:mb-6 gap-4 sm:gap-0">
            {/* Home team */}
            <div className="flex items-center space-x-3 md:space-x-4 flex-1 min-w-0">
              <ClubLogo clubName={match.home_team} size="lg" />
              <div className="text-left min-w-0">
                <h3 className={`text-base md:text-lg lg:text-xl font-bold ${textColorClass} leading-tight truncate`}>
                  {truncateTeamName(match.home_team, 15)}
                </h3>
                <span className={`text-xs md:text-sm ${textColorSecondary} font-medium`}>HOME</span>
              </div>
            </div>
            
            {/* VS + Match info */}
            <div className="flex-shrink-0 text-center mx-4 md:mx-8">
              <span className={`text-xl md:text-2xl font-bold ${textColorClass}`}>VS</span>
              <div className={`text-xs md:text-sm ${textColorSecondary} mt-1 md:mt-2 font-medium`}>
                GW{match.round} • {match.date}
              </div>
            </div>
            
            {/* Away team */}
            <div className="flex items-center space-x-3 md:space-x-4 flex-1 justify-end min-w-0">
              <div className="text-right min-w-0">
                <h3 className={`text-base md:text-lg lg:text-xl font-bold ${textColorClass} leading-tight truncate`}>
                  {truncateTeamName(match.away_team, 15)}
                </h3>
                <span className={`text-xs md:text-sm ${textColorSecondary} font-medium`}>AWAY</span>
              </div>
              <ClubLogo clubName={match.away_team} size="lg" />
            </div>
          </div>

          {/* Prédiction principale - Plus visible */}
          <div className="text-center mb-4 md:mb-6">
            <div className={`${predictionColors.badge} ${predictionColors.text} inline-flex items-center px-4 md:px-6 py-2 md:py-3 rounded-full font-bold text-sm md:text-base shadow-sm`}>
              {getPredictionLabel(match.ensemble.prediction)}
            </div>
          </div>
        </div>

        {/* Content area compacte en bas (25% height) */}
        <div className="mx-4 md:mx-6 lg:mx-8 mb-4 md:mb-6 bg-white/96 backdrop-blur-sm rounded-lg p-4 md:p-6 border border-white/30 shadow-lg">
          {/* Barre de confiance */}
          <div className="mb-3 md:mb-4">
            <div className="flex justify-between items-center mb-2 md:mb-3">
              <span className="text-sm md:text-base font-semibold text-gray-700">
                {confidenceColors.label}
              </span>
              <span className={`text-sm md:text-base font-bold ${confidenceColors.text}`}>
                {formatConfidence(confidence)}
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-3 md:h-4">
              <div 
                className={`h-3 md:h-4 rounded-full transition-all duration-1000 ease-out ${getProgressBarColor(confidence)} shadow-sm`}
                style={{ 
                  width: `${Math.max(confidencePercentage, 2)}%`,
                  minWidth: '8px'
                }}
              />
            </div>
          </div>

          {/* Probabilités détaillées */}
          <div className="grid grid-cols-3 gap-2 md:gap-4">
            <div className="text-center">
              <div className="font-semibold text-gray-700 mb-1 md:mb-2 text-xs md:text-sm">Home</div>
              <div className="font-bold text-green-600 text-base md:text-lg">
                {(match.ensemble.probabilities.home * 100).toFixed(1)}%
              </div>
            </div>
            <div className="text-center">
              <div className="font-semibold text-gray-700 mb-1 md:mb-2 text-xs md:text-sm">Draw</div>
              <div className="font-bold text-yellow-600 text-base md:text-lg">
                {(match.ensemble.probabilities.draw * 100).toFixed(1)}%
              </div>
            </div>
            <div className="text-center">
              <div className="font-semibold text-gray-700 mb-1 md:mb-2 text-xs md:text-sm">Away</div>
              <div className="font-bold text-blue-600 text-base md:text-lg">
                {(match.ensemble.probabilities.away * 100).toFixed(1)}%
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}