-- =====================================================
-- ODDSY FOOTBALL DATABASE SCHEMA
-- =====================================================
-- Database initialization script for PostgreSQL
-- Creates tables for football match predictions system

-- Drop existing tables if they exist (for clean re-initialization)
DROP TABLE IF EXISTS predictions CASCADE;
DROP TABLE IF EXISTS model_performance CASCADE;
DROP TABLE IF EXISTS matches CASCADE;
DROP TABLE IF EXISTS teams CASCADE;

-- =====================================================
-- TEAMS TABLE
-- =====================================================
CREATE TABLE teams (
    team_id SERIAL PRIMARY KEY,
    team_name VARCHAR(100) NOT NULL UNIQUE,
    short_name VARCHAR(10),
    league VARCHAR(50) DEFAULT 'Premier League',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert Premier League teams
INSERT INTO teams (team_name, short_name) VALUES
('Arsenal', 'ARS'),
('Aston Villa', 'AVL'),
('Brighton', 'BHA'),
('Burnley', 'BUR'),
('Chelsea', 'CHE'),
('Crystal Palace', 'CRY'),
('Everton', 'EVE'),
('Fulham', 'FUL'),
('Leeds', 'LEE'),
('Leicester', 'LEI'),
('Liverpool', 'LIV'),
('Man City', 'MCI'),
('Man United', 'MUN'),
('Newcastle', 'NEW'),
('Nott''m Forest', 'NFO'),
('Southampton', 'SOU'),
('Sunderland', 'SUN'),
('Tottenham', 'TOT'),
('West Ham', 'WHU'),
('Wolves', 'WOL'),
('Bournemouth', 'BOU'),
('Brentford', 'BRE');

-- =====================================================
-- MATCHES TABLE
-- =====================================================
CREATE TABLE matches (
    match_id SERIAL PRIMARY KEY,
    match_date DATE NOT NULL,
    season VARCHAR(9) NOT NULL, -- Format: 2025-2026
    matchday INTEGER,
    home_team_id INTEGER REFERENCES teams(team_id),
    away_team_id INTEGER REFERENCES teams(team_id),
    
    -- Actual results
    full_time_result CHAR(1) CHECK (full_time_result IN ('H', 'D', 'A')),
    home_goals INTEGER,
    away_goals INTEGER,
    
    -- Betting odds
    home_odds DECIMAL(5,2),
    draw_odds DECIMAL(5,2),
    away_odds DECIMAL(5,2),
    
    -- Advanced stats
    home_shots INTEGER,
    away_shots INTEGER,
    home_shots_target INTEGER,
    away_shots_target INTEGER,
    home_corners INTEGER,
    away_corners INTEGER,
    home_xg DECIMAL(4,2),
    away_xg DECIMAL(4,2),
    
    -- Metadata
    data_source VARCHAR(50) DEFAULT 'manual',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    UNIQUE(match_date, home_team_id, away_team_id),
    CHECK (home_team_id != away_team_id)
);

-- =====================================================
-- MODEL PERFORMANCE TABLE
-- =====================================================
CREATE TABLE model_performance (
    performance_id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    evaluation_date DATE NOT NULL,
    dataset_used VARCHAR(100),
    
    -- Performance metrics
    accuracy DECIMAL(5,4),
    precision_home DECIMAL(5,4),
    precision_draw DECIMAL(5,4),
    precision_away DECIMAL(5,4),
    recall_home DECIMAL(5,4),
    recall_draw DECIMAL(5,4),
    recall_away DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    log_loss DECIMAL(6,4),
    
    -- Additional metadata
    total_predictions INTEGER,
    correct_predictions INTEGER,
    hyperparameters JSONB,
    feature_importance JSONB,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(model_name, model_version, evaluation_date)
);

-- =====================================================
-- PREDICTIONS TABLE
-- =====================================================
CREATE TABLE predictions (
    prediction_id SERIAL PRIMARY KEY,
    match_id INTEGER REFERENCES matches(match_id),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    
    -- Predictions
    predicted_result CHAR(1) CHECK (predicted_result IN ('H', 'D', 'A')),
    probability_home DECIMAL(5,4) NOT NULL,
    probability_draw DECIMAL(5,4) NOT NULL,
    probability_away DECIMAL(5,4) NOT NULL,
    confidence_score DECIMAL(5,4),
    
    -- Features used (stored as JSON for flexibility)
    features_used JSONB,
    
    -- Evaluation
    is_correct BOOLEAN,
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CHECK (probability_home + probability_draw + probability_away BETWEEN 0.99 AND 1.01),
    CHECK (probability_home >= 0 AND probability_home <= 1),
    CHECK (probability_draw >= 0 AND probability_draw <= 1),
    CHECK (probability_away >= 0 AND probability_away <= 1)
);

-- =====================================================
-- INDEXES FOR PERFORMANCE
-- =====================================================
CREATE INDEX idx_matches_date ON matches(match_date);
CREATE INDEX idx_matches_season ON matches(season);
CREATE INDEX idx_matches_teams ON matches(home_team_id, away_team_id);
CREATE INDEX idx_predictions_match ON predictions(match_id);
CREATE INDEX idx_predictions_model ON predictions(model_name, model_version);
CREATE INDEX idx_model_performance_date ON model_performance(evaluation_date);

-- =====================================================
-- VIEWS FOR COMMON QUERIES
-- =====================================================

-- View: Match results with team names
CREATE VIEW match_results AS
SELECT 
    m.match_id,
    m.match_date,
    m.season,
    m.matchday,
    ht.team_name as home_team,
    at.team_name as away_team,
    m.full_time_result,
    m.home_goals,
    m.away_goals,
    m.home_odds,
    m.draw_odds,
    m.away_odds,
    m.home_xg,
    m.away_xg
FROM matches m
JOIN teams ht ON m.home_team_id = ht.team_id
JOIN teams at ON m.away_team_id = at.team_id;

-- View: Model performance summary
CREATE VIEW model_performance_summary AS
SELECT 
    model_name,
    model_version,
    COUNT(*) as evaluations,
    AVG(accuracy) as avg_accuracy,
    MAX(accuracy) as best_accuracy,
    MAX(evaluation_date) as last_evaluation
FROM model_performance
GROUP BY model_name, model_version
ORDER BY avg_accuracy DESC;

-- View: Prediction accuracy by model
CREATE VIEW prediction_accuracy AS
SELECT 
    p.model_name,
    p.model_version,
    COUNT(*) as total_predictions,
    SUM(CASE WHEN p.is_correct THEN 1 ELSE 0 END) as correct_predictions,
    ROUND(AVG(CASE WHEN p.is_correct THEN 1.0 ELSE 0.0 END), 4) as accuracy,
    ROUND(AVG(p.confidence_score), 4) as avg_confidence
FROM predictions p
WHERE p.is_correct IS NOT NULL
GROUP BY p.model_name, p.model_version
ORDER BY accuracy DESC;

-- =====================================================
-- SAMPLE DATA INSERTION
-- =====================================================
-- Insert a few sample matches for testing
INSERT INTO matches (match_date, season, matchday, home_team_id, away_team_id, full_time_result, home_goals, away_goals, home_odds, draw_odds, away_odds)
VALUES 
('2025-08-15', '2025-2026', 1, 1, 2, 'H', 2, 1, 2.10, 3.40, 3.20),
('2025-08-15', '2025-2026', 1, 3, 4, 'D', 1, 1, 2.80, 3.10, 2.90),
('2025-08-16', '2025-2026', 1, 5, 6, 'A', 0, 2, 1.95, 3.60, 3.80);

COMMENT ON TABLE teams IS 'Premier League teams master data';
COMMENT ON TABLE matches IS 'Historical and current EPL match results with statistics';
COMMENT ON TABLE predictions IS 'Model predictions for matches with probabilities';
COMMENT ON TABLE model_performance IS 'Model evaluation metrics and performance tracking';