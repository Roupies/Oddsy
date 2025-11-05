#!/bin/bash
# Migration script for Oddsy database
set -e

echo "🔄 Starting database migrations..."

# Database connection parameters
DB_HOST=${DB_HOST:-localhost}
DB_PORT=${DB_PORT:-5432}
DB_NAME=${DB_NAME:-oddsy_football}
DB_USER=${DB_USER:-oddsy_user}
DB_PASSWORD=${DB_PASSWORD:-oddsy_password}

# Wait for database to be ready
echo "⏳ Waiting for database to be ready..."
until PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c '\q' 2>/dev/null; do
  echo "Database not ready, waiting 2 seconds..."
  sleep 2
done

echo "✅ Database is ready!"

# Run migrations (add your migration logic here)
echo "🚀 Running migrations..."

# Example: Create tables if they don't exist
PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME << 'EOF'
-- Create predictions table if not exists
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    gameweek INTEGER NOT NULL,
    home_team VARCHAR(50) NOT NULL,
    away_team VARCHAR(50) NOT NULL,
    prediction VARCHAR(10) NOT NULL,
    confidence FLOAT NOT NULL,
    home_prob FLOAT NOT NULL,
    draw_prob FLOAT NOT NULL,
    away_prob FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create fixtures table if not exists
CREATE TABLE IF NOT EXISTS fixtures (
    id SERIAL PRIMARY KEY,
    gameweek INTEGER NOT NULL,
    home_team VARCHAR(50) NOT NULL,
    away_team VARCHAR(50) NOT NULL,
    kickoff_utc TIMESTAMP NOT NULL,
    venue VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_predictions_gameweek ON predictions(gameweek);
CREATE INDEX IF NOT EXISTS idx_fixtures_gameweek ON fixtures(gameweek);
CREATE INDEX IF NOT EXISTS idx_fixtures_kickoff ON fixtures(kickoff_utc);
EOF

echo "✅ Database migrations completed successfully!"