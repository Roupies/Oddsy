#!/bin/bash
# Complete setup script for Oddsy production environment
set -e

echo "🚀 Setting up Oddsy production environment..."

# Check if we're in the right directory
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ Error: Please run this script from the /prod directory"
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running. Please start Docker first."
    exit 1
fi

echo "1️⃣ Starting PostgreSQL and pgAdmin..."
docker-compose up -d postgres pgadmin

echo "2️⃣ Waiting for PostgreSQL to be ready..."
sleep 10

echo "3️⃣ Running database migrations..."
./scripts/migrate.sh

echo "4️⃣ Installing backend dependencies..."
cd backend
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install -r requirements.txt
cd ..

echo "5️⃣ Installing frontend dependencies..."
cd frontend
npm install
cd ..

echo "✅ Setup complete! You can now start the services:"
echo ""
echo "Backend:  cd backend && source venv/bin/activate && uvicorn main:app --reload --port 8000"
echo "Frontend: cd frontend && npm run dev"
echo ""
echo "🌐 URLs:"
echo "  Frontend: http://localhost:3000"
echo "  Backend:  http://localhost:8000"
echo "  pgAdmin:  http://localhost:8080 (admin@oddsy.local / admin_password)"