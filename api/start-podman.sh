#!/bin/bash

# Start URL Checker API with Podman
# This script starts the services individually for better Podman compatibility

echo "🚀 Starting URL Checker API with Podman"
echo "========================================"

# Create network if it doesn't exist
echo "📡 Creating network..."
podman network create url-checker-network 2>/dev/null || echo "Network already exists"

# Create volumes if they don't exist
echo "💾 Creating volumes..."
podman volume create redis_data 2>/dev/null || echo "Redis volume already exists"
podman volume create mysql_data 2>/dev/null || echo "MySQL volume already exists"

# Start Redis
echo "🟥 Starting Redis..."
podman run -d \
  --name url-checker-redis \
  --network url-checker-network \
  -p 6379:6379 \
  -v redis_data:/data \
  redis:7-alpine \
  redis-server --appendonly yes

# Wait a moment for Redis to start
sleep 2

# Start MySQL
echo "🟨 Starting MySQL..."
podman run -d \
  --name url-checker-mysql \
  --network url-checker-network \
  -p 3306:3306 \
  -e MYSQL_ROOT_PASSWORD=password \
  -e MYSQL_DATABASE=url_checker \
  -e MYSQL_USER=api_user \
  -e MYSQL_PASSWORD=api_password \
  -v mysql_data:/var/lib/mysql \
  -v ./scripts/init.sql:/docker-entrypoint-initdb.d/init.sql:ro \
  mysql:8.0

# Wait for MySQL to initialize
echo "⏳ Waiting for MySQL to initialize..."
sleep 15

# Start phpMyAdmin
echo "🟦 Starting phpMyAdmin..."
podman run -d \
  --name url-checker-phpmyadmin \
  --network url-checker-network \
  -p 8080:80 \
  -e PMA_HOST=mysql \
  -e PMA_PORT=3306 \
  -e PMA_USER=root \
  -e PMA_PASSWORD=password \
  phpmyadmin/phpmyadmin

# Build and start API (if Dockerfile exists)
if [ -f "Dockerfile" ]; then
    echo "🟩 Building and starting API..."
    
    # Build the API image
    podman build -t url-checker-api .
    
    # Start the API container
    podman run -d \
      --name url-checker-api \
      --network url-checker-network \
      -p 3000:3000 \
      -e NODE_ENV=production \
      -e PORT=3000 \
      -e REDIS_HOST=redis \
      -e REDIS_PORT=6379 \
      -e MYSQL_HOST=mysql \
      -e MYSQL_PORT=3306 \
      -e MYSQL_USER=root \
      -e MYSQL_PASSWORD=password \
      -e MYSQL_DATABASE=url_checker \
      -e QUEUE_BATCH_SIZE=1000 \
      -e AI_MODELS_PATH=/ai \
      -e LOG_LEVEL=info \
      -v ./logs:/app/logs \
      -v ../ai:/ai:ro \
      url-checker-api
else
    echo "⚠️  Dockerfile not found. Skipping API container."
    echo "💡 You can run the API manually with: node simple-server.js"
fi

echo ""
echo "✅ Services started!"
echo "🔗 Access points:"
echo "   - API: http://localhost:3000"
echo "   - phpMyAdmin: http://localhost:8080"
echo "   - Redis: localhost:6379"
echo "   - MySQL: localhost:3306"
echo ""
echo "📊 Check status with: podman ps"
echo "🛑 Stop all with: ./stop-podman.sh" 