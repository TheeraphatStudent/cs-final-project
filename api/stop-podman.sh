#!/bin/bash

# Stop URL Checker API Podman containers

echo "🛑 Stopping URL Checker API containers..."
echo "========================================"

# Stop and remove containers
echo "Stopping containers..."
podman stop url-checker-api url-checker-phpmyadmin url-checker-mysql url-checker-redis 2>/dev/null
echo "Removing containers..."
podman rm url-checker-api url-checker-phpmyadmin url-checker-mysql url-checker-redis 2>/dev/null

# Optional: Remove network and volumes (uncomment if you want to clean everything)
# echo "Removing network..."
# podman network rm url-checker-network 2>/dev/null
# echo "Removing volumes..."
# podman volume rm redis_data mysql_data 2>/dev/null

echo "✅ All containers stopped and removed!"
echo "💡 Data volumes are preserved. Use 'podman volume rm redis_data mysql_data' to remove them." 