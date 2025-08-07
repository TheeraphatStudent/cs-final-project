# URL Checker API

A comprehensive Express.js API that integrates with AI models for malicious URL detection, using Redis for queueing and MySQL for persistent storage.

## Features

- **RESTful API** with Express.js
- **AI Model Integration** - Supports XGBoost, Neural Network, and SVM models
- **Redis Queue System** - Batch processing every 1000 URLs
- **MySQL Database** - Persistent storage for predictions
- **Docker Compose** - Complete containerized setup
- **Caching** - Redis-based response caching
- **Rate Limiting** - Protection against abuse
- **Comprehensive Logging** - Winston-based logging system
- **Health Checks** - System monitoring endpoints

## API Endpoints

### Main Endpoint
```
POST /api/url_checked
```

**Request Body:**
```json
{
    "url": "https://www.google.com",
    "model": "xg_boost"
}
```

**Response:**
```json
{
    "isMalicious": false,
    "prediction": "safe",
    "confidence": 0.95,
    "reason": "Known safe domain",
    "model_used": "xg_boost",
    "cached": false
}
```

### Additional Endpoints

- `POST /api/batch_check` - Check multiple URLs
- `GET /api/queue/status` - Get queue status
- `POST /api/queue/process` - Force process queue
- `DELETE /api/queue/clear` - Clear queue
- `GET /api/models` - Available AI models
- `GET /api/health` - Health check

## Quick Start

### Using Docker Compose

1. **Clone and navigate to the API directory:**
   ```bash
   cd api
   ```

2. **Start all services:**
   ```bash
   docker-compose up -d
   ```

3. **Test the API:**
   ```bash
   curl -X POST http://localhost:3000/api/url_checked \
     -H "Content-Type: application/json" \
     -d '{"url": "https://www.google.com", "model": "xg_boost"}'
   ```

4. **Access services:**
   - API: http://localhost:3000
   - phpMyAdmin: http://localhost:8080
   - Redis: localhost:6379
   - MySQL: localhost:3306

### Manual Setup

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Set up environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start Redis and MySQL separately**

4. **Run the application:**
   ```bash
   npm start
   ```

## Environment Variables

```env
PORT=3000
NODE_ENV=development

REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=

MYSQL_HOST=mysql
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=password
MYSQL_DATABASE=url_checker

QUEUE_BATCH_SIZE=1000
AI_MODELS_PATH=/ai
LOG_LEVEL=info
```

## Architecture

### Components

1. **Express.js API Server**
   - RESTful endpoints
   - Request validation (Joi)
   - Rate limiting
   - CORS and security headers

2. **Redis Integration**
   - Queue management
   - Response caching
   - Pub/Sub for batch processing

3. **MySQL Database**
   - Persistent storage
   - Batch insertions
   - Performance indexes

4. **AI Model Integration**
   - Python subprocess execution
   - Multiple model support
   - Error handling and timeouts

### Data Flow

1. URL received via POST request
2. Check Redis cache for existing prediction
3. If not cached, call AI model for prediction
4. Cache result in Redis
5. Add URL to processing queue
6. When queue reaches 1000 items, batch insert to MySQL
7. Return prediction result

## Database Schema

```sql
CREATE TABLE url_container (
    id INT AUTO_INCREMENT PRIMARY KEY,
    url VARCHAR(2048) NOT NULL,
    model_used VARCHAR(50) NOT NULL,
    prediction VARCHAR(20) NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    reason TEXT,
    is_malicious BOOLEAN NOT NULL,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Indexes for performance
    INDEX idx_url (url(255)),
    INDEX idx_model (model_used),
    INDEX idx_prediction (prediction),
    INDEX idx_processed_at (processed_at)
);
```

## Available AI Models

- **xg_boost** - XGBoost classifier (default)
- **neural_network** - Deep learning model
- **svm** - Support Vector Machine

## Rate Limiting

- **URL Check**: 100 requests per 15 minutes per IP
- **Batch Check**: 10 requests per 15 minutes per IP

## Monitoring

### Health Check
```bash
curl http://localhost:3000/api/health
```

### Queue Status
```bash
curl http://localhost:3000/api/queue/status
```

### Logs
- Error logs: `logs/error.log`
- Combined logs: `logs/combined.log`
- Console output in development mode

## Development

### Running in Development Mode
```bash
npm run dev
```

### Testing
```bash
npm test
```

### Adding New AI Models

1. Create prediction script in `/ai/your_model/predict.py`
2. Update `aiService.js` modelPaths
3. Ensure script returns JSON format:
   ```json
   {
     "prediction": "safe|malicious",
     "confidence": 0.95,
     "reason": "Explanation"
   }
   ```

## Production Deployment

### Docker Compose (Recommended)
```bash
docker-compose -f docker-compose.yml up -d
```

### Environment Considerations
- Set `NODE_ENV=production`
- Use strong passwords for database
- Configure proper Redis persistence
- Set up SSL/TLS termination
- Configure proper logging levels
- Use process managers (PM2)

## Troubleshooting

### Common Issues

1. **AI Model not found**
   - Ensure AI models are properly mounted in Docker
   - Check `AI_MODELS_PATH` environment variable

2. **Database connection issues**
   - Verify MySQL is running and accessible
   - Check database credentials
   - Ensure database `url_checker` exists

3. **Redis connection issues**
   - Verify Redis is running
   - Check Redis host and port configuration

4. **Python dependencies**
   - Ensure all Python packages are installed
   - Check Python path configuration

### Logs
Check application logs for detailed error information:
```bash
docker-compose logs api
```

## Security

- Rate limiting to prevent abuse
- Input validation with Joi
- Security headers with Helmet
- Environment-based configuration
- Non-root Docker container execution

## Performance

- Redis caching for faster responses
- Batch processing for database efficiency
- Connection pooling for database
- Indexed database queries
- Horizontal scaling ready

## Contributing

1. Fork the repository
2. Create feature branch
3. Follow coding standards
4. Add tests for new features
5. Submit pull request

## License

MIT License 