require('dotenv').config();

const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const path = require('path');
const fs = require('fs');

const logger = require('./config/logger');
const urlRoutes = require('./routes/urlRoutes');
const queueService = require('./services/queueService');
const {
    urlCheckRateLimit,
    batchCheckRateLimit,
    requestLogger,
    errorHandler,
    notFoundHandler,
    validateContentType,
    healthCheck
} = require('./middleware/validation');

const app = express();
const PORT = process.env.PORT || 3000;

if (!fs.existsSync('logs')) {
    fs.mkdirSync('logs');
}

app.use(helmet({
    contentSecurityPolicy: {
        directives: {
            defaultSrc: ["'self'"],
            styleSrc: ["'self'", "'unsafe-inline'"],
            scriptSrc: ["'self'"],
            imgSrc: ["'self'", "data:", "https:"],
        }
    },
    hsts: {
        maxAge: 31536000,
        includeSubDomains: true,
        preload: true
    }
}));

app.use(cors({
    origin: process.env.ALLOWED_ORIGINS ? process.env.ALLOWED_ORIGINS.split(',') : '*',
    methods: ['GET', 'POST', 'PUT', 'DELETE'],
    allowedHeaders: ['Content-Type', 'Authorization'],
    credentials: true
}));

app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

app.use(requestLogger);
app.use(healthCheck);
app.use(validateContentType);

app.get('/', (req, res) => {
    res.json({
        success: true,
        message: 'URL Checker API is running',
        version: '1.0.0',
        endpoints: {
            'POST /api/url_checked': 'Check single URL for malicious content',
            'POST /api/batch_check': 'Check multiple URLs for malicious content',
            'GET /api/queue/status': 'Get queue processing status',
            'POST /api/queue/process': 'Force process queue',
            'DELETE /api/queue/clear': 'Clear processing queue',
            'GET /api/models': 'Get available AI models',
            'GET /api/health': 'Health check endpoint'
        },
        documentation: 'https://your-docs-url.com'
    });
});

app.use('/api/url_checked', urlCheckRateLimit);
app.use('/api/batch_check', batchCheckRateLimit);

app.use('/api', urlRoutes);

app.use(notFoundHandler);
app.use(errorHandler);

async function startServer() {
    try {
        logger.info('Starting URL Checker API server...');

        await new Promise((resolve) => {
            setTimeout(resolve, 2000);
        });

        queueService.schedulePeriodicProcessing(5);

        const server = app.listen(PORT, '0.0.0.0', () => {
            logger.info(`Server is running on port ${PORT}`, {
                port: PORT,
                environment: process.env.NODE_ENV || 'development',
                nodeVersion: process.version
            });
        });

        const gracefulShutdown = async (signal) => {
            logger.info(`Received ${signal}. Starting graceful shutdown...`);
            
            server.close(async () => {
                try {
                    const redisClient = require('./config/redis');
                    const database = require('./config/database');
                    
                    await redisClient.close();
                    await database.close();
                    
                    logger.info('Graceful shutdown completed');
                    process.exit(0);
                } catch (error) {
                    logger.error('Error during shutdown:', error);
                    process.exit(1);
                }
            });

            setTimeout(() => {
                logger.warn('Forced shutdown after timeout');
                process.exit(1);
            }, 10000);
        };

        process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
        process.on('SIGINT', () => gracefulShutdown('SIGINT'));

        process.on('uncaughtException', (error) => {
            logger.error('Uncaught Exception:', error);
            process.exit(1);
        });

        process.on('unhandledRejection', (reason, promise) => {
            logger.error('Unhandled Rejection at:', promise, 'reason:', reason);
        });

    } catch (error) {
        logger.error('Failed to start server:', error);
        process.exit(1);
    }
}

if (require.main === module) {
    startServer();
}

module.exports = app; 