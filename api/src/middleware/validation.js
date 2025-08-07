const rateLimit = require('express-rate-limit');
const logger = require('../config/logger');

const urlCheckRateLimit = rateLimit({
    windowMs: 15 * 60 * 1000,
    max: 100,
    message: {
        success: false,
        error: 'Too many requests',
        message: 'Rate limit exceeded. Please try again later.'
    },
    standardHeaders: true,
    legacyHeaders: false,
    keyGenerator: (req) => {
        return req.ip || req.connection.remoteAddress;
    },
    onLimitReached: (req) => {
        logger.warn(`Rate limit exceeded for IP: ${req.ip}`, {
            ip: req.ip,
            userAgent: req.get('User-Agent'),
            endpoint: req.path
        });
    }
});

const batchCheckRateLimit = rateLimit({
    windowMs: 15 * 60 * 1000,
    max: 10,
    message: {
        success: false,
        error: 'Too many batch requests',
        message: 'Batch rate limit exceeded. Please try again later.'
    },
    standardHeaders: true,
    legacyHeaders: false,
    keyGenerator: (req) => {
        return req.ip || req.connection.remoteAddress;
    }
});

const requestLogger = (req, res, next) => {
    const startTime = Date.now();
    
    res.on('finish', () => {
        const duration = Date.now() - startTime;
        logger.info('Request completed', {
            method: req.method,
            url: req.url,
            statusCode: res.statusCode,
            duration: `${duration}ms`,
            ip: req.ip,
            userAgent: req.get('User-Agent')
        });
    });

    next();
};

const errorHandler = (err, req, res, next) => {
    logger.error('Unhandled error:', {
        error: err.message,
        stack: err.stack,
        url: req.url,
        method: req.method,
        ip: req.ip
    });

    if (err.name === 'ValidationError') {
        return res.status(400).json({
            success: false,
            error: 'Validation error',
            details: err.message
        });
    }

    if (err.name === 'SyntaxError' && err.status === 400 && 'body' in err) {
        return res.status(400).json({
            success: false,
            error: 'Invalid JSON',
            message: 'Request body contains invalid JSON'
        });
    }

    res.status(500).json({
        success: false,
        error: 'Internal server error',
        message: process.env.NODE_ENV === 'development' ? err.message : 'Something went wrong'
    });
};

const notFoundHandler = (req, res) => {
    logger.warn(`404 - Route not found: ${req.method} ${req.url}`, {
        ip: req.ip,
        userAgent: req.get('User-Agent')
    });

    res.status(404).json({
        success: false,
        error: 'Route not found',
        message: `Cannot ${req.method} ${req.url}`
    });
};

const validateContentType = (req, res, next) => {
    if (req.method === 'POST' || req.method === 'PUT' || req.method === 'PATCH') {
        if (!req.is('application/json')) {
            return res.status(400).json({
                success: false,
                error: 'Invalid content type',
                message: 'Content-Type must be application/json'
            });
        }
    }
    next();
};

const healthCheck = (req, res, next) => {
    if (req.path === '/health' || req.path === '/api/health') {
        return res.json({
            success: true,
            status: 'healthy',
            timestamp: new Date().toISOString(),
            uptime: process.uptime()
        });
    }
    next();
};

module.exports = {
    urlCheckRateLimit,
    batchCheckRateLimit,
    requestLogger,
    errorHandler,
    notFoundHandler,
    validateContentType,
    healthCheck
}; 