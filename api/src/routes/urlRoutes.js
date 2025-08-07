const express = require('express');
const Joi = require('joi');
const aiService = require('../services/aiService');
const queueService = require('../services/queueService');
const redisClient = require('../config/redis');
const logger = require('../config/logger');

const router = express.Router();

const urlCheckSchema = Joi.object({
    url: Joi.string().uri({ scheme: ['http', 'https'] }).required(),
    model: Joi.string().valid('xg_boost', 'neural_network', 'svm').default('xg_boost')
});

const batchUrlSchema = Joi.object({
    urls: Joi.array().items(Joi.string().uri({ scheme: ['http', 'https'] })).min(1).max(100).required(),
    model: Joi.string().valid('xg_boost', 'neural_network', 'svm').default('xg_boost')
});

router.post('/url_checked', async (req, res) => {
    try {
        const { error, value } = urlCheckSchema.validate(req.body);
        
        if (error) {
            return res.status(400).json({
                success: false,
                error: 'Invalid input',
                details: error.details[0].message
            });
        }

        const { url, model } = value;
        
        const cacheKey = `prediction:${model}:${Buffer.from(url).toString('base64')}`;
        const cachedResult = await redisClient.getCache(cacheKey);
        
        if (cachedResult) {
            logger.info(`Cache hit for URL: ${url}`);
            return res.json({
                isMalicious: cachedResult.is_malicious,
                prediction: cachedResult.prediction,
                confidence: cachedResult.confidence,
                reason: cachedResult.reason,
                model_used: cachedResult.model_used,
                cached: true
            });
        }

        const prediction = await aiService.predictURL(url, model);
        
        await redisClient.setCache(cacheKey, prediction, 3600);
        
        await queueService.addUrlToQueue(url, model);

        const response = {
            isMalicious: prediction.is_malicious,
            prediction: prediction.prediction,
            confidence: prediction.confidence,
            reason: prediction.reason,
            model_used: prediction.model_used,
            cached: false
        };

        logger.info(`URL prediction completed: ${url} - ${prediction.prediction}`, {
            url,
            model,
            prediction: prediction.prediction,
            confidence: prediction.confidence
        });

        res.json(response);

    } catch (error) {
        logger.error('Error in URL check endpoint:', error);
        res.status(500).json({
            success: false,
            error: 'Internal server error',
            message: 'Failed to process URL prediction'
        });
    }
});

router.post('/batch_check', async (req, res) => {
    try {
        const { error, value } = batchUrlSchema.validate(req.body);
        
        if (error) {
            return res.status(400).json({
                success: false,
                error: 'Invalid input',
                details: error.details[0].message
            });
        }

        const { urls, model } = value;
        
        logger.info(`Batch URL check requested for ${urls.length} URLs with model: ${model}`);

        const results = [];
        const cachePromises = urls.map(async (url) => {
            const cacheKey = `prediction:${model}:${Buffer.from(url).toString('base64')}`;
            const cachedResult = await redisClient.getCache(cacheKey);
            return { url, cached: cachedResult };
        });

        const cacheResults = await Promise.all(cachePromises);
        const uncachedUrls = [];

        for (const { url, cached } of cacheResults) {
            if (cached) {
                results.push({
                    url,
                    isMalicious: cached.is_malicious,
                    prediction: cached.prediction,
                    confidence: cached.confidence,
                    reason: cached.reason,
                    model_used: cached.model_used,
                    cached: true
                });
            } else {
                uncachedUrls.push(url);
            }
        }

        if (uncachedUrls.length > 0) {
            const urlData = uncachedUrls.map(url => ({ url, model }));
            const predictions = await aiService.batchPredict(urlData, model);

            for (const prediction of predictions) {
                const cacheKey = `prediction:${model}:${Buffer.from(prediction.url).toString('base64')}`;
                await redisClient.setCache(cacheKey, prediction, 3600);
                
                await queueService.addUrlToQueue(prediction.url, model);

                results.push({
                    url: prediction.url,
                    isMalicious: prediction.is_malicious,
                    prediction: prediction.prediction,
                    confidence: prediction.confidence,
                    reason: prediction.reason,
                    model_used: prediction.model_used,
                    cached: false
                });
            }
        }

        res.json({
            success: true,
            total_processed: urls.length,
            cached_results: results.filter(r => r.cached).length,
            new_predictions: results.filter(r => !r.cached).length,
            results: results
        });

    } catch (error) {
        logger.error('Error in batch check endpoint:', error);
        res.status(500).json({
            success: false,
            error: 'Internal server error',
            message: 'Failed to process batch URL prediction'
        });
    }
});

router.get('/queue/status', async (req, res) => {
    try {
        const status = await queueService.getQueueStatus();
        res.json({
            success: true,
            ...status
        });
    } catch (error) {
        logger.error('Error getting queue status:', error);
        res.status(500).json({
            success: false,
            error: 'Failed to get queue status'
        });
    }
});

router.post('/queue/process', async (req, res) => {
    try {
        const result = await queueService.forceProcessQueue();
        res.json({
            success: true,
            ...result
        });
    } catch (error) {
        logger.error('Error forcing queue processing:', error);
        res.status(500).json({
            success: false,
            error: 'Failed to process queue'
        });
    }
});

router.delete('/queue/clear', async (req, res) => {
    try {
        const result = await queueService.clearQueue();
        res.json(result);
    } catch (error) {
        logger.error('Error clearing queue:', error);
        res.status(500).json({
            success: false,
            error: 'Failed to clear queue'
        });
    }
});

router.get('/models', async (req, res) => {
    try {
        const models = aiService.getAvailableModels();
        res.json({
            success: true,
            available_models: models
        });
    } catch (error) {
        logger.error('Error getting available models:', error);
        res.status(500).json({
            success: false,
            error: 'Failed to get available models'
        });
    }
});

router.get('/health', async (req, res) => {
    try {
        const aiHealth = await aiService.healthCheck();
        const queueStatus = await queueService.getQueueStatus();
        
        res.json({
            success: true,
            status: 'healthy',
            ai_service: aiHealth,
            queue_service: {
                queue_length: queueStatus.queue_length,
                is_processing: queueStatus.is_processing
            },
            timestamp: new Date().toISOString()
        });
    } catch (error) {
        logger.error('Health check failed:', error);
        res.status(503).json({
            success: false,
            status: 'unhealthy',
            error: error.message,
            timestamp: new Date().toISOString()
        });
    }
});

module.exports = router; 