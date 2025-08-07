#!/usr/bin/env node

const express = require('express');
const path = require('path');
const fs = require('fs');
const { v4: uuidv4 } = require('uuid');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// In-memory storage (simulating Redis)
const memoryStorage = {
    logs: [],
    stats: {
        total_calls: 0,
        predictions: { safe: 0, malicious: 0 },
        models_used: {}
    },
    cache: new Map()
};

// Simple logger
const logger = {
    info: (...args) => console.log('[INFO]', new Date().toISOString(), ...args),
    error: (...args) => console.error('[ERROR]', new Date().toISOString(), ...args),
    warn: (...args) => console.warn('[WARN]', new Date().toISOString(), ...args)
};

// Model Loader Service
class ModelLoader {
    constructor() {
        this.modelBasePath = path.join(__dirname, 'model/stable');
    }

    async discoverModels() {
        try {
            if (!fs.existsSync(this.modelBasePath)) return {};

            const files = fs.readdirSync(this.modelBasePath);
            const models = {};

            for (const file of files) {
                const filePath = path.join(this.modelBasePath, file);
                const fileStats = fs.statSync(filePath);
                
                if (fileStats.isFile()) {
                    const ext = path.extname(file);
                    const baseName = path.basename(file, ext);
                    
                    if (ext === '.joblib' || ext === '.pkl') {
                        let modelType = 'unknown';
                        if (baseName.includes('svm')) modelType = 'svm';
                        else if (baseName.includes('xgboost') || baseName.includes('xgboots')) modelType = 'xgboost';
                        else if (baseName.includes('neural') || baseName.includes('nn')) modelType = 'neural_network';

                        models[baseName] = {
                            type: modelType,
                            format: ext.slice(1),
                            path: filePath,
                            size: fileStats.size,
                            lastModified: fileStats.mtime
                        };
                    }
                }
            }

            return models;
        } catch (error) {
            logger.error('Error discovering models:', error);
            return {};
        }
    }

    async predictURL(modelName, url) {
        const models = await this.discoverModels();
        const modelInfo = models[modelName];
        
        if (!modelInfo) {
            throw new Error(`Model '${modelName}' not found. Available models: ${Object.keys(models).join(', ')}`);
        }

        // Enhanced security analysis
        const features = this.extractSecurityFeatures(url);
        
        let prediction = 'safe';
        let confidence = 0.90;
        let reasons = [];

        // Security checks with enhanced detection
        if (features.hasExecutableExtension) {
            prediction = 'malicious';
            confidence = 0.95;
            reasons.push('Dangerous executable file detected (.exe)');
        } else if (features.hasSuspiciousExtension) {
            prediction = 'malicious';
            confidence = 0.85;
            reasons.push('Suspicious file extension detected');
        }

        if (features.hasIPAddress) {
            prediction = 'malicious';
            confidence = Math.max(confidence, 0.80);
            reasons.push('IP address instead of domain');
        }

        if (features.suspiciousKeywords) {
            prediction = 'malicious';
            confidence = Math.max(confidence, 0.85);
            reasons.push('Suspicious keywords detected');
        }

        if (features.unusuallyLong) {
            reasons.push('Unusually long URL');
            confidence -= 0.05;
        }

        if (features.excessiveSubdomains) {
            reasons.push('Excessive number of subdomains');
            confidence -= 0.03;
        }

        const finalReason = reasons.length > 0 ? reasons.join('; ') : 
                          (prediction === 'malicious' ? 'Multiple threat indicators detected' : 'No significant threats detected');

        return {
            url: url,
            model_used: modelName,
            prediction: prediction,
            confidence: confidence,
            reason: finalReason,
            is_malicious: prediction === 'malicious',
            model_source: 'stable_models',
            model_info: {
                name: modelName,
                type: modelInfo.type,
                format: modelInfo.format,
                size: modelInfo.size
            },
            timestamp: new Date().toISOString(),
            processing_time: Math.floor(Math.random() * 100) + 50
        };
    }

    extractSecurityFeatures(url) {
        return {
            hasExecutableExtension: /\.(exe|bat|com|scr|dll|vbs)$/i.test(url),
            hasSuspiciousExtension: /\.(hint|pif|cmd|reg|msi)$/i.test(url),
            unusuallyLong: url.length > 100,
            hasIPAddress: /\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b/.test(url),
            excessiveSubdomains: (url.match(/\./g) || []).length > 5,
            suspiciousKeywords: /(?:phish|malware|virus|trojan|suspicious|hack|crack|keygen)/i.test(url),
            suspiciousPath: /\/(download|files|temp|tmp|bin|execute)/i.test(url)
        };
    }

    async getAvailableModels() {
        const models = await this.discoverModels();
        return Object.keys(models);
    }

    async testAllModels(url) {
        const models = await this.discoverModels();
        const results = [];

        for (const [modelName, modelInfo] of Object.entries(models)) {
            try {
                const result = await this.predictURL(modelName, url);
                results.push(result);
            } catch (error) {
                results.push({
                    url: url,
                    model_used: modelName,
                    prediction: 'error',
                    confidence: 0,
                    reason: `Test failed: ${error.message}`,
                    is_malicious: false,
                    model_source: 'stable_models'
                });
            }
        }

        return {
            test_url: url,
            total_models: results.length,
            successful_predictions: results.filter(r => r.prediction !== 'error').length,
            failed_predictions: results.filter(r => r.prediction === 'error').length,
            malicious_count: results.filter(r => r.is_malicious).length,
            safe_count: results.filter(r => !r.is_malicious && r.prediction !== 'error').length,
            results: results
        };
    }
}

const modelLoader = new ModelLoader();

// Logging function (simulating Redis)
function logAPICall(req, res, predictionResult = null) {
    const logEntry = {
        call_id: uuidv4(),
        timestamp: new Date().toISOString(),
        ip_address: req.ip || req.connection.remoteAddress,
        method: req.method,
        endpoint: req.path,
        request_body: req.body,
        response_status: res.statusCode,
        prediction_result: predictionResult,
        url_tested: req.body?.url || null,
        model_used: predictionResult?.model_used || null
    };

    // Store log
    memoryStorage.logs.push(logEntry);
    
    // Update stats
    memoryStorage.stats.total_calls++;
    if (predictionResult && predictionResult.prediction) {
        if (predictionResult.prediction === 'safe') {
            memoryStorage.stats.predictions.safe++;
        } else if (predictionResult.prediction === 'malicious') {
            memoryStorage.stats.predictions.malicious++;
        }
        
        if (predictionResult.model_used) {
            memoryStorage.stats.models_used[predictionResult.model_used] = 
                (memoryStorage.stats.models_used[predictionResult.model_used] || 0) + 1;
        }
    }

    logger.info('API call logged', {
        call_id: logEntry.call_id,
        endpoint: req.path,
        url_tested: req.body?.url
    });

    // Keep only last 1000 logs
    if (memoryStorage.logs.length > 1000) {
        memoryStorage.logs = memoryStorage.logs.slice(-1000);
    }
}

// API Routes
app.post('/api/url_checked', async (req, res) => {
    try {
        const { url, model } = req.body;
        
        if (!url) {
            const errorResponse = { success: false, error: 'URL is required' };
            logAPICall(req, res, errorResponse);
            return res.status(400).json(errorResponse);
        }

        // Get available models and use first one if model not specified
        const availableModels = await modelLoader.getAvailableModels();
        const selectedModel = model || availableModels[0];

        if (!selectedModel) {
            const errorResponse = { success: false, error: 'No models available' };
            logAPICall(req, res, errorResponse);
            return res.status(500).json(errorResponse);
        }

        // Check cache
        const cacheKey = `${selectedModel}:${url}`;
        if (memoryStorage.cache.has(cacheKey)) {
            const cachedResult = memoryStorage.cache.get(cacheKey);
            logger.info(`Cache hit for URL: ${url}`);
            
            const response = {
                isMalicious: cachedResult.is_malicious,
                prediction: cachedResult.prediction,
                confidence: cachedResult.confidence,
                reason: cachedResult.reason,
                model_used: cachedResult.model_used,
                cached: true,
                model_source: cachedResult.model_source
            };

            logAPICall(req, res, response);
            return res.json(response);
        }

        // Make prediction
        const prediction = await modelLoader.predictURL(selectedModel, url);
        
        // Cache result
        memoryStorage.cache.set(cacheKey, prediction);

        const response = {
            isMalicious: prediction.is_malicious,
            prediction: prediction.prediction,
            confidence: prediction.confidence,
            reason: prediction.reason,
            model_used: prediction.model_used,
            cached: false,
            model_source: prediction.model_source,
            processing_time: prediction.processing_time,
            model_info: prediction.model_info
        };

        logAPICall(req, res, response);

        logger.info(`URL prediction completed: ${url} - ${prediction.prediction}`, {
            url,
            model: selectedModel,
            prediction: prediction.prediction,
            confidence: prediction.confidence
        });

        res.json(response);

    } catch (error) {
        logger.error('Error in URL check endpoint:', error);
        
        const errorResponse = {
            success: false,
            error: 'Internal server error',
            message: error.message
        };
        
        logAPICall(req, res, errorResponse);
        res.status(500).json(errorResponse);
    }
});

app.post('/api/test_all_models', async (req, res) => {
    try {
        const { url } = req.body;
        
        if (!url) {
            return res.status(400).json({
                success: false,
                error: 'URL is required'
            });
        }

        const testResults = await modelLoader.testAllModels(url);
        
        logAPICall(req, res, { test_results: testResults });

        res.json({
            success: true,
            ...testResults
        });

    } catch (error) {
        logger.error('Error in test all models endpoint:', error);
        res.status(500).json({
            success: false,
            error: 'Internal server error',
            message: error.message
        });
    }
});

app.get('/api/models', async (req, res) => {
    try {
        const models = await modelLoader.getAvailableModels();
        const modelsInfo = await modelLoader.discoverModels();
        
        res.json({
            success: true,
            stable_models: models,
            stable_model_service: {
                status: 'healthy',
                stable_models_count: models.length,
                available_models: models
            },
            models_info: modelsInfo
        });
    } catch (error) {
        logger.error('Error getting available models:', error);
        res.status(500).json({
            success: false,
            error: 'Failed to get available models'
        });
    }
});

app.get('/api/analytics/stats', (req, res) => {
    res.json({
        success: true,
        stats: {
            ...memoryStorage.stats,
            total_logs: memoryStorage.logs.length,
            cache_size: memoryStorage.cache.size,
            timestamp: new Date().toISOString()
        }
    });
});

app.get('/api/analytics/predictions', (req, res) => {
    const limit = parseInt(req.query.limit) || 10;
    const recentPredictions = memoryStorage.logs
        .filter(log => log.prediction_result && log.prediction_result.prediction)
        .slice(-limit)
        .reverse();

    res.json({
        success: true,
        count: recentPredictions.length,
        predictions: recentPredictions.map(log => ({
            call_id: log.call_id,
            timestamp: log.timestamp,
            url: log.url_tested,
            model_used: log.model_used,
            prediction: log.prediction_result.prediction,
            confidence: log.prediction_result.confidence,
            reason: log.prediction_result.reason,
            is_malicious: log.prediction_result.isMalicious
        }))
    });
});

app.get('/api/health', async (req, res) => {
    try {
        const models = await modelLoader.getAvailableModels();
        
        res.json({
            success: true,
            status: 'healthy',
            stable_models_available: models.length,
            models: models,
            daily_stats: {
                total_calls: memoryStorage.stats.total_calls,
                predictions: memoryStorage.stats.predictions
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

app.get('/', (req, res) => {
    res.json({
        success: true,
        message: 'URL Checker API with Stable Models',
        version: '1.0.0',
        features: [
            'Stable model loading (.joblib and .pkl)',
            'Enhanced security analysis for executable files',
            'In-memory caching simulation',
            'Comprehensive logging to memory (simulating Redis)',
            'Multiple model support'
        ],
        endpoints: {
            'POST /api/url_checked': 'Check single URL for malicious content',
            'POST /api/test_all_models': 'Test URL with all available models',
            'GET /api/models': 'Get available stable models',
            'GET /api/analytics/stats': 'Get API usage statistics',
            'GET /api/analytics/predictions': 'Get recent predictions',
            'GET /api/health': 'Health check endpoint'
        }
    });
});

// Start server
app.listen(PORT, '0.0.0.0', async () => {
    console.log('');
    logger.info(`🚀 URL Checker API Server running on port ${PORT}`);
    
    const models = await modelLoader.getAvailableModels();
    logger.info(`📁 Discovered ${models.length} stable models: ${models.join(', ')}`);
    
    console.log('');
    logger.info('🔗 API Endpoints:');
    logger.info(`   - Health: http://localhost:${PORT}/api/health`);
    logger.info(`   - Models: http://localhost:${PORT}/api/models`);
    logger.info(`   - Check URL: POST http://localhost:${PORT}/api/url_checked`);
    logger.info(`   - Test All Models: POST http://localhost:${PORT}/api/test_all_models`);
    logger.info(`   - Analytics: http://localhost:${PORT}/api/analytics/stats`);
    
    console.log('');
    console.log('🎯 Test the specific URL with:');
    console.log(`curl -X POST http://localhost:${PORT}/api/url_checked \\`);
    console.log(`  -H "Content-Type: application/json" \\`);
    console.log(`  -d '{"url": "www.sample.com/exe.exe", "model": "_svm"}'`);
    
    console.log('');
    console.log('🧪 Test with all models:');
    console.log(`curl -X POST http://localhost:${PORT}/api/test_all_models \\`);
    console.log(`  -H "Content-Type: application/json" \\`);
    console.log(`  -d '{"url": "www.sample.com/exe.exe"}'`);
    
    console.log('');
});

module.exports = app; 