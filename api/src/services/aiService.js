const { spawn } = require('child_process');
const path = require('path');
const logger = require('../config/logger');

class AIService {
    constructor() {
        this.modelPaths = {
            'xg_boost': 'src_xgboost',
            'neural_network': 'src',
            'svm': 'src_svm'
        };
        this.aiBasePath = process.env.AI_MODELS_PATH || path.join(__dirname, '../../../ai');
    }

    async predictURL(url, modelType = 'xg_boost') {
        if (!this.modelPaths[modelType]) {
            throw new Error(`Invalid model type: ${modelType}. Available models: ${Object.keys(this.modelPaths).join(', ')}`);
        }

        const modelPath = path.join(this.aiBasePath, this.modelPaths[modelType]);
        const scriptPath = path.join(modelPath, 'predict.py');

        return new Promise((resolve, reject) => {
            const pythonProcess = spawn('python3', [
                scriptPath,
                '--url', url,
                '--format', 'json'
            ], {
                cwd: modelPath,
                env: {
                    ...process.env,
                    PYTHONPATH: modelPath
                }
            });

            let stdout = '';
            let stderr = '';

            pythonProcess.stdout.on('data', (data) => {
                stdout += data.toString();
            });

            pythonProcess.stderr.on('data', (data) => {
                stderr += data.toString();
            });

            pythonProcess.on('close', (code) => {
                if (code !== 0) {
                    logger.error(`Python process exited with code ${code}`, { stderr, stdout });
                    reject(new Error(`AI prediction failed: ${stderr || 'Unknown error'}`));
                    return;
                }

                try {
                    const output = JSON.parse(stdout.trim());
                    const result = {
                        url: url,
                        model_used: modelType,
                        prediction: output.prediction || output.result,
                        confidence: parseFloat(output.confidence || 0),
                        reason: output.reason || output.explanation || 'No reason provided',
                        is_malicious: this.isMaliciousResult(output.prediction || output.result),
                        timestamp: new Date().toISOString()
                    };

                    logger.info(`AI prediction completed for URL: ${url}`, { 
                        model: modelType, 
                        prediction: result.prediction,
                        confidence: result.confidence 
                    });

                    resolve(result);
                } catch (parseError) {
                    logger.error('Error parsing AI response:', { parseError, stdout, stderr });
                    reject(new Error(`Failed to parse AI response: ${parseError.message}`));
                }
            });

            pythonProcess.on('error', (error) => {
                logger.error('Python process error:', error);
                reject(new Error(`Failed to start AI prediction: ${error.message}`));
            });

            setTimeout(() => {
                pythonProcess.kill();
                reject(new Error('AI prediction timeout'));
            }, 30000);
        });
    }

    isMaliciousResult(prediction) {
        if (typeof prediction === 'boolean') {
            return prediction;
        }
        
        if (typeof prediction === 'string') {
            return prediction.toLowerCase() === 'malicious' || prediction === '1';
        }
        
        if (typeof prediction === 'number') {
            return prediction > 0.5;
        }
        
        return false;
    }

    async batchPredict(urls, modelType = 'xg_boost') {
        const results = [];
        const batchSize = 50;

        for (let i = 0; i < urls.length; i += batchSize) {
            const batch = urls.slice(i, i + batchSize);
            const batchPromises = batch.map(urlData => 
                this.predictURL(urlData.url, urlData.model || modelType)
                    .catch(error => {
                        logger.error(`Batch prediction failed for URL: ${urlData.url}`, error);
                        return {
                            url: urlData.url,
                            model_used: modelType,
                            prediction: 'error',
                            confidence: 0,
                            reason: `Prediction failed: ${error.message}`,
                            is_malicious: false,
                            timestamp: new Date().toISOString()
                        };
                    })
            );

            const batchResults = await Promise.all(batchPromises);
            results.push(...batchResults);

            logger.info(`Processed batch ${Math.floor(i/batchSize) + 1}/${Math.ceil(urls.length/batchSize)}`);
        }

        return results;
    }

    getAvailableModels() {
        return Object.keys(this.modelPaths);
    }

    async healthCheck() {
        try {
            const testUrl = 'https://www.google.com';
            const result = await this.predictURL(testUrl, 'xg_boost');
            return {
                status: 'healthy',
                models: this.getAvailableModels(),
                test_prediction: result
            };
        } catch (error) {
            logger.error('AI service health check failed:', error);
            return {
                status: 'unhealthy',
                error: error.message,
                models: this.getAvailableModels()
            };
        }
    }
}

module.exports = new AIService(); 