const redisClient = require('../config/redis');
const database = require('../config/database');
const aiService = require('./aiService');
const logger = require('../config/logger');

class QueueService {
    constructor() {
        this.isProcessing = false;
        this.queueName = 'url_prediction_queue';
        this.init();
    }

    async init() {
        try {
            await redisClient.subscribeToBatchReady(this.handleBatchReady.bind(this));
            logger.info('Queue service initialized successfully');
        } catch (error) {
            logger.error('Error initializing queue service:', error);
            throw error;
        }
    }

    async addUrlToQueue(url, model = 'xg_boost') {
        try {
            const queueData = {
                url: url,
                model: model,
                status: 'pending',
                created_at: new Date().toISOString()
            };

            const queueLength = await redisClient.addToQueue(this.queueName, queueData);
            logger.info(`URL added to queue: ${url} (Queue length: ${queueLength})`);
            
            return {
                success: true,
                queueLength: queueLength,
                message: 'URL added to processing queue'
            };
        } catch (error) {
            logger.error('Error adding URL to queue:', error);
            throw error;
        }
    }

    async handleBatchReady(channel, queueName) {
        if (this.isProcessing) {
            logger.info(`Batch processing already in progress for queue: ${queueName}`);
            return;
        }

        this.isProcessing = true;
        logger.info(`Processing batch for queue: ${queueName}`);

        try {
            await this.processBatch(queueName);
        } catch (error) {
            logger.error('Error processing batch:', error);
        } finally {
            this.isProcessing = false;
        }
    }

    async processBatch(queueName = null) {
        const targetQueue = queueName || this.queueName;
        
        try {
            const queueItems = await redisClient.processBatch(targetQueue);
            
            if (queueItems.length === 0) {
                logger.info('No items to process in queue');
                return { processed: 0 };
            }

            logger.info(`Processing ${queueItems.length} items from queue`);

            const predictions = await aiService.batchPredict(queueItems);
            
            const validPredictions = predictions.filter(pred => pred && pred.url);
            
            if (validPredictions.length > 0) {
                await database.insertBatch(validPredictions);
                logger.info(`Successfully inserted ${validPredictions.length} predictions into database`);
            }

            return {
                processed: queueItems.length,
                successful: validPredictions.length,
                failed: queueItems.length - validPredictions.length
            };

        } catch (error) {
            logger.error('Error in batch processing:', error);
            throw error;
        }
    }

    async forceProcessQueue() {
        if (this.isProcessing) {
            return { message: 'Batch processing already in progress' };
        }

        this.isProcessing = true;
        
        try {
            const result = await this.processBatch();
            return {
                success: true,
                ...result,
                message: 'Forced queue processing completed'
            };
        } catch (error) {
            logger.error('Error in forced queue processing:', error);
            throw error;
        } finally {
            this.isProcessing = false;
        }
    }

    async getQueueStatus() {
        try {
            const queueLength = await redisClient.getQueueLength(this.queueName);
            const dbStats = await database.getStats();
            
            return {
                queue_length: queueLength,
                is_processing: this.isProcessing,
                database_stats: dbStats,
                batch_size: process.env.QUEUE_BATCH_SIZE || 1000
            };
        } catch (error) {
            logger.error('Error getting queue status:', error);
            throw error;
        }
    }

    async clearQueue() {
        try {
            await redisClient.clearQueue(this.queueName);
            return { success: true, message: 'Queue cleared successfully' };
        } catch (error) {
            logger.error('Error clearing queue:', error);
            throw error;
        }
    }

    async schedulePeriodicProcessing(intervalMinutes = 5) {
        setInterval(async () => {
            try {
                const queueLength = await redisClient.getQueueLength(this.queueName);
                
                if (queueLength > 0 && !this.isProcessing) {
                    logger.info(`Periodic processing: ${queueLength} items in queue`);
                    await this.processBatch();
                }
            } catch (error) {
                logger.error('Error in periodic processing:', error);
            }
        }, intervalMinutes * 60 * 1000);

        logger.info(`Scheduled periodic queue processing every ${intervalMinutes} minutes`);
    }

    async processSpecificUrls(urls, model = 'xg_boost') {
        try {
            logger.info(`Processing ${urls.length} specific URLs with model: ${model}`);
            
            const urlData = urls.map(url => ({ url, model }));
            const predictions = await aiService.batchPredict(urlData, model);
            
            const validPredictions = predictions.filter(pred => pred && pred.url);
            
            if (validPredictions.length > 0) {
                await database.insertBatch(validPredictions);
            }

            return {
                processed: urls.length,
                successful: validPredictions.length,
                failed: urls.length - validPredictions.length,
                predictions: validPredictions
            };

        } catch (error) {
            logger.error('Error processing specific URLs:', error);
            throw error;
        }
    }
}

module.exports = new QueueService(); 