const redis = require('redis');
const logger = require('./logger');

class RedisClient {
    constructor() {
        this.client = null;
        this.publisher = null;
        this.subscriber = null;
        this.init();
    }

    async init() {
        try {
            const config = {
                socket: {
                    host: process.env.REDIS_HOST || 'localhost',
                    port: process.env.REDIS_PORT || 6379,
                    reconnectStrategy: (retries) => Math.min(retries * 50, 500)
                }
            };

            if (process.env.REDIS_PASSWORD) {
                config.password = process.env.REDIS_PASSWORD;
            }

            this.client = redis.createClient(config);
            this.publisher = redis.createClient(config);
            this.subscriber = redis.createClient(config);

            this.client.on('error', (err) => logger.error('Redis Client Error:', err));
            this.publisher.on('error', (err) => logger.error('Redis Publisher Error:', err));
            this.subscriber.on('error', (err) => logger.error('Redis Subscriber Error:', err));

            await this.client.connect();
            await this.publisher.connect();
            await this.subscriber.connect();

            logger.info('Redis connected successfully');
        } catch (error) {
            logger.error('Redis connection failed:', error);
            throw error;
        }
    }

    async addToQueue(queueName, data) {
        try {
            const serializedData = JSON.stringify({
                ...data,
                timestamp: new Date().toISOString(),
                id: require('uuid').v4()
            });
            
            await this.client.lPush(queueName, serializedData);
            const queueLength = await this.client.lLen(queueName);
            
            logger.info(`Added item to queue ${queueName}. Queue length: ${queueLength}`);
            
            if (queueLength >= (process.env.QUEUE_BATCH_SIZE || 1000)) {
                await this.publisher.publish('batch_ready', queueName);
                logger.info(`Batch ready signal sent for queue: ${queueName}`);
            }
            
            return queueLength;
        } catch (error) {
            logger.error('Error adding to queue:', error);
            throw error;
        }
    }

    async processBatch(queueName, batchSize = null) {
        const size = batchSize || process.env.QUEUE_BATCH_SIZE || 1000;
        
        try {
            const items = await this.client.lRange(queueName, 0, size - 1);
            
            if (items.length === 0) {
                return [];
            }

            await this.client.lTrim(queueName, items.length, -1);
            
            const processedItems = items.map(item => {
                try {
                    return JSON.parse(item);
                } catch (error) {
                    logger.error('Error parsing queue item:', error);
                    return null;
                }
            }).filter(item => item !== null);

            logger.info(`Processed batch of ${processedItems.length} items from queue ${queueName}`);
            return processedItems;
        } catch (error) {
            logger.error('Error processing batch:', error);
            throw error;
        }
    }

    async getQueueLength(queueName) {
        try {
            return await this.client.lLen(queueName);
        } catch (error) {
            logger.error('Error getting queue length:', error);
            throw error;
        }
    }

    async clearQueue(queueName) {
        try {
            await this.client.del(queueName);
            logger.info(`Queue ${queueName} cleared`);
        } catch (error) {
            logger.error('Error clearing queue:', error);
            throw error;
        }
    }

    async setCache(key, value, ttl = 3600) {
        try {
            await this.client.setEx(key, ttl, JSON.stringify(value));
        } catch (error) {
            logger.error('Error setting cache:', error);
            throw error;
        }
    }

    async getCache(key) {
        try {
            const value = await this.client.get(key);
            return value ? JSON.parse(value) : null;
        } catch (error) {
            logger.error('Error getting cache:', error);
            return null;
        }
    }

    async subscribeToBatchReady(callback) {
        try {
            await this.subscriber.subscribe('batch_ready', callback);
            logger.info('Subscribed to batch_ready channel');
        } catch (error) {
            logger.error('Error subscribing to batch_ready:', error);
            throw error;
        }
    }

    async close() {
        try {
            if (this.client) await this.client.disconnect();
            if (this.publisher) await this.publisher.disconnect();
            if (this.subscriber) await this.subscriber.disconnect();
            logger.info('Redis connections closed');
        } catch (error) {
            logger.error('Error closing Redis connections:', error);
        }
    }
}

module.exports = new RedisClient(); 