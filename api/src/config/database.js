const mysql = require('mysql2/promise');
const logger = require('./logger');

class Database {
    constructor() {
        this.pool = null;
        this.init();
    }

    async init() {
        try {
            this.pool = mysql.createPool({
                host: process.env.MYSQL_HOST || 'localhost',
                port: process.env.MYSQL_PORT || 3306,
                user: process.env.MYSQL_USER || 'root',
                password: process.env.MYSQL_PASSWORD || 'password',
                database: process.env.MYSQL_DATABASE || 'url_checker',
                waitForConnections: true,
                connectionLimit: 10,
                queueLimit: 0,
                acquireTimeout: 60000,
                timeout: 60000
            });

            await this.createTables();
            logger.info('MySQL database connected successfully');
        } catch (error) {
            logger.error('MySQL connection failed:', error);
            throw error;
        }
    }

    async createTables() {
        const createUrlContainerTable = `
            CREATE TABLE IF NOT EXISTS url_container (
                id INT AUTO_INCREMENT PRIMARY KEY,
                url VARCHAR(2048) NOT NULL,
                model_used VARCHAR(50) NOT NULL,
                prediction VARCHAR(20) NOT NULL,
                confidence DECIMAL(5,4) NOT NULL,
                reason TEXT,
                is_malicious BOOLEAN NOT NULL,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_url (url(255)),
                INDEX idx_model (model_used),
                INDEX idx_prediction (prediction),
                INDEX idx_processed_at (processed_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        `;

        try {
            await this.pool.execute(createUrlContainerTable);
            logger.info('Database tables created successfully');
        } catch (error) {
            logger.error('Error creating tables:', error);
            throw error;
        }
    }

    async insertUrlData(urlData) {
        const query = `
            INSERT INTO url_container 
            (url, model_used, prediction, confidence, reason, is_malicious) 
            VALUES (?, ?, ?, ?, ?, ?)
        `;
        
        const values = [
            urlData.url,
            urlData.model_used,
            urlData.prediction,
            urlData.confidence,
            urlData.reason,
            urlData.is_malicious
        ];

        try {
            const [result] = await this.pool.execute(query, values);
            return result;
        } catch (error) {
            logger.error('Error inserting URL data:', error);
            throw error;
        }
    }

    async insertBatch(urlDataArray) {
        const query = `
            INSERT INTO url_container 
            (url, model_used, prediction, confidence, reason, is_malicious) 
            VALUES ?
        `;

        const values = urlDataArray.map(data => [
            data.url,
            data.model_used,
            data.prediction,
            data.confidence,
            data.reason,
            data.is_malicious
        ]);

        try {
            const [result] = await this.pool.query(query, [values]);
            logger.info(`Batch inserted ${result.affectedRows} records`);
            return result;
        } catch (error) {
            logger.error('Error inserting batch data:', error);
            throw error;
        }
    }

    async getStats() {
        const query = `
            SELECT 
                COUNT(*) as total_urls,
                SUM(CASE WHEN is_malicious = 1 THEN 1 ELSE 0 END) as malicious_count,
                SUM(CASE WHEN is_malicious = 0 THEN 1 ELSE 0 END) as safe_count,
                model_used,
                AVG(confidence) as avg_confidence
            FROM url_container 
            GROUP BY model_used
        `;

        try {
            const [rows] = await this.pool.execute(query);
            return rows;
        } catch (error) {
            logger.error('Error getting stats:', error);
            throw error;
        }
    }

    async close() {
        if (this.pool) {
            await this.pool.end();
            logger.info('MySQL connection pool closed');
        }
    }
}

module.exports = new Database(); 