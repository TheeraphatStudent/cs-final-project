-- Initialize database for URL Checker API
USE url_checker;

-- Create url_container table if not exists
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
    INDEX idx_processed_at (processed_at),
    INDEX idx_is_malicious (is_malicious)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Create user for API access
CREATE USER IF NOT EXISTS 'api_user'@'%' IDENTIFIED BY 'api_password';
GRANT SELECT, INSERT, UPDATE, DELETE ON url_checker.* TO 'api_user'@'%';
FLUSH PRIVILEGES;

-- Insert sample data for testing
INSERT INTO url_container (url, model_used, prediction, confidence, reason, is_malicious) VALUES
('https://www.google.com', 'xg_boost', 'safe', 0.9500, 'Known safe domain', FALSE),
('https://www.facebook.com', 'xg_boost', 'safe', 0.9200, 'Trusted social media platform', FALSE),
('https://malicious-example.com', 'xg_boost', 'malicious', 0.8500, 'Suspicious URL patterns detected', TRUE),
('https://www.github.com', 'neural_network', 'safe', 0.9800, 'Developer platform - safe', FALSE),
('https://phishing-site-example.com', 'svm', 'malicious', 0.7800, 'Phishing indicators present', TRUE);

-- Create indexes for better performance
CREATE INDEX idx_confidence ON url_container(confidence);
CREATE INDEX idx_created_date ON url_container(DATE(created_at));

-- Show table structure
DESCRIBE url_container; 