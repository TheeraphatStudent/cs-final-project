import pandas as pd
import numpy as np
import os
import re
from urllib.parse import urlparse
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class URLDataProcessor:
    """Process and clean URL datasets for training"""
    
    def __init__(self):
        self.suspicious_extensions = [
            '.exe', '.bat', '.cmd', '.sh', '.msi', '.dll', '.vbs', '.ps1', 
            '.js', '.jar', '.zip', '.rar', '.7z', '.tar', '.gz', '.iso'
        ]
        
        self.suspicious_keywords = [
            'login', 'signin', 'verify', 'account', 'secure', 'update', 'install',
            'password', 'bank', 'credit', 'card', 'pay', 'money', 'wallet', 'crypto',
            'bitcoin', 'wallet', 'phish', 'hack', 'exploit', 'malware', 'virus',
            'trojan', 'ransomware', 'spyware', 'adware', 'keylogger', 'botnet'
        ]
        
        self.legitimate_domains = [
            'google.com', 'microsoft.com', 'apple.com', 'amazon.com', 'facebook.com',
            'twitter.com', 'instagram.com', 'linkedin.com', 'github.com', 'wikipedia.org',
            'youtube.com', 'netflix.com', 'spotify.com', 'reddit.com', 'pinterest.com',
            'tiktok.com', 'snapchat.com', 'whatsapp.com', 'telegram.org', 'discord.com'
        ]
        
        self.common_tlds = [
            'com', 'org', 'net', 'edu', 'gov', 'mil', 'int', 'io', 'co', 'uk',
            'us', 'ca', 'au', 'de', 'fr', 'jp', 'cn', 'ru', 'br', 'in'
        ]
    
    def extract_domain_components(self, url):
        """Extract domain components from URL without using tldextract"""
        if pd.isna(url):
            return {
                'subdomain': '',
                'domain': '',
                'suffix': '',
                'is_private': False
            }
        
        parsed = urlparse(url)
        hostname = parsed.netloc
        
        if ':' in hostname:
            hostname = hostname.split(':')[0]
        
        parts = hostname.split('.')
        
        if not parts or parts[0] == '':
            return {
                'subdomain': '',
                'domain': '',
                'suffix': '',
                'is_private': False
            }
        
        # Single part -> localhost
        if len(parts) == 1:
            return {
                'subdomain': '',
                'domain': parts[0],
                'suffix': '',
                'is_private': False
            }
        
        # Two part -> example.com
        if len(parts) == 2:
            if parts[1].lower() in self.common_tlds:
                return {
                    'subdomain': '',
                    'domain': parts[0],
                    'suffix': parts[1],
                    'is_private': False
                }
            else:
                return {
                    'subdomain': '',
                    'domain': parts[0],
                    'suffix': parts[1],
                    'is_private': True
                }
        
        # Three part -> foo.baa.com
        if parts[-2] + '.' + parts[-1] in [tld + '.uk' for tld in ['co', 'org', 'net', 'me', 'ltd', 'plc']]:
            return {
                'subdomain': '.'.join(parts[:-2]),
                'domain': parts[-2],
                'suffix': parts[-1],
                'is_private': False
            }
        
        if parts[-1].lower() in self.common_tlds:
            return {
                'subdomain': '.'.join(parts[:-2]),
                'domain': parts[-2],
                'suffix': parts[-1],
                'is_private': False
            }
        
        return {
            'subdomain': '.'.join(parts[:-2]),
            'domain': parts[-2],
            'suffix': parts[-1],
            'is_private': True
        }
    
    def load_dataset(self, file_path):
        """Load dataset from various file formats"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        logger.info(f"Loading dataset from: {file_path}")
        
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.txt'):
            # Try to detect delimiter
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
                if ',' in first_line:
                    df = pd.read_csv(file_path)
                elif '\t' in first_line:
                    df = pd.read_csv(file_path, sep='\t')
                else:
                    df = pd.read_csv(file_path, sep=' ')
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        logger.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        logger.info(f"Columns: {', '.join(df.columns)}")
        
        return df
    
    def clean_dataset(self, df):
        """Clean and preprocess the dataset"""
        logger.info("Cleaning dataset...")
        
        df = df.copy()
        
        initial_rows = len(df)
        df = df.drop_duplicates()
        if len(df) < initial_rows:
            logger.info(f"Removed {initial_rows - len(df)} duplicate rows")
        
        df = df.dropna(subset=['url'])
        logger.info(f"Dataset now has {len(df)} rows after removing missing URLs")
        
        df['url'] = df['url'].apply(self._standardize_url)
        
        domain_info = df['url'].apply(self.extract_domain_components)
        df['subdomain'] = domain_info.apply(lambda x: x['subdomain'])
        df['domain'] = domain_info.apply(lambda x: x['domain'])
        df['suffix'] = domain_info.apply(lambda x: x['suffix'])
        df['is_private_domain'] = domain_info.apply(lambda x: x['is_private'])
        df['path'] = df['url'].apply(lambda x: urlparse(x).path)
        df['query'] = df['url'].apply(lambda x: urlparse(x).query)
        df['fragment'] = df['url'].apply(lambda x: urlparse(x).fragment)
        df['url_length'] = df['url'].apply(len)
        df['path_length'] = df['path'].apply(len)
        df['query_length'] = df['query'].apply(len)
        df['num_dots'] = df['url'].apply(lambda x: x.count('.'))
        df['num_slashes'] = df['url'].apply(lambda x: x.count('/'))
        df['num_digits'] = df['url'].apply(lambda x: len(re.findall(r'\d', x)))
        df['num_special_chars'] = df['url'].apply(lambda x: len(re.findall(r'[!@#$%^&*(),.?":{}|<>]', x)))
        
        # ! Check suspicious
        df['has_suspicious_extension'] = df['url'].apply(
            lambda x: 1 if any(ext in x.lower() for ext in self.suspicious_extensions) else 0
        )
        
        df['has_suspicious_keyword'] = df['url'].apply(
            lambda x: 1 if any(keyword in x.lower() for keyword in self.suspicious_keywords) else 0
        )
        
        df['is_legitimate_domain'] = df['domain'].apply(
            lambda x: 1 if any(domain.split('.')[0] in x.lower() for domain in self.legitimate_domains) else 0
        )
        
        # ! Protocol
        df['has_https'] = df['url'].apply(lambda x: 1 if 'https' in x else 0)
        df['has_http'] = df['url'].apply(lambda x: 1 if 'http' in x and 'https' not in x else 0)
        
        if 'threat' in df.columns:
            df['label'] = df['threat'].apply(lambda x: 'dangerous' if x != 'none' and pd.notna(x) else 'safe')
        elif 'url_status' in df.columns:
            df['label'] = df['url_status'].apply(lambda x: 'dangerous' if x == 'offline' or x == 'suspicious' else 'safe')
        else:
            df['label'] = df.apply(self._heuristic_label, axis=1)
        
        df['processed_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        logger.info("Dataset cleaning completed")
        return df
    
    def _standardize_url(self, url):
        """Standardize URL format"""
        if pd.isna(url):
            return url
        
        url = str(url).strip()
        
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        
        return url
    
    def _heuristic_label(self, row):
        """Apply heuristic rules to determine if a URL is safe or dangerous"""
        score = 0
        
        if row['has_suspicious_extension']:
            score += 2
        if row['has_suspicious_keyword']:
            score += 2
        if row['num_special_chars'] > 5:
            score += 1
        if row['num_digits'] > 10:
            score += 1
        if row['url_length'] > 100:
            score += 1
        if not row['has_https']:
            score += 1
        if row['is_private_domain']:
            score += 1
        
        if row['is_legitimate_domain']:
            score -= 3
        if row['has_https']:
            score -= 1
        
        return 'dangerous' if score > 2 else 'safe'
    
    def prepare_training_data(self, df):
        """Prepare data for model training"""
        logger.info("Preparing training data...")
        
        urls = df['url'].tolist()
        labels = df['label'].tolist()
        
        label_counts = df['label'].value_counts()
        logger.info(f"Class distribution: {label_counts.to_dict()}")
        
        if len(label_counts) == 2:
            imbalance_ratio = label_counts.max() / label_counts.min()
            logger.info(f"Class imbalance ratio: {imbalance_ratio:.2f}")
            
            if imbalance_ratio > 2:
                logger.warning("Significant class imbalance detected. Consider using class weights during training.")
        
        return urls, labels
    
    def save_processed_dataset(self, df, output_path):
        """Save processed dataset to file"""
        logger.info(f"Saving processed dataset to: {output_path}")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} rows to {output_path}")
        
        return output_path