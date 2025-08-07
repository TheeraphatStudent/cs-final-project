import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
from datetime import datetime
import re

class DataProcessor:
    """
    Data Processor for URL Classification
    
    
    - Handles data loading and preprocessing from Excel files
    - Implements URL feature extraction and analysis
    - Performs data normalization and transformation
    - Manages data persistence and storage
    """

    def __init__(self, input_path, output_path):
        """
        Initialize the DataProcessor
        

        - Sets up input/output paths for data processing pipeline
        - Initializes LabelEncoder for categorical feature encoding
        - Prepares for feature extraction and transformation
        """
        self.input_path = input_path
        self.output_path = output_path
        self.label_encoder = LabelEncoder()
        
    def extract_url_components(self, url):
        """
        Extract components from URL
        
        
        1. URL Parsing:
           - Protocol detection (HTTP/HTTPS)
           - Domain structure analysis
           - Path and query parameter extraction
        2. Component Extraction:
           - Subdomain identification
           - Domain name extraction
           - TLD (Top Level Domain) parsing
           - Path and query string separation
        3. Feature Vector Creation:
           - Combine all components into structured format
           - Handle missing or invalid URLs
        """
        if pd.isna(url):
            return {
                'subdomain': '',
                'domain': '',
                'suffix': '',
                'path': '',
                'query': '',
                'fragment': ''
            }
            
        # Convert to string and lowercase
        url = str(url).lower()
        
        # Extract protocol
        has_https = 'https://' in url
        has_http = 'http://' in url
        
        # Remove protocol
        url = re.sub(r'^https?://', '', url)
        
        # Split into parts
        parts = url.split('/', 1)
        domain_part = parts[0]
        path_part = parts[1] if len(parts) > 1 else ''
        
        # Split domain into components
        domain_parts = domain_part.split('.')
        if len(domain_parts) >= 2:
            suffix = domain_parts[-1]
            domain = domain_parts[-2]
            subdomain = '.'.join(domain_parts[:-2]) if len(domain_parts) > 2 else ''
        else:
            suffix = ''
            domain = domain_part
            subdomain = ''
            
        # Split path into components
        path_parts = path_part.split('?', 1)
        path = path_parts[0]
        query = path_parts[1].split('#')[0] if len(path_parts) > 1 else ''
        fragment = path_parts[1].split('#')[1] if len(path_parts) > 1 and '#' in path_parts[1] else ''
        
        return {
            'subdomain': subdomain,
            'domain': domain,
            'suffix': suffix,
            'path': path,
            'query': query,
            'fragment': fragment,
            'has_https': has_https,
            'has_http': has_http
        }
        
    def calculate_url_metrics(self, url):
        """
        Calculate metrics for URL
        
        
        1. Length Analysis:
           - Total URL length
           - Path length
           - Query string length
        2. Character Analysis:
           - Dot count
           - Slash count
           - Digit count
           - Special character count
        3. Statistical Features:
           - Character distribution
           - Pattern frequency
        """
        if pd.isna(url):
            return {
                'url_length': 0,
                'path_length': 0,
                'query_length': 0,
                'num_dots': 0,
                'num_slashes': 0,
                'num_digits': 0,
                'num_special_chars': 0
            }
            
        url = str(url)
        components = self.extract_url_components(url)
        
        # Calculate metrics
        url_length = len(url)
        path_length = len(components['path'])
        query_length = len(components['query'])
        num_dots = url.count('.')
        num_slashes = url.count('/')
        num_digits = sum(c.isdigit() for c in url)
        num_special_chars = len(re.findall(r'[^a-zA-Z0-9./]', url))
        
        return {
            'url_length': url_length,
            'path_length': path_length,
            'query_length': query_length,
            'num_dots': num_dots,
            'num_slashes': num_slashes,
            'num_digits': num_digits,
            'num_special_chars': num_special_chars
        }
        
    def check_suspicious_patterns(self, url):
        if pd.isna(url):
            return {
                'has_suspicious_extension': False,
                'has_suspicious_keyword': False,
                'is_legitimate_domain': False
            }
            
        url = str(url).lower()
        
        highly_suspicious_extensions = [
            '.exe', '.bat', '.cmd', '.scr', '.pif', '.vbs', '.js', '.dll', '.sys', '.drv',
            '.hint', '.tmp', '.dat', '.bin', '.msi', '.torrent', '.lnk', '.reg'
        ]
        
        potentially_suspicious_extensions = [
            '.zip', '.rar', '.jar', '.apk', '.dmg', '.deb', '.rpm', '.pdf', '.doc', '.docx'
        ]
        
        # Parse URL to get path only (not domain)
        from urllib.parse import urlparse
        has_suspicious_extension = False
        
        try:
            parsed = urlparse(url if url.startswith(('http://', 'https://')) else 'http://' + url)
            path = parsed.path.lower()
            
            # Check for highly suspicious extensions in path (normal case)
            has_path_suspicious = any(path.endswith(ext) for ext in highly_suspicious_extensions)
            
            # SPECIAL CASE: Check if domain suffix looks like a file extension
            # This handles cases like www.google.exe where .exe is parsed as TLD
            domain_parts = parsed.netloc.split('.')
            if len(domain_parts) >= 2:
                suffix = '.' + domain_parts[-1]
                has_domain_suspicious = suffix in highly_suspicious_extensions
            else:
                has_domain_suspicious = False
            
            # Combine both checks
            has_suspicious_extension = has_path_suspicious or has_domain_suspicious
            
        except:
            # Fallback to simple check if parsing fails
            has_suspicious_extension = any(url.endswith(ext) for ext in highly_suspicious_extensions)
            
        # Check for suspicious keywords (expanded list)
        suspicious_keywords = [
            'login', 'signin', 'account', 'bank', 'secure', 'verify', 'update',
            'confirm', 'password', 'credit', 'card', 'paypal', 'ebay',
            'admin', 'root', 'download', 'install'
        ]
        has_suspicious_keyword = any(keyword in url for keyword in suspicious_keywords)
        
        # Check for legitimate domains (expanded list)
        legitimate_domains = ['google', 'facebook', 'amazon', 'microsoft', 'apple', 'github', 'stackoverflow', 'wikipedia']
        is_legitimate_domain = any(domain in url for domain in legitimate_domains)
        
        return {
            'has_suspicious_extension': has_suspicious_extension,
            'has_suspicious_keyword': has_suspicious_keyword,
            'is_legitimate_domain': is_legitimate_domain
        }
        
    def process_data(self):
        try:
            print("Loading data from CSV...")
            # Read CSV file directly (new format)
            df = pd.read_csv(self.input_path)
            
            # Verify required columns exist
            required_input_cols = ['no', 'url', 'type', 'isMalicious']
            missing_cols = [col for col in required_input_cols if col not in df.columns]
            if missing_cols:
                print(f"Missing required columns: {missing_cols}")
                return None
            
            print(f"Loaded {len(df)} records from dataset")
            print("Processing URLs...")
            
            # Extract URL components
            url_components = df['url'].apply(self.extract_url_components)
            url_components_df = pd.DataFrame(url_components.tolist())
            
            # Calculate URL metrics
            url_metrics = df['url'].apply(self.calculate_url_metrics)
            url_metrics_df = pd.DataFrame(url_metrics.tolist())
            
            # Check suspicious patterns
            suspicious_patterns = df['url'].apply(self.check_suspicious_patterns)
            suspicious_patterns_df = pd.DataFrame(suspicious_patterns.tolist())
            
            # Combine all features
            df = pd.concat([
                df,
                url_components_df,
                url_metrics_df,
                suspicious_patterns_df
            ], axis=1)
            
            # Ensure all required columns are present for model training
            required_columns = [
                'no', 'url', 'type', 'isMalicious',
                'subdomain', 'domain', 'suffix', 'is_private_domain', 'path', 'query', 'fragment',
                'url_length', 'path_length', 'query_length', 'num_dots', 'num_slashes', 
                'num_digits', 'num_special_chars', 'has_suspicious_extension', 
                'has_suspicious_keyword', 'is_legitimate_domain', 'has_https', 'has_http'
            ]
            
            # Add missing columns with default values
            for col in required_columns:
                if col not in df.columns:
                    df[col] = None
            
            # Reorder columns
            df = df[required_columns]
            
            # Save to CSV
            print(f"Saving processed data to {self.output_path}")
            df.to_csv(self.output_path, index=False)
            
            # Print dataset statistics
            print("\nDataset Statistics:")
            print(f"Total samples: {len(df)}")
            print(f"Malicious URLs: {df['isMalicious'].sum()}")
            print(f"Safe URLs: {len(df) - df['isMalicious'].sum()}")
            print(f"URL types: {df['type'].value_counts().to_dict()}")
            
            return df
            
        except Exception as e:
            print(f"Error processing data: {str(e)}")
            return None
            
    def get_data_summary(self):
        """
        Get summary statistics of the processed data
        
        
        1. Data Analysis:
           - Calculate total samples
           - Count unique domains
           - Analyze suspicious patterns
        2. Statistical Summary:
           - Generate descriptive statistics
           - Calculate pattern frequencies
        """
        try:
            df = pd.read_csv(self.output_path)
            return {
                'total_samples': len(df),
                'unique_domains': df['domain'].nunique(),
                'suspicious_extensions': df['has_suspicious_extension'].sum(),
                'suspicious_keywords': df['has_suspicious_keyword'].sum(),
                'legitimate_domains': df['is_legitimate_domain'].sum()
            }
        except Exception as e:
            print(f"Error getting data summary: {str(e)}")
            return None
