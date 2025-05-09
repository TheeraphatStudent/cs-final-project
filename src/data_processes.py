import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
from datetime import datetime
import re

class DataProcessor:
    def __init__(self, input_path, output_path):
        """
        Initialize the DataProcessor
        
        Args:
            input_path (str): Path to input Excel file
            output_path (str): Path to save processed CSV file
        """
        self.input_path = input_path
        self.output_path = output_path
        self.label_encoder = LabelEncoder()
        
    def extract_url_components(self, url):
        """
        Extract components from URL
        
        Args:
            url (str): URL to process
            
        Returns:
            dict: Dictionary containing URL components
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
        
        Args:
            url (str): URL to analyze
            
        Returns:
            dict: Dictionary containing URL metrics
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
        """
        Check for suspicious patterns in URL
        
        Args:
            url (str): URL to check
            
        Returns:
            dict: Dictionary containing suspicious pattern flags
        """
        if pd.isna(url):
            return {
                'has_suspicious_extension': False,
                'has_suspicious_keyword': False,
                'is_legitimate_domain': False
            }
            
        url = str(url).lower()
        
        # Check for suspicious extensions
        suspicious_extensions = ['.exe', '.zip', '.rar', '.pdf', '.doc', '.docx']
        has_suspicious_extension = any(ext in url for ext in suspicious_extensions)
        
        # Check for suspicious keywords
        suspicious_keywords = ['login', 'signin', 'account', 'bank', 'secure', 'verify']
        has_suspicious_keyword = any(keyword in url for keyword in suspicious_keywords)
        
        # Check for legitimate domains (simplified)
        legitimate_domains = ['google', 'facebook', 'amazon', 'microsoft', 'apple']
        is_legitimate_domain = any(domain in url for domain in legitimate_domains)
        
        return {
            'has_suspicious_extension': has_suspicious_extension,
            'has_suspicious_keyword': has_suspicious_keyword,
            'is_legitimate_domain': is_legitimate_domain
        }
        
    def process_data(self):
        """
        Process data from Excel file to CSV
        
        Returns:
            pd.DataFrame: Processed DataFrame or None if error
        """
        try:
            print("Loading data from Excel...")
            # Read Excel file with explicit engine
            df = pd.read_excel(self.input_path, engine='openpyxl')
            
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
            
            # Ensure all required columns are present
            required_columns = [
                'id', 'dateadded', 'url', 'url_status', 'threat', 'tags',
                'urlhaus_link', 'subdomain', 'domain', 'suffix', 'is_private_domain',
                'path', 'query', 'fragment', 'url_length', 'path_length',
                'query_length', 'num_dots', 'num_slashes', 'num_digits',
                'num_special_chars', 'has_suspicious_extension',
                'has_suspicious_keyword', 'is_legitimate_domain',
                'has_https', 'has_http'
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
            
            return df
            
        except Exception as e:
            print(f"Error processing data: {str(e)}")
            return None
            
    def get_data_summary(self):
        """
        Get summary statistics of the processed data
        
        Returns:
            dict: Dictionary containing data statistics
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
