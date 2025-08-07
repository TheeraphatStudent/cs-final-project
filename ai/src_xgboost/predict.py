#!/usr/bin/env python3

import sys
import json
import argparse
from model import URLClassifier
import warnings
warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser(description='Predict URL maliciousness using XGBoost model')
    parser.add_argument('--url', required=True, help='URL to predict')
    parser.add_argument('--format', choices=['json', 'text'], default='json', help='Output format')
    
    args = parser.parse_args()
    
    try:
        classifier = URLClassifier()
        classifier.load_model()
        
        prediction, confidence, reason = classifier.predict(args.url)
        
        if args.format == 'json':
            result = {
                'url': args.url,
                'prediction': prediction,
                'confidence': float(confidence),
                'reason': reason,
                'model': 'xg_boost'
            }
            print(json.dumps(result))
        else:
            print(f"URL: {args.url}")
            print(f"Prediction: {prediction}")
            print(f"Confidence: {confidence:.4f}")
            print(f"Reason: {reason}")
            
    except Exception as e:
        error_result = {
            'error': str(e),
            'url': args.url,
            'prediction': 'error',
            'confidence': 0.0,
            'reason': f'Error: {str(e)}',
            'model': 'xg_boost'
        }
        if args.format == 'json':
            print(json.dumps(error_result))
        else:
            print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 