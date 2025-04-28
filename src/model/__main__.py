from url_safety_model import URLSafetyModel
from data_processor import URLDataProcessor
import argparse
import pandas as pd
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(params=None):
    """
    Main function to run the URL Safety Classifier.
    
    Args:
        params (dict): Dictionary containing parameters:
            - mode (str): 'train', 'predict', or 'process'
            - dataset (str): Path to the dataset file (CSV or Excel)
            - url (str): URL to check (required for predict mode)
            - model_path (str): Path to save/load the model
            - output_dataset (str): Path to save the processed dataset (for process mode)
    """

    default_params = {
        'mode': 'process',
        'dataset': 'src/dataset/dataset_1.xlsx',
        'url': None,
        'model_path': 'url_safety_model.keras',
        'output_dataset': 'src/dataset/processed_dataset.csv'
    }
    
    if params:
        default_params.update(params)
    
    if default_params['mode'] not in ['train', 'predict', 'process']:
        logger.error(f"Invalid mode: {default_params['mode']}. Must be 'train', 'predict', or 'process'")
        return
    
    url_model = URLSafetyModel()
    data_processor = URLDataProcessor()

    if default_params['mode'] == 'process':
        # Process dataset only
        try:
            if not os.path.exists(default_params['dataset']):
                logger.error(f"Error: Dataset file not found at {default_params['dataset']}")
                return

            df = data_processor.load_dataset(default_params['dataset'])
            df_cleaned = data_processor.clean_dataset(df)

            output_path = data_processor.save_processed_dataset(df_cleaned, default_params['output_dataset'])
            
            logger.info(f"Dataset processing completed. Saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error during dataset processing: {str(e)}")
            return

    elif default_params['mode'] == 'train':
        if not os.path.exists(default_params['dataset']):
            logger.error(f"Error: Dataset file not found at {default_params['dataset']}")
            return

        try:
            df = data_processor.load_dataset(default_params['dataset'])
            df_cleaned = data_processor.clean_dataset(df)
            
            urls, labels = data_processor.prepare_training_data(df_cleaned)
            
            logger.info(f"Loaded dataset with {len(urls)} URLs")
            logger.info(f"Training model...")
            
            history = url_model.train(urls, labels, epochs=20)
            
            url_model.save_model(default_params['model_path'])
            
            logger.info(f"History: {history}")
            logger.info(f"Model saved to: {default_params['model_path']}")
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            return

    elif default_params['mode'] == 'predict':
        if not default_params['url']:
            logger.error("Error: URL is required for predict mode")
            return

        try:
            if os.path.exists(default_params['model_path']):
                url_model.load_model(default_params['model_path'])
                logger.info(f"Loaded model from: {default_params['model_path']}")
            else:
                logger.error("Error: Model not found. Please train the model first.")
                return

            result = url_model.predict(default_params['url'])
            
            print("\nURL Safety Analysis:")
            print("-" * 50)
            print(f"URL: {result['url']}")
            print(f"Combined Safety Score: {result['safety_score']:.2f}")
            print(f"Neural Network Score: {result['nn_score']:.2f}")
            print(f"SVM Score: {result['svm_score']:.2f}")
            print(f"Is Safe: {result['is_safe']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Risk Level: {result['risk_level']}")
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='URL Safety Classifier')
    parser.add_argument('--mode', choices=['train', 'predict', 'process'],
                      help='Mode: train the model, predict URL safety, or process dataset')
    parser.add_argument('--dataset', type=str,
                      help='Path to the dataset file (CSV or Excel)')
    parser.add_argument('--url', type=str,
                      help='URL to check (required for predict mode)')
    parser.add_argument('--model-path', type=str,
                      help='Path to save/load the model')
    parser.add_argument('--output-dataset', type=str,
                      help='Path to save the processed dataset (for process mode)')
    args = parser.parse_args()
    
    params = {}
    if args.mode:
        params['mode'] = args.mode
    if args.dataset:
        params['dataset'] = args.dataset
    if args.url:
        params['url'] = args.url
    if args.model_path:
        params['model_path'] = args.model_path
    if args.output_dataset:
        params['output_dataset'] = args.output_dataset

    main(params)

# main({'mode': 'process'})

# main({
#     'mode': 'train',
#     'dataset': 'src/dataset/dataset_1.xlsx',
#     'model_path': 'my_model.keras'
# })

main({
    'mode': 'predict',
    'url': 'https://google.com',
    'model_path': 'my_model.keras'
})
