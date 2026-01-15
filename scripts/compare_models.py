import pandas as pd
import numpy as np
import mlflow
import logging
from mlflow.tracking import MlflowClient
from datetime import datetime
import os
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compare_and_promote_best_model():
    """
    Сравнивает Production и Staging модели,
    промотирует лучшую в Production
    """
    try:
        logger.info("="*60)
        logger.info("🔄 STARTING MODEL COMPARISON")
        logger.info("="*60)

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(base_dir, 'data', 'current.csv')
        
        if not os.path.exists(data_path):
            logger.error(f"Data file not found: {data_path}")
            return {'status': 'error', 'error_message': 'Data file not found'}
        
        data = pd.read_csv(data_path)
        
        if 'survived' not in data.columns:
            logger.error("Target column 'survived' not found")
            return {'status': 'error', 'error_message': 'Target column survived not found'}
        
        X = data.drop('survived', axis=1)
        y_true = data['survived'].values
        
        logger.info(f"Data loaded. Shape: {data.shape}")
 
        mlflow.set_tracking_uri("http://localhost:5000")
        client = MlflowClient()
        MODEL_NAME = "Titanic_Survival_Predictor"

        logger.info("Searching for models in registry...")
        
        prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if not prod_versions:
            logger.error(f"No Production model found for {MODEL_NAME}")
            return {'status': 'error', 'error_message': 'No Production model found'}
        
        prod_version = prod_versions[0]
        logger.info(f"Found Production model: v{prod_version.version}")
        
        staging_versions = client.get_latest_versions(MODEL_NAME, stages=["Staging"])
        if not staging_versions:
            logger.warning(f"No Staging model found for {MODEL_NAME}")
            return {'status': 'skipped', 'message': 'No Staging model for comparison'}
        
        staging_version = staging_versions[0]
        logger.info(f"Found Staging model: v{staging_version.version}")

        logger.info("Loading and testing models...")
        
        prod_model = mlflow.pyfunc.load_model(model_uri=prod_version.source)
        staging_model = mlflow.pyfunc.load_model(model_uri=staging_version.source)

        prod_predictions = prod_model.predict(X)
        staging_predictions = staging_model.predict(X)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        def calculate_metrics(y_true, y_pred):
            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
            }
        
        prod_metrics = calculate_metrics(y_true, prod_predictions)
        staging_metrics = calculate_metrics(y_true, staging_predictions)
        
        logger.info(f"Production model (v{prod_version.version}):")
        for metric, value in prod_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        logger.info(f"Staging model (v{staging_version.version}):")
        for metric, value in staging_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        logger.info("Making promotion decision...")
        
        prod_accuracy = prod_metrics['accuracy']
        staging_accuracy = staging_metrics['accuracy']
        accuracy_threshold = 0.01 
        
        if staging_accuracy > prod_accuracy + accuracy_threshold:
            logger.info(f"Staging model is better by {(staging_accuracy - prod_accuracy):.4f}")
            logger.info("Promoting Staging to Production...")

            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=prod_version.version,
                stage="Archived",
                archive_existing_versions=False
            )
            logger.info(f"Production model v{prod_version.version} moved to Archived")

            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=staging_version.version,
                stage="Production",
                archive_existing_versions=False
            )
            logger.info(f"Staging model v{staging_version.version} promoted to Production")

            all_staging = client.get_latest_versions(MODEL_NAME, stages=["Staging"])
            for model in all_staging:
                if model.version != staging_version.version:
                    client.transition_model_version_stage(
                        name=MODEL_NAME,
                        version=model.version,
                        stage="Archived",
                        archive_existing_versions=False
                    )
                    logger.info(f"Duplicate Staging model v{model.version} moved to Archived")
            
            result = {
                'status': 'promoted',
                'message': f'Staging model v{staging_version.version} promoted to Production',
                'old_production_version': prod_version.version,
                'new_production_version': staging_version.version,
                'prod_accuracy': float(prod_accuracy),
                'staging_accuracy': float(staging_accuracy),
                'improvement': float(staging_accuracy - prod_accuracy),
                'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
            }
            
        else:
            logger.info(f"Staging model not sufficiently better ({(staging_accuracy - prod_accuracy):.4f})")
            logger.info("Keeping current Production model.")

            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=staging_version.version,
                stage="Archived",
                archive_existing_versions=False
            )
            logger.info(f"Staging model v{staging_version.version} moved to Archived")
            
            result = {
                'status': 'kept',
                'message': f'Production model v{prod_version.version} remains best',
                'prod_accuracy': float(prod_accuracy),
                'staging_accuracy': float(staging_accuracy),
                'improvement': float(staging_accuracy - prod_accuracy),
                'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
            }

        logger.info("Logging comparison results to MLflow...")
        
        with mlflow.start_run(run_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            mlflow.log_param("comparison_date", datetime.now().strftime('%Y-%m-%d'))
            mlflow.log_param("test_data_size", len(data))
            mlflow.log_param("accuracy_threshold", accuracy_threshold)
            mlflow.log_param("old_production_version", prod_version.version)
            mlflow.log_param("new_staging_version", staging_version.version)
            mlflow.log_param("decision", result['status'])
            
            mlflow.log_metric("prod_accuracy", prod_accuracy)
            mlflow.log_metric("prod_precision", prod_metrics['precision'])
            mlflow.log_metric("prod_recall", prod_metrics['recall'])
            mlflow.log_metric("prod_f1", prod_metrics['f1'])
            
            mlflow.log_metric("staging_accuracy", staging_accuracy)
            mlflow.log_metric("staging_precision", staging_metrics['precision'])
            mlflow.log_metric("staging_recall", staging_metrics['recall'])
            mlflow.log_metric("staging_f1", staging_metrics['f1'])
            
            mlflow.log_metric("accuracy_difference", staging_accuracy - prod_accuracy)
        
        logger.info("="*60)
        logger.info("MODEL COMPARISON COMPLETED")
        logger.info("="*60)
        
        return result
        
    except Exception as e:
        logger.error(f"ERROR IN MODEL COMPARISON: {str(e)}", exc_info=True)
        return {
            'status': 'error',
            'error_message': str(e),
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
        }

def main():
    """Функция для локального тестирования"""
    print("="*70)
    print("MODEL COMPARISON TEST")
    print("="*70)
    
    result = compare_and_promote_best_model()
    
    print("\n" + "="*70)
    print("COMPARISON RESULTS:")
    print("="*70)
    
    if result['status'] == 'promoted':
        print(f"NEW MODEL PROMOTED!")
        print(f"   Old Production: version {result.get('old_production_version')}")
        print(f"   New Production: version {result.get('new_production_version')}")
        print(f"   Accuracy improvement: {result.get('improvement'):.4f}")
    elif result['status'] == 'kept':
        print(f"CURRENT MODEL RETAINED")
        print(f"   Improvement: {result.get('improvement'):.4f}")
    elif result['status'] == 'skipped':
        print(f"COMPARISON SKIPPED: {result.get('message')}")
    else:
        print(f"ERROR: {result.get('error_message', 'Unknown error')}")
    
    return result

if __name__ == "__main__":
    main()