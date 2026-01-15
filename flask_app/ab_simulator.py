import pandas as pd
import numpy as np
import time
import random
import json
from datetime import datetime
import logging
import os  

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ABTestSimulator:
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.test_logs = []
        self.results_cache = {}
        
    def simulate_ab_test(self, data_path, test_duration_sec=30, 
                         traffic_split=0.3, batch_size=15, sleep_between_batches=1):
        """
        Симуляция A/B теста на исторических данных
        """
        logger.info("="*60)
        logger.info("STARTING A/B TEST SIMULATION")
        logger.info(f"  Duration: {test_duration_sec} sec")
        logger.info(f"  Traffic split: {traffic_split*100:.0f}% to Staging")
        logger.info(f"  Batch size: {batch_size}")
        logger.info("="*60)

        self.model_manager.refresh_models()
        time.sleep(3)
        
        try:
            data = pd.read_csv(data_path)
        except Exception as e:
            logger.error(f"Failed to load data from {data_path}: {e}")
            return {'error': f'Failed to load data: {str(e)}'}
        
        if 'survived' not in data.columns:
            logger.error("Data does not contain 'survived' column")
            return {'error': "Data does not contain 'survived' column"}
        
        X = data.drop('survived', axis=1)
        y_true = data['survived'].values
        total_samples = len(data)

        model_info = self.model_manager.get_model_info()
        if not model_info['Production']['loaded']:
            logger.error("Production model not loaded!")
            return {'error': 'Production model not loaded'}
        if not model_info['Staging']['loaded']:
            logger.error("Staging model not loaded!")
            return {'error': 'Staging model not loaded'}

        stats = {
            'production': {'total': 0, 'correct': 0},
            'staging': {'total': 0, 'correct': 0},
            'start_time': datetime.now().isoformat(),
            'parameters': {
                'traffic_split': traffic_split,
                'test_duration_sec': test_duration_sec,
                'total_samples': total_samples,
                'batch_size': batch_size
            }
        }
        
        start_time = time.time()
        processed_samples = 0
        
        try:
            while (time.time() - start_time) < test_duration_sec and processed_samples < total_samples:
                end_idx = min(processed_samples + batch_size, total_samples)
                batch_X = X.iloc[processed_samples:end_idx]
                batch_y = y_true[processed_samples:end_idx]
                
                for i in range(len(batch_X)):
                    features = batch_X.iloc[i].to_dict()
                    true_label = int(batch_y[i])

                    if random.random() < traffic_split:
                        model_version = 'Staging'
                    else:
                        model_version = 'Production'
                    
                    try:
                        prediction = self.model_manager.predict(features, model_version)
                        pred_label = 1 if float(prediction) >= 0.5 else 0

                        model_key = model_version.lower()
                        stats[model_key]['total'] += 1
                        if pred_label == true_label:
                            stats[model_key]['correct'] += 1

                        log_entry = {
                            'timestamp': datetime.now().isoformat(),
                            'source': 'simulator',
                            'request_id': f"sim_req_{processed_samples + i}",
                            'model_version': model_version,
                            'model_loaded_version': self.model_manager.model_versions.get(model_version),
                            'prediction': float(prediction),
                            'true_label': true_label,
                            'predicted_label': pred_label,
                            'is_correct': pred_label == true_label,
                            'features_count': len(features)
                        }
                        
                        self.test_logs.append(log_entry)
                        self._save_to_unified_log(log_entry) 
                        
                    except Exception as e:
                        logger.error(f"Prediction error for request {processed_samples + i}: {e}")
                
                processed_samples += len(batch_X)
                
                if (time.time() - start_time) < test_duration_sec:
                    time.sleep(sleep_between_batches)

                if processed_samples % (batch_size * 5) == 0:
                    progress = min(100, processed_samples / total_samples * 100)
                    logger.info(f"Progress: {progress:.1f}% ({processed_samples}/{total_samples})")
            
            stats['end_time'] = datetime.now().isoformat()
            stats['duration_actual_sec'] = time.time() - start_time

            for model in ['production', 'staging']:
                if stats[model]['total'] > 0:
                    stats[model]['accuracy'] = stats[model]['correct'] / stats[model]['total']
                else:
                    stats[model]['accuracy'] = 0

            stats['accuracy_diff'] = stats['staging']['accuracy'] - stats['production']['accuracy']
            stats['total_processed'] = processed_samples

            self.results_cache = stats

            logger.info("="*60)
            logger.info("A/B TEST SIMULATION RESULTS")
            logger.info("="*60)
            logger.info(f"Production (v{model_info['Production']['version']}):")
            logger.info(f"  Requests: {stats['production']['total']}")
            logger.info(f"  Accuracy: {stats['production']['accuracy']:.4f}")
            logger.info(f"Staging (v{model_info['Staging']['version']}):")
            logger.info(f"  Requests: {stats['staging']['total']}")
            logger.info(f"  Accuracy: {stats['staging']['accuracy']:.4f}")
            logger.info(f"Difference (Staging - Production): {stats['accuracy_diff']:.4f}")
            logger.info("="*60)

            aggregated_result = {
                'timestamp': datetime.now().isoformat(),
                'source': 'simulator_aggregated',
                'production_accuracy': stats['production']['accuracy'],
                'staging_accuracy': stats['staging']['accuracy'],
                'accuracy_diff': stats['accuracy_diff'],
                'production_requests': stats['production']['total'],
                'staging_requests': stats['staging']['total'],
                'total_processed': stats['total_processed']
            }
            self._save_to_unified_log(aggregated_result)

            return self._convert_to_json_serializable(stats)
            
        except Exception as e:
            logger.error(f"A/B test simulation failed: {e}")
            return {'error': f'Simulation failed: {str(e)}'}
    
    def _save_to_unified_log(self, log_entry):
        """Сохраняет запись в единый лог-файл"""
        try:
            log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
            log_path = os.path.join(log_dir, 'titanic_ml_pipeline.log')
            
            with open(log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.warning(f"Failed to save log: {e}")
    
    def _convert_to_json_serializable(self, obj):
        """Конвертирует объект с numpy типами в JSON-сериализуемый"""
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def get_recent_logs(self, limit=100):
        """Получение последних логов в упрощенном формате"""
        logs = self.test_logs[-limit:] if self.test_logs else []

        simplified_logs = []
        for log in logs:
            simplified_logs.append({
                'timestamp': log.get('timestamp', ''),
                'model_version': log.get('model_version', ''),
                'model_loaded_version': log.get('model_loaded_version', ''),
                'prediction': log.get('prediction', 0),
                'true_label': log.get('true_label', 0),
                'is_correct': log.get('is_correct', False)
            })
        
        return simplified_logs
    
    def get_results(self):
        """Получение результатов последнего теста"""
        if not self.results_cache:
            return {'message': 'No test results available'}
        return self.results_cache

from model_manager import model_manager
ab_simulator = ABTestSimulator(model_manager)