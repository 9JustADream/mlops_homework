import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleMetricsCollector:
    def __init__(self):
        self.metrics_history = []
    
    def calculate_basic_metrics(self, y_true, y_pred, model_name=""):
        """Вычисление базовых метрик для модели"""
        metrics = {
            'model': model_name,
            'timestamp': datetime.now().isoformat(),
            'samples': len(y_true),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        self.metrics_history.append(metrics)
        return metrics
    
    def compare_models(self, metrics_a, metrics_b):
        """Сравнение двух моделей"""
        comparison = {
            'accuracy_diff': metrics_b['accuracy'] - metrics_a['accuracy'],
            'precision_diff': metrics_b['precision'] - metrics_a['precision'],
            'recall_diff': metrics_b['recall'] - metrics_a['recall'],
            'f1_diff': metrics_b['f1_score'] - metrics_a['f1_score'],
            'relative_improvement': {
                'accuracy': (metrics_b['accuracy'] - metrics_a['accuracy']) / metrics_a['accuracy'] if metrics_a['accuracy'] > 0 else 0,
                'f1': (metrics_b['f1_score'] - metrics_a['f1_score']) / metrics_a['f1_score'] if metrics_a['f1_score'] > 0 else 0
            }
        }
        
        # Простая проверка значимости (разница больше 1%)
        comparison['significant_improvement'] = (
            comparison['accuracy_diff'] > 0.01 or 
            comparison['f1_diff'] > 0.01
        )
        
        return comparison
    
    def save_report(self, metrics_prod, metrics_staging, comparison, output_dir='reports'):
        """Сохранение отчета в JSON"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(output_dir, f'metrics_report_{timestamp}.json')
        
        report = {
            'timestamp': timestamp,
            'production_metrics': metrics_prod,
            'staging_metrics': metrics_staging,
            'comparison': comparison
        }
        
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Report saved to {report_path}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
        
        return report_path

# Создаем глобальный экземпляр
metrics_collector = SimpleMetricsCollector()