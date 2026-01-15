from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import random
import uuid
import time
from datetime import datetime
import os
import json

from model_manager import model_manager
from ab_simulator import ab_simulator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

CONFIG = {
    'traffic_split': 0.3,
    'distribution_method': 'random',
    'enable_ab_testing': True
}

def save_to_unified_log(log_entry):
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

@app.route('/health', methods=['GET'])
def health():
    """Проверка доступности сервиса и моделей"""
    model_info = model_manager.get_model_info()
    status = "healthy" if model_info['Production']['loaded'] else "degraded"
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'source': 'health_check',
        'status': status,
        'production_version': model_info['Production']['version'],
        'staging_version': model_info['Staging']['version']
    }
    save_to_unified_log(log_entry)
    
    return jsonify({
        'status': status,
        'models': model_info,
        'config': CONFIG,
        'timestamp': datetime.now().isoformat(),
        'service': 'Titanic A/B Testing API'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Основной эндпоинт для предсказаний с A/B распределением"""
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data', 'request_id': request_id}), 400
        
        features = data.get('features')
        if not features:
            return jsonify({'error': 'No features provided', 'request_id': request_id}), 400
        
        user_id = data.get('user_id')

        if not CONFIG['enable_ab_testing'] or not model_manager.models.get('Staging'):
            model_version = "Production"
        elif CONFIG['distribution_method'] == 'deterministic' and user_id is not None:
            model_version = "Staging" if user_id % 2 == 0 else "Production"
        else:
            model_version = "Staging" if random.random() < CONFIG['traffic_split'] else "Production"
        
        prediction = model_manager.predict(features, model_version)
        latency_ms = round((time.time() - start_time) * 1000, 2)

        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'source': 'flask_predict',
            'request_id': request_id,
            'model_version': model_version,
            'model_loaded_version': model_manager.model_versions.get(model_version),
            'prediction': float(prediction) if hasattr(prediction, '__float__') else str(prediction),
            'latency_ms': latency_ms,
            'features_count': len(features) if isinstance(features, dict) else 0
        }
        
        save_to_unified_log(log_entry)
        logger.info(f"[{request_id}] {model_version} -> {prediction} ({latency_ms}ms)")
        
        return jsonify({
            'request_id': request_id,
            'prediction': float(prediction) if hasattr(prediction, '__float__') else str(prediction),
            'model_version': model_version,
            'model_loaded_version': model_manager.model_versions.get(model_version),
            'latency_ms': latency_ms,
            'timestamp': datetime.now().isoformat(),
            'ab_testing_enabled': CONFIG['enable_ab_testing']
        })
        
    except ValueError as e:
        error_log = {
            'timestamp': datetime.now().isoformat(),
            'source': 'flask_error',
            'request_id': request_id,
            'error_type': 'validation',
            'error_message': str(e)
        }
        save_to_unified_log(error_log)
        logger.error(f"[{request_id}] Validation error: {e}")
        return jsonify({'error': str(e), 'request_id': request_id}), 400
    except Exception as e:
        error_log = {
            'timestamp': datetime.now().isoformat(),
            'source': 'flask_error',
            'request_id': request_id,
            'error_type': 'internal',
            'error_message': str(e)
        }
        save_to_unified_log(error_log)
        logger.error(f"[{request_id}] Error: {e}")
        return jsonify({'error': 'Internal server error', 'request_id': request_id}), 500

@app.route('/abtest/run', methods=['POST'])
def run_ab_test():
    """Запуск симуляции A/B теста"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'source': 'ab_test_start',
        'action': 'starting_simulation'
    }
    save_to_unified_log(log_entry)
    
    try:
        data = request.get_json() or {}
        
        data_path = data.get('data_path', '../data/current.csv')
        test_duration = data.get('duration', 30)
        traffic_split = data.get('split', 0.3)
        
        logger.info(f"Starting A/B test simulation: duration={test_duration}s, split={traffic_split}")
        
        result = ab_simulator.simulate_ab_test(
            data_path=data_path,
            test_duration_sec=test_duration,
            traffic_split=traffic_split,
            batch_size=10,
            sleep_between_batches=1
        )
        
        if 'error' in result:
            error_log = {
                'timestamp': datetime.now().isoformat(),
                'source': 'ab_test_error',
                'error': result['error']
            }
            save_to_unified_log(error_log)
            return jsonify({'error': result['error']}), 500
        
        test_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"A/B test completed successfully. Test ID: {test_id}")
        
        return jsonify({
            'status': 'success',
            'message': 'A/B test simulation completed',
            'results': result,
            'test_id': test_id
        })
        
    except Exception as e:
        error_log = {
            'timestamp': datetime.now().isoformat(),
            'source': 'ab_test_error',
            'error': str(e)
        }
        save_to_unified_log(error_log)
        logger.error(f"Error starting A/B test: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/abtest/results', methods=['GET'])
def get_ab_results():
    """Получение результатов последнего A/B теста"""
    recent_logs = ab_simulator.get_recent_logs(limit=100)
    
    if not recent_logs:
        return jsonify({'message': 'No AB test logs available'})
    
    prod_stats = {'total': 0, 'correct': 0}
    stag_stats = {'total': 0, 'correct': 0}
    
    for log in recent_logs:
        model = log.get('model_version', '').lower()
        if model == 'production':
            prod_stats['total'] += 1
            if log.get('is_correct', False):
                prod_stats['correct'] += 1
        elif model == 'staging':
            stag_stats['total'] += 1
            if log.get('is_correct', False):
                stag_stats['correct'] += 1
    
    prod_accuracy = prod_stats['correct'] / prod_stats['total'] if prod_stats['total'] > 0 else 0
    stag_accuracy = stag_stats['correct'] / stag_stats['total'] if stag_stats['total'] > 0 else 0

    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'source': 'ab_results_api',
        'production_accuracy': prod_accuracy,
        'staging_accuracy': stag_accuracy,
        'accuracy_diff': stag_accuracy - prod_accuracy,
        'production_requests': prod_stats['total'],
        'staging_requests': stag_stats['total'],
        'total_logs': len(recent_logs)
    }
    
    save_to_unified_log(log_entry)
    
    return jsonify({
        'total_logs': len(recent_logs),
        'results': {
            'production': {
                'requests': prod_stats['total'],
                'correct': prod_stats['correct'],
                'accuracy': prod_accuracy
            },
            'staging': {
                'requests': stag_stats['total'],
                'correct': stag_stats['correct'],
                'accuracy': stag_accuracy
            }
        },
        'sample_logs': recent_logs[:50]
    })

@app.route('/models/refresh', methods=['POST'])
def refresh():
    """Принудительное обновление моделей из реестра"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'source': 'model_refresh',
        'action': 'refreshing_models'
    }
    save_to_unified_log(log_entry)
    
    try:
        model_manager.refresh_models()
        model_info = model_manager.get_model_info()
        
        logger.info(f"Models refreshed: Production v{model_info['Production']['version']}, "
                   f"Staging v{model_info['Staging']['version']}")
        
        return jsonify({
            'message': 'Models refreshed successfully', 
            'models': model_info
        })
    except Exception as e:
        error_log = {
            'timestamp': datetime.now().isoformat(),
            'source': 'model_refresh_error',
            'error': str(e)
        }
        save_to_unified_log(error_log)
        logger.error(f"Failed to refresh models: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/models/info', methods=['GET'])
def get_models_info():
    """Получение информации о загруженных моделях"""
    model_info = model_manager.get_model_info()
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'source': 'model_info',
        'production_version': model_info['Production']['version'],
        'staging_version': model_info['Staging']['version']
    }
    save_to_unified_log(log_entry)
    
    return jsonify({
        'models': model_info,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    logger.info("="*60)
    logger.info("🚀 Titanic A/B Testing Flask Server")
    logger.info("="*60)
    logger.info(f"   A/B Testing: {'ENABLED' if CONFIG['enable_ab_testing'] else 'DISABLED'}")
    logger.info(f"   Traffic Split: {CONFIG['traffic_split']*100:.0f}% to Staging")
    logger.info(f"   Distribution: {CONFIG['distribution_method']}")
    logger.info("="*60)
    
    app.run(host='0.0.0.0', port=5001, debug=True)