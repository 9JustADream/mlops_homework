import pandas as pd
import numpy as np
import mlflow
import mlflow.pyfunc  
import logging
import os
import pickle
import sys
import time
from datetime import datetime
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from pycaret.classification import setup, compare_models, tune_model, finalize_model, pull

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.model_wrapper import PyCaretModelWrapper

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_and_register_model():
    logger.info("ШАГ 1: Начало процесса обучения и регистрации модели.")
    
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, 'data')
        models_dir = os.path.join(base_dir, 'models')
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)

        data_path = os.path.join(data_dir, 'current.csv')

        data = pd.read_csv(data_path)

        logger.info(f"Данные загружены. Размер: {data.shape}")

        logger.info("ШАГ 2: Запуск AutoML с помощью PyCaret...")

        target_column = 'survived'

        exp = setup(
            data=data,
            target=target_column,
            session_id=42,
            log_experiment=False, 
            experiment_name='Titanic_Classification',
            log_plots=False,  
            verbose=False,
            transformation=False,
            normalize=True,
            train_size=0.8,
            fold=3, 
            log_data=False,
            fix_imbalance=False,
            remove_multicollinearity=True,
            multicollinearity_threshold=0.9,
            n_jobs=-1)

        best_model = compare_models(n_select=1, sort='Accuracy', fold=3, turbo=True, verbose=False)
        tuned_model = tune_model(best_model, optimize='Accuracy', n_iter=20, verbose=False)
        final_model = finalize_model(tuned_model)
        
        model_type_name = type(final_model).__name__
        logger.info(f"Лучшая модель определена: {model_type_name}")

        results_df = pull()
        if not results_df.empty:
            best_metrics = results_df.iloc[0].to_dict()
            accuracy = best_metrics.get('Accuracy', 0.0)
        else:
            best_metrics = {}
            accuracy = 0.0
        logger.info(f"Метрики обучения получены. Accuracy: {accuracy:.4f}")

        logger.info("ШАГ 3: Сохранение локальной копии модели...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        local_model_filename = f'titanic_model_{timestamp}.pkl'
        local_model_path = os.path.join(models_dir, local_model_filename)
        with open(local_model_path, 'wb') as f:
            pickle.dump(final_model, f)
        logger.info(f"Локальная копия модели сохранена: {local_model_path}")

        logger.info("ШАГ 4: Начало ручного логирования в MLflow...")

        mlflow.set_tracking_uri("http://localhost:5000")
        experiment_name = "Titanic_Classification"
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=f"run_{timestamp}") as run:
            run_id = run.info.run_id
            logger.info(f"Создан MLflow run с ID: {run_id}")

            mlflow.log_param("model_type", model_type_name)
            mlflow.log_param("target_column", target_column)
            mlflow.log_param("training_samples", len(data))
            mlflow.log_param("pycaret_session_id", 42)

            for metric_name, metric_value in best_metrics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(metric_name, float(metric_value))

            logger.info("Логирование модели как pyfunc артефакта...")
            pyfunc_wrapper = PyCaretModelWrapper(final_model)

            input_example = data.drop(columns=[target_column]).iloc[:5]
            signature = infer_signature(input_example, final_model.predict(input_example))

            artifact_path = "model"
            mlflow.pyfunc.log_model(
                name=artifact_path,
                python_model=pyfunc_wrapper,
                signature=signature,
                input_example=input_example.iloc[:1],
                registered_model_name="Titanic_Survival_Predictor" 
            )

            logger.info("ШАГ 5: Регистрация модели в Model Registry...")
            client = MlflowClient()
            registered_model_name = "Titanic_Survival_Predictor"

            time.sleep(10)
            model_versions = client.search_model_versions(f"run_id='{run_id}'")
            if model_versions:
                new_version = model_versions[0].version
                logger.info(f"Модель зарегистрирована как {registered_model_name}, версия {new_version}")
            else:
                model_uri = f"runs:/{run_id}/{artifact_path}"
                mv = mlflow.register_model(model_uri, registered_model_name)
                new_version = mv.version
                logger.info(f"Модель зарегистрирована через URI, версия {new_version}")

            logger.info("ШАГ 6: Управление стадиями модели (Staging/Production)...")
            model_stage = "None"
            staging_threshold = 0.65 
            
            if accuracy >= staging_threshold:
                logger.info(f"Accuracy ({accuracy:.4f}) >= {staging_threshold}. Перевод модели в стадию 'Staging'.")
                client.transition_model_version_stage(
                    name=registered_model_name,
                    version=new_version,
                    stage="Staging",
                    archive_existing_versions=False
                )
                model_stage = "Staging"
            else:
                logger.info(f"Accuracy ({accuracy:.4f}) < {staging_threshold}. Модель остается без стадии (None).")
            
            logger.info("="*60)
            logger.info("ПРОЦЕСС ОБУЧЕНИЯ И РЕГИСТРАЦИИ УСПЕШНО ЗАВЕРШЕН!")
            logger.info("="*60)

            model_uri_for_load = f"models:/{registered_model_name}/{new_version}"
            if model_stage != "None":
                model_uri_for_load = f"models:/{registered_model_name}/{model_stage}"
            
            result = {
                'status': 'success',
                'model_path': local_model_path,
                'model_type': model_type_name,
                'accuracy': float(accuracy),
                'mlflow_run_id': run_id,
                'registered_model_name': registered_model_name,
                'model_version': new_version,
                'model_stage': model_stage,
                'model_uri': model_uri_for_load,
                'timestamp': timestamp,
                'message': f'Модель {model_type_name} зарегистрирована как {registered_model_name} v{new_version} ({model_stage})'
            }
            
            return result
            
    except Exception as e:
        logger.error(f"КРИТИЧЕСКАЯ ОШИБКА В ПРОЦЕССЕ: {str(e)}", exc_info=True)
        return {
            'status': 'error',
            'error_message': str(e),
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
        }

def main():
    """Функция для локального тестирования скрипта без Airflow."""
    print("="*70)
    print("ТЕСТИРОВАНИЕ: PyCaret + MLflow (ручное логирование через pyfunc)")
    print("="*70)
    
    result = train_and_register_model()
    
    print("\n" + "="*70)
    print("РЕЗУЛЬТАТЫ ВЫПОЛНЕНИЯ:")
    print("="*70)
    
    if result['status'] == 'success':
        print(f"УСПЕШНО!")
        print(f"   Тип модели:       {result['model_type']}")
        print(f"   Точность (Accuracy): {result['accuracy']:.4f}")
        print(f"   Стадия модели:    {result['model_stage']}")
        print(f"   Имя в реестре:    {result['registered_model_name']} v{result['model_version']}")
        print(f"   Run ID MLflow:    {result['mlflow_run_id']}")
        print(f"   URI модели:       {result['model_uri']}")
        print(f"\nОткройте MLflow UI: http://localhost:5000")
        print("Проверьте вкладки 'Experiments' и 'Models'.")
    else:
        print(f"ОШИБКА: {result.get('error_message', 'Неизвестная ошибка')}")
    
    return result

if __name__ == "__main__":
    main()