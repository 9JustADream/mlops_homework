from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from datetime import datetime, timedelta
import sys
import os
import requests
import json
import mlflow
from mlflow.tracking import MlflowClient
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.check_drift import check_drift
from scripts.train_model import train_and_register_model

default_args = {
    'owner': 'ml_engineer',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=60),
}

dag = DAG(
    'titanic_ml_pipeline',
    default_args=default_args,
    description='End-to-end ML pipeline: drift detection, training, A/B testing, and promotion',
    schedule='@daily',
    catchup=False,
    tags=['mlops', 'titanic', 'ab-testing', 'automl']
)

def check_drift_task(**context):
    """Task 1: Проверка дрифта данных"""
    print("="*60)
    print("STARTING DRIFT DETECTION")
    print("="*60)
    
    drift_detected = check_drift()

    ti = context['ti']
    ti.xcom_push(key='drift_detected', value=drift_detected)
    
    if drift_detected:
        print("Drift detected! Proceeding to model training.")
    else:
        print("No drift detected. Pipeline will skip training.")
    
    return drift_detected

def train_model_conditional(**context):
    """Task 2: Обучение модели (только при дрифте)"""
    ti = context['ti']
    drift_detected = ti.xcom_pull(task_ids='check_drift_task', key='drift_detected')
    
    if drift_detected:
        print("="*60)
        print("STARTING MODEL TRAINING")
        print("="*60)
        
        training_result = train_and_register_model()
        
        ti.xcom_push(key='training_result', value=training_result)
        
        if training_result.get('status') == 'success':
            print(f"Model trained successfully: {training_result.get('model_type')}")
            print(f"   Stage: {training_result.get('model_stage')}")
            print(f"   Version: {training_result.get('model_version')}")
            print(f"   Accuracy: {training_result.get('accuracy'):.4f}")
        else:
            error_msg = training_result.get('error_message', 'Unknown error')
            print(f"Training error: {error_msg}")
            raise ValueError(f"Training failed: {error_msg}")
        
        return training_result
    else:
        print("No drift detected, skipping training.")
        return {'status': 'skipped', 'message': 'No drift detected'}

def trigger_ab_test_task(**context):
    """Task 3: Запуск A/B теста через Flask API"""
    ti = context['ti']
    training_result = ti.xcom_pull(task_ids='train_model_task')
    
    if not training_result or training_result.get('status') != 'success':
        print("No new model trained, skipping A/B test.")
        return {'status': 'skipped', 'reason': 'No new model'}
    
    print("="*60)
    print("TRIGGERING A/B TEST VIA FLASK API")
    print("="*60)

    new_model_version = training_result.get('model_version')
    print(f"New model version to test: v{new_model_version}")
    
    try:
        print("Waiting for model to be available in MLflow...")
        time.sleep(10)

        print("🔄 Refreshing models in Flask...")
        refresh_response = requests.post(
            'http://localhost:5001/models/refresh',
            timeout=30
        )
        
        if refresh_response.status_code != 200:
            print(f"Failed to refresh models: {refresh_response.text}")
            time.sleep(5)
            refresh_response = requests.post('http://localhost:5001/models/refresh', timeout=30)

        test_params = {
            'duration': 60,  
            'split': 0.50,
            'data_path': os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'data',
                'current.csv'
            )
        }
        
        print(f"AB Test params: duration={test_params['duration']}s, split={test_params['split']}")

        response = requests.post(
            'http://localhost:5001/abtest/run',
            json=test_params,
            timeout=180
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"A/B test started successfully")
            print(f"   Test ID: {result.get('test_id')}")

            ti.xcom_push(key='new_model_version', value=new_model_version)
            ti.xcom_push(key='ab_test_id', value=result.get('test_id'))

            test_duration = test_params['duration'] + 10
            print(f"   Waiting for test completion ({test_duration} seconds)...")
            time.sleep(test_duration)
            
            return result
        else:
            error_msg = f"Flask API error: {response.status_code} - {response.text}"
            print(f"{error_msg}")
            raise Exception(error_msg)
            
    except requests.exceptions.ConnectionError:
        error_msg = "Cannot connect to Flask server at localhost:5001"
        print(f"{error_msg}")
        print("   Make sure Flask server is running: python flask_app/app.py --port 5001")
        raise Exception(error_msg)
    except Exception as e:
        print(f"Error triggering A/B test: {e}")
        raise

def analyze_ab_results_task(**context):
    """Task 4: Анализ результатов A/B теста"""
    ti = context['ti']
    ab_test_result = ti.xcom_pull(task_ids='trigger_ab_test')
    
    if not ab_test_result or ab_test_result.get('status') == 'skipped':
        print("No A/B test results to analyze")
        return {'status': 'skipped'}

    new_model_version = ti.xcom_pull(task_ids='trigger_ab_test', key='new_model_version')
    
    if not new_model_version:
        print("Cannot get new model version")
        return {'status': 'error', 'message': 'Model version not found'}
    
    print("="*60)
    print("ANALYZING A/B TEST RESULTS")
    print("="*60)
    
    try:
        response = requests.get('http://localhost:5001/abtest/results', timeout=30)
        
        if response.status_code != 200:
            print(f"Cannot get results: {response.status_code}")
            return {'status': 'failed_to_get_results'}
        
        results = response.json()

        MIN_IMPROVEMENT = 0.01
        MIN_REQUESTS = 20

        prod_accuracy = results['results']['production'].get('accuracy', 0)
        stag_accuracy = results['results']['staging'].get('accuracy', 0)
        prod_requests = results['results']['production'].get('requests', 0)
        stag_requests = results['results']['staging'].get('requests', 0)
        
        accuracy_improvement = stag_accuracy - prod_accuracy
        
        print("RESULTS ANALYSIS:")
        print(f"   Production accuracy: {prod_accuracy:.4f} ({prod_requests} requests)")
        print(f"   Staging accuracy: {stag_accuracy:.4f} ({stag_requests} requests)")
        print(f"   Accuracy improvement: {accuracy_improvement:.4f}")

        has_enough_data = (prod_requests >= MIN_REQUESTS and stag_requests >= MIN_REQUESTS)
        meets_accuracy_criteria = (accuracy_improvement >= MIN_IMPROVEMENT)
        
        if has_enough_data and meets_accuracy_criteria:
            print(f"Staging model meets promotion criteria!")

            promotion_result = promote_staging_model(new_model_version)
            
            result = {
                'status': 'promoted',
                'decision_reason': 'Meets promotion criteria',
                'accuracy_improvement': accuracy_improvement,
                'production_accuracy': prod_accuracy,
                'staging_accuracy': stag_accuracy,
                'promotion_details': promotion_result
            }
            
        else:
            print(f"Staging model does not meet promotion criteria")

            archive_result = archive_staging_model(new_model_version)
            
            result = {
                'status': 'not_promoted',
                'decision_reason': 'Does not meet promotion criteria',
                'accuracy_improvement': accuracy_improvement,
                'production_accuracy': prod_accuracy,
                'staging_accuracy': stag_accuracy,
                'archive_details': archive_result
            }
        
        ti.xcom_push(key='analysis_result', value=result)
        return result
        
    except Exception as e:
        print(f"Error analyzing A/B test results: {e}")
        raise

def promote_staging_model(model_version):
    """Продвижение staging модели в production"""
    print(f"Promoting model version {model_version} to Production...")
    
    client = MlflowClient()
    model_name = "Titanic_Survival_Predictor"
    
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    old_prod_version = prod_versions[0].version if prod_versions else None

    if old_prod_version:
        client.transition_model_version_stage(
            name=model_name,
            version=old_prod_version,
            stage="Archived"
        )
        print(f"   Archived old production model: v{old_prod_version}")

    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage="Production"
    )
    
    print(f"Model v{model_version} promoted to Production")
    
    return {
        'status': 'promoted',
        'old_production_version': old_prod_version,
        'new_production_version': model_version
    }

def archive_staging_model(model_version):
    """Архивация staging модели"""
    print(f"Archiving model version {model_version}...")
    
    client = MlflowClient()
    model_name = "Titanic_Survival_Predictor"

    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage="Archived"
    )
    
    print(f"Model v{model_version} archived")
    
    return {
        'status': 'archived',
        'staging_version': model_version
    }

start_task = EmptyOperator(task_id='start', dag=dag)
check_drift_task_op = PythonOperator(task_id='check_drift_task', python_callable=check_drift_task, dag=dag)
train_model_task_op = PythonOperator(task_id='train_model_task', python_callable=train_model_conditional, dag=dag)
trigger_ab_test_op = PythonOperator(task_id='trigger_ab_test', python_callable=trigger_ab_test_task, dag=dag)
analyze_results_op = PythonOperator(task_id='analyze_results', python_callable=analyze_ab_results_task, dag=dag)
end_task = EmptyOperator(task_id='end', dag=dag)

start_task >> check_drift_task_op >> train_model_task_op >> trigger_ab_test_op >> analyze_results_op >> end_task

if __name__ == "__main__":
    print("DAG 'titanic_ml_pipeline' успешно создан!")
    print("   Tasks: start → check_drift → train_model → trigger_ab_test → analyze_results → end")