import mlflow
import logging
from mlflow.tracking import MlflowClient
from datetime import datetime
from threading import Lock
import pandas as pd
import numpy as np
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, tracking_uri="http://localhost:5000", model_name="Titanic_Survival_Predictor"):
        self.tracking_uri = tracking_uri
        self.model_name = model_name
        self.models = {}
        self.model_uris = {}
        self.model_versions = {}
        self.lock = Lock()
        self.last_update = None
        
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        self._load_models()
        logger.info("✅ ModelManager initialized")

    def _load_models(self):
        """Загрузка Production и Staging моделей через MlflowClient"""
        with self.lock:
            self.models.clear()
            self.model_uris.clear()
            self.model_versions.clear()
            
            for stage in ["Production", "Staging"]:
                try:
                    model_versions = self.client.get_latest_versions(
                        self.model_name, 
                        stages=[stage]
                    )
                    if model_versions:
                        model_version = model_versions[0]
                        model_uri = model_version.source
                        logger.info(f"Found {stage} model: v{model_version.version}")
                        
                        model = mlflow.pyfunc.load_model(model_uri=model_uri)
                        self.models[stage] = model
                        self.model_uris[stage] = model_uri
                        self.model_versions[stage] = model_version.version
                        logger.info(f"  Successfully loaded")
                    else:
                        self.models[stage] = None
                        logger.warning(f"  {stage} model not found in registry")
                        
                except Exception as e:
                    logger.error(f"  Error loading {stage} model: {e}")
                    self.models[stage] = None
            
            self.last_update = datetime.now()

    def refresh_models(self):
        """Принудительное обновление моделей из реестра"""
        logger.info("🔄 Refreshing models from MLflow...")
        old_versions = self.model_versions.copy()
        self._load_models()
        
        for stage in ["Production", "Staging"]:
            old = old_versions.get(stage)
            new = self.model_versions.get(stage)
            if old != new:
                logger.info(f"  {stage}: v{old} -> v{new}" if old else f"  {stage}: None -> v{new}")
        return True

    def _normalize_features(self, features):
        """Нормализация признаков перед предсказанием"""
        if isinstance(features, dict):
            features_df = pd.DataFrame([features])
        else:
            features_df = features

        int_features = ['pclass', 'sex', 'sibsp', 'parch', 'embarked', 'title', 'family_size', 'is_alone']
        
        for feature in int_features:
            if feature in features_df.columns:
                try:
                    features_df[feature] = pd.to_numeric(features_df[feature], errors='coerce').fillna(0).astype(int)
                except Exception as e:
                    logger.warning(f"Could not convert {feature} to int: {e}")
        
        return features_df

    def predict(self, features, model_version="Production"):
        if model_version not in self.models or self.models[model_version] is None:
            raise ValueError(f"Model '{model_version}' is not available")
        
        try:
            features_df = self._normalize_features(features)

            prediction = self.models[model_version].predict(features_df)

            if hasattr(prediction, '__len__') and len(prediction) == 1:
                return float(prediction[0])
            else:
                return float(prediction)
                
        except Exception as e:
            logger.error(f"Prediction error ({model_version}): {e}")
            raise

    def get_model_info(self):
        """Получение информации о текущих моделях"""
        info = {}
        for stage in ["Production", "Staging"]:
            info[stage] = {
                'loaded': self.models.get(stage) is not None,
                'version': self.model_versions.get(stage),
                'uri': self.model_uris.get(stage),
                'last_update': self.last_update.isoformat() if self.last_update else None
            }
        return info

model_manager = ModelManager()