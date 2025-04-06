import os
from ciat.cultural_impact_model import CulturalImpactModel
from ciat.data_processor import CulturalDataProcessor
from ciat.logger import setup_logger
import pandas as pd
import numpy as np

logger = setup_logger("init_app")

def prepare_directories(base_dir: str) -> dict:
    data_dir = os.path.join(base_dir, 'data')
    model_path = os.path.join(data_dir, 'ciat_model.joblib')
    upload_folder = os.path.join(base_dir, 'uploads')

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'assessments'), exist_ok=True)
    os.makedirs(upload_folder, exist_ok=True)

    return {
        "data_dir": data_dir,
        "model_path": model_path,
        "upload_folder": upload_folder
    }

def load_or_train_model(model_path: str, data_processor: CulturalDataProcessor) -> CulturalImpactModel:
    model = CulturalImpactModel()
    if os.path.exists(model_path):
        try:
            model = CulturalImpactModel.load_model(model_path)
            logger.info(f"Loaded pre-trained CIAT model from {model_path}")
            return model
        except Exception as e:
            logger.warning(f"Model load failed: {str(e)}")

    try:
        logger.info("Training model on synthetic data")
        synthetic_X = pd.DataFrame({
            'power_distance': np.random.randint(30, 90, 100),
            'individualism': np.random.randint(20, 80, 100),
            'masculinity': np.random.randint(30, 70, 100),
            'uncertainty_avoidance': np.random.randint(40, 90, 100),
            'technical_requirements': np.random.randint(1, 6, 100),
            'stakeholder_count': np.random.randint(5, 50, 100),
            'team_size': np.random.randint(3, 30, 100),
            'language_barriers': np.random.randint(1, 6, 100),
            'communication_barriers': np.random.randint(1, 6, 100)
        })
        synthetic_y = pd.Series(np.random.binomial(1, 0.6, 100))
        model.train(synthetic_X, synthetic_y)
        model.save_model(model_path)
        logger.info("Model trained and saved successfully with synthetic data")
        return model
    except Exception as synth_error:
        logger.error(f"Error training synthetic model: {str(synth_error)}")
        return CulturalImpactModel()

def prepare_static_assets(base_dir: str):
    static_dirs = ["css", "js", "images"]
    static_base = os.path.join(base_dir, "static")
    for subdir in static_dirs:
        os.makedirs(os.path.join(static_base, subdir), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "templates"), exist_ok=True)
