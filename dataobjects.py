from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class PipelineConfig:
    data_source: str
    target_column: str
    test_size: float = 0.2
    validation_size: float = 0.2
    max_features: int = 50
    cv_folds: int = 5
    scoring_metric: str = "accuracy"
    model_registry_name: str = "ml_model"
    experiment_name: str = "ml_pipeline"

@dataclass
class DataIngestionResult:
    data_path: str
    row_count: int
    column_count: int
    data_quality_score: float
    PipelineConfig: PipelineConfig

@dataclass
class PreprocessingResult:
    processed_data_path: str
    feature_columns: List[str]
    scaler_path: str
    encoders_path: str
    preprocessing_stats: Dict[str, Any]
    PipelineConfig: PipelineConfig

@dataclass
class ModelTrainingResult:
    model_path: str
    model_name: str
    cv_scores: List[float]
    best_params: Dict[str, Any]
    training_metrics: Dict[str, float]
    preprocessing_result: PreprocessingResult


@dataclass
class ModelEvaluationResult:
    test_metrics: Dict[str, float]
    confusion_matrix: List[List[int]]
    classification_report: str
    model_approved: bool
    model_training_result: ModelTrainingResult
    PipelineConfig: PipelineConfig

@dataclass
class DeploymentResult:
    deployment_id: str
    model_version: str
    endpoint_url: str
    deployment_status: str


@dataclass
class ErrorResult:
    error_message: str
    error_type: str


class CustomException(Exception):
    """Custom exception for specific error handling."""
    def __init__(self, message, errors=None):
        super().__init__(message)
        self.errors = errors
