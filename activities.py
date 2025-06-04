
import logging
import pickle
from datetime import datetime


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mlflow
import mlflow.sklearn

from temporalio import activity
from dataobjects import PipelineConfig, DataIngestionResult, PreprocessingResult, ModelTrainingResult, ModelEvaluationResult, DeploymentResult, ErrorResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ErrorAPIUnavailable = "DataPipelineAPIFailure"

@activity.defn
async def ingest_data(config: PipelineConfig) -> DataIngestionResult:
    """
    Ingest data from various sources with quality checks
    """
    logger.info(f"Starting data ingestion from {config.data_source}")
    
    try:
        # Simulate data loading (replace with actual data source)
        if config.data_source.endswith('.csv'):
            df = pd.read_csv(config.data_source)
        elif config.data_source.startswith('postgresql://'):
            # Simulate database connection
            df = pd.DataFrame({
                'feature1': np.random.randn(1000),
                'feature2': np.random.randn(1000),
                'feature3': np.random.choice(['A', 'B', 'C'], 1000),
                'target': np.random.choice([0, 1], 1000)
            })
        else:
            raise ValueError(f"Unsupported data source: {config.data_source}")
        
        # Data quality checks
        missing_percentage = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        duplicate_percentage = df.duplicated().sum() / len(df)
        data_quality_score = 1.0 - (missing_percentage + duplicate_percentage)
        
        # Save raw data
        data_path = f"demodata/data/raw_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        df.to_pickle(data_path)
        
        result = DataIngestionResult(
            data_path=data_path,
            row_count=len(df),
            column_count=len(df.columns),
            data_quality_score=data_quality_score,
            PipelineConfig=config
        )
        
        logger.info(f"Data ingestion completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Data ingestion failed: {str(e)}")
        raise

@activity.defn
async def preprocess_data(ingestion_result: DataIngestionResult) -> PreprocessingResult:
    """
    Preprocess data including cleaning, feature engineering, and encoding
    """
    logger.info("Starting data preprocessing")
    
    try:
        # Load raw data
        df = pd.read_pickle(ingestion_result.data_path)
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        # Fill missing values
        for col in numeric_columns:
            if col != ingestion_result.PipelineConfig.target_column:
                df[col].fillna(df[col].median(), inplace=True)
        
        for col in categorical_columns:
            if col != ingestion_result.PipelineConfig.target_column:
                df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)
        
        # Remove duplicates
        df.drop_duplicates(inplace=True)
        
        # Feature engineering
        if 'feature1' in df.columns and 'feature2' in df.columns:
            df['feature1_x_feature2'] = df['feature1'] * df['feature2']
            df['feature1_plus_feature2'] = df['feature1'] + df['feature2']
        
        # Encode categorical variables
        encoders = {}
        for col in categorical_columns:
            if col != ingestion_result.PipelineConfig.target_column:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
        
        # Separate features and target
        X = df.drop(columns=[ingestion_result.PipelineConfig.target_column])
        y = df[ingestion_result.PipelineConfig.target_column]
        
        # Feature scaling
        scaler = StandardScaler()
        numeric_features = X.select_dtypes(include=[np.number]).columns
        X[numeric_features] = scaler.fit_transform(X[numeric_features])
        
        # Feature selection (keep top features based on correlation)
        if len(X.columns) > ingestion_result.PipelineConfig.max_features:
            correlations = abs(X.corrwith(y))
            top_features = correlations.nlargest(ingestion_result.PipelineConfig.max_features).index.tolist()
            X = X[top_features]
        
        # Save processed data and artifacts
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        processed_data_path = f"demodata/data/processed_data_{timestamp}.pkl"
        scaler_path = f"demodata/artifacts/scaler_{timestamp}.pkl"
        encoders_path = f"demodata/artifacts/encoders_{timestamp}.pkl"
        
        processed_df = pd.concat([X, y], axis=1)
        processed_df.to_pickle(processed_data_path)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        with open(encoders_path, 'wb') as f:
            pickle.dump(encoders, f)
        
        preprocessing_stats = {
            'original_shape': df.shape,
            'processed_shape': processed_df.shape,
            'features_selected': list(X.columns),
            'numeric_features': list(numeric_features),
            'categorical_features_encoded': list(encoders.keys())
        }
        
        result = PreprocessingResult(
            processed_data_path=processed_data_path,
            feature_columns=list(X.columns),
            scaler_path=scaler_path,
            encoders_path=encoders_path,
            preprocessing_stats=preprocessing_stats,
            PipelineConfig=ingestion_result.PipelineConfig
        )
        
        logger.info(f"Preprocessing completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise

@activity.defn
async def train_model(preprocessing_result: PreprocessingResult) -> ModelTrainingResult:
    """
    Train multiple models with hyperparameter tuning
    """
    logger.info("Starting model training")
    
    try:
        # Load processed data
        df = pd.read_pickle(preprocessing_result.processed_data_path)
        X = df[preprocessing_result.feature_columns]
        y = df[preprocessing_result.PipelineConfig.target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=preprocessing_result.PipelineConfig.test_size, random_state=42, stratify=y
        )
        
        # Define models and hyperparameters
        models = {
            'RandomForest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                }
            },
            'LogisticRegression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                }
            }
        }
        
        best_model = None
        best_score = 0
        best_model_name = ""
        best_params = {}
        
        # Start MLflow experiment
        mlflow.set_experiment(preprocessing_result.PipelineConfig.experiment_name)
        
        with mlflow.start_run():
            for model_name, model_config in models.items():
                logger.info(f"Training {model_name}")
                
                # Grid search with cross-validation
                grid_search = GridSearchCV(
                    model_config['model'],
                    model_config['params'],
                    cv=preprocessing_result.PipelineConfig.cv_folds,
                    scoring=preprocessing_result.PipelineConfig.scoring_metric,
                    n_jobs=-1
                )
                
                grid_search.fit(X_train, y_train)
                
                # Cross-validation scores
                cv_scores = cross_val_score(
                    grid_search.best_estimator_,
                    X_train, y_train,
                    cv=preprocessing_result.PipelineConfig.cv_folds,
                    scoring=preprocessing_result.PipelineConfig.scoring_metric
                )
                
                mean_cv_score = cv_scores.mean()
                
                # Log to MLflow
                with mlflow.start_run(nested=True):
                    mlflow.log_params(grid_search.best_params_)
                    mlflow.log_metric("cv_mean_score", mean_cv_score)
                    mlflow.log_metric("cv_std_score", cv_scores.std())
                    mlflow.sklearn.log_model(grid_search.best_estimator_, model_name)
                
                # Update best model
                if mean_cv_score > best_score:
                    best_score = mean_cv_score
                    best_model = grid_search.best_estimator_
                    best_model_name = model_name
                    best_params = grid_search.best_params_
        
        # Save best model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = f"demodata/models/best_model_{timestamp}.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        
        # Calculate training metrics
        train_predictions = best_model.predict(X_train)
        training_metrics = {
            'train_accuracy': accuracy_score(y_train, train_predictions),
            'cv_mean_score': float(best_score),
            'cv_std_score': float(cv_scores.std())
        }
        
        result = ModelTrainingResult(
            model_path=model_path,
            model_name=best_model_name,
            cv_scores=cv_scores.tolist(),
            best_params=best_params,
            training_metrics=training_metrics,
            preprocessing_result=preprocessing_result
        )
        
        logger.info(f"Model training completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise

@activity.defn
async def evaluate_model(
    training_result: ModelTrainingResult,
) -> ModelEvaluationResult:
    """
    Evaluate model on test set and determine if it meets approval criteria
    """
    logger.info("Starting model evaluation")
    
    try:
        # Load model and data
        with open(training_result.model_path, 'rb') as f:
            model = pickle.load(f)
        
        df = pd.read_pickle(training_result.preprocessing_result.processed_data_path)
        X = df[training_result.preprocessing_result.feature_columns]
        y = df[training_result.preprocessing_result.PipelineConfig.target_column]
        
        # Split to get test set (same split as training)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=training_result.preprocessing_result.PipelineConfig.test_size, random_state=42, stratify=y
        )
        
        # Make predictions
        test_predictions = model.predict(X_test)
        test_probabilities = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        test_accuracy = accuracy_score(y_test, test_predictions)
        cm = confusion_matrix(y_test, test_predictions)
        report = classification_report(y_test, test_predictions)
        
        test_metrics = {
            'test_accuracy': float(test_accuracy),
            'test_samples': len(y_test)
        }
        
        # Model approval logic (customize based on your requirements)
        approval_threshold = 0.8
        model_approved = test_accuracy >= approval_threshold
        
        result = ModelEvaluationResult(
            test_metrics=test_metrics,
            confusion_matrix=cm.tolist(),
            classification_report=report,
            model_approved=model_approved,
            model_training_result=training_result,
            PipelineConfig=training_result.preprocessing_result.PipelineConfig
        )
        
        logger.info(f"Model evaluation completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {str(e)}")
        raise

@activity.defn
async def deploy_model(
    evaluation_result: ModelEvaluationResult
) -> DeploymentResult:
    """
    Deploy approved model to production
    """
    logger.info("Starting model deployment")
    
    try:
        if not evaluation_result.model_approved:
            raise ValueError("Model not approved for deployment")
        
        # Register model in MLflow
        model_version = f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run():
            # Load and log the model
            with open(evaluation_result.model_training_result.model_path, 'rb') as f:
                model = pickle.load(f)
            
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name=evaluation_result.model_training_result.preprocessing_result.PipelineConfig.model_registry_name
            )
            
            # Log evaluation metrics
            mlflow.log_metrics(evaluation_result.test_metrics)
        
        # Simulate deployment (replace with actual deployment logic)
        deployment_id = f"deployment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        endpoint_url = f"https://api.example.com/models/{deployment_id}/predict"
        
        result = DeploymentResult(
            deployment_id=deployment_id,
            model_version=model_version,
            endpoint_url=endpoint_url,
            deployment_status="SUCCESS"
        )
        
        logger.info(f"Model deployment completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Model deployment failed: {str(e)}")
        raise

@activity.defn
async def send_notification(error_result: ErrorResult) -> None:
    """
    Send notification about pipeline status
    """
    logger.info(f"Notification [{error_result.error_type}]: {error_result.error_message}")
    # Implement actual notification logic (email, Slack, etc.)