
from datetime import timedelta
from typing import Any, Dict


from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from activities import ingest_data, send_notification, preprocess_data, train_model, evaluate_model, deploy_model
    from dataobjects import PipelineConfig, ErrorResult

@workflow.defn
class MLPipelineWorkflow:
    """
    Main ML Pipeline Workflow orchestrating all activities
    """
    
    @workflow.run
    async def run(self, config: PipelineConfig) -> Dict[str, Any]:
        """
        Execute the complete ML pipeline
        """
        
        # Define retry policy
        retry_policy = RetryPolicy(
            initial_interval=timedelta(seconds=1),
            backoff_coefficient=2.0,
            maximum_interval=timedelta(minutes=5),
            maximum_attempts=3
        )
        
        pipeline_start_time = workflow.now()
        
        try:
            # Stage 1: Data Ingestion
            workflow.logger.info("Starting ML Pipeline - Data Ingestion")
            ingestion_result = await workflow.execute_activity(
                ingest_data,
                config,
                start_to_close_timeout=timedelta(minutes=30),
                retry_policy=retry_policy
            )
            
            # Data quality check
            if ingestion_result.data_quality_score < 0.7:
                await workflow.execute_activity(
                    send_notification,
                    f"Data quality warning: score {ingestion_result.data_quality_score}",
                    "WARNING",
                    start_to_close_timeout=timedelta(minutes=1)
                )
            
            # Stage 2: Data Preprocessing
            workflow.logger.info("Starting Data Preprocessing")
            preprocessing_result = await workflow.execute_activity(
                preprocess_data,
                ingestion_result,
                start_to_close_timeout=timedelta(minutes=45),
                retry_policy=retry_policy
            )
            
            # Stage 3: Model Training
            workflow.logger.info("Starting Model Training")
            training_result = await workflow.execute_activity(
                train_model,
                preprocessing_result,
                start_to_close_timeout=timedelta(hours=2),
                retry_policy=retry_policy
            )
            
            # Stage 4: Model Evaluation
            workflow.logger.info("Starting Model Evaluation")
            evaluation_result = await workflow.execute_activity(
                evaluate_model,
                training_result,
                start_to_close_timeout=timedelta(minutes=30),
                retry_policy=retry_policy
            )
            
            # Stage 5: Model Deployment (only if approved)
            deployment_result = None
            if evaluation_result.model_approved:
                workflow.logger.info("Starting Model Deployment")
                deployment_result = await workflow.execute_activity(
                    deploy_model,
                    evaluation_result,  
                    start_to_close_timeout=timedelta(minutes=30),
                    retry_policy=retry_policy
                )

                result = ErrorResult(
                    error_message=f"Model deployed successfully: {deployment_result.endpoint_url}",
                    error_type="SUCCESS"
                )
                
                await workflow.execute_activity(
                    send_notification,
                    result,
                    start_to_close_timeout=timedelta(minutes=1)
                )
            else:
                result = ErrorResult(
                    error_message="Model failed approval criteria - deployment skipped",
                    error_type="WARNING"
                )
                await workflow.execute_activity(
                    send_notification,
                    result,
                    start_to_close_timeout=timedelta(minutes=1)
                )
            
            # Pipeline completion
            pipeline_end_time = workflow.now()
            execution_time = (pipeline_end_time - pipeline_start_time).total_seconds()
            
            return {
                "status": "SUCCESS",
                "execution_time_seconds": execution_time,
                "ingestion_result": ingestion_result,
                "preprocessing_result": preprocessing_result,
                "training_result": training_result,
                "evaluation_result": evaluation_result,
                "deployment_result": deployment_result,
                "completion_time": pipeline_end_time.isoformat()
            }
            
        except Exception as e:
            # Error handling and notification
            error_result = ErrorResult(
                error_message=str(e),
                error_type="ERROR"
            )
            await workflow.execute_activity(
                send_notification,
               error_result,
                start_to_close_timeout=timedelta(minutes=1)
            )
            
            return {
                "status": "FAILED",
                "error": str(e),
                "completion_time": workflow.now().isoformat()
            }
