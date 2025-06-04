import asyncio
import os
from temporalio.client import Client, TLSConfig
from datetime import datetime
from dataobjects import PipelineConfig
from MLPipelineWorkflow import MLPipelineWorkflow
import logging
from typing import Optional

logger = logging.getLogger(__name__)

async def get_client()-> Client:

    if (
        os.getenv("TEMPORAL_MTLS_TLS_CERT")
        and os.getenv("TEMPORAL_MTLS_TLS_KEY") is not None
    ):
        server_root_ca_cert: Optional[bytes] = None
        with open(os.getenv("TEMPORAL_MTLS_TLS_CERT"), "rb") as f:
            client_cert = f.read()

        with open(os.getenv("TEMPORAL_MTLS_TLS_KEY"), "rb") as f:
            client_key = f.read()

        # Start client
        client = await Client.connect(
            os.getenv("TEMPORAL_HOST_URL"),
            namespace=os.getenv("TEMPORAL_NAMESPACE"),
            tls=TLSConfig(
                server_root_ca_cert=server_root_ca_cert,
                client_cert=client_cert,
                client_private_key=client_key,
            ),
            #data_converter=dataclasses.replace(
            #    temporalio.converter.default(), payload_codec=EncryptionCodec()
            #),            
        )
    else:
        client = await Client.connect(
            "localhost:7233",
        )

    return client   

async def start_pipeline():
    """
    Example function to start the ML pipeline workflow
    """

    client = await get_client()
    
    config = PipelineConfig(
        data_source="demodata/data/churn.csv",  # or path to CSV file
        target_column="target",
        test_size=0.2,
        validation_size=0.2,
        max_features=20,
        cv_folds=5,
        scoring_metric="accuracy",
        model_registry_name="customer_churn_model",
        experiment_name="churn_prediction_pipeline"
    )
    
    # Start workflow
    handle = await client.start_workflow(
        MLPipelineWorkflow.run,
        config,
        id=f"ml-pipeline-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        task_queue="ml-pipeline-task-queue"
    )
    
    logger.info(f"Started workflow with ID: {handle.id}")
    
    # Wait for result
    result = await handle.result()
    logger.info(f"Pipeline completed with result: {result}")
    
    return result

if __name__ == "__main__":
    # Run worker
    asyncio.run(start_pipeline())
    
    # To start a pipeline workflow, run:
    # asyncio.run(start_pipeline())  