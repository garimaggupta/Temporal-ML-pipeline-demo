import asyncio
from temporalio.worker import Worker
from activities import ingest_data, send_notification, preprocess_data, train_model, evaluate_model, deploy_model
from MLPipelineWorkflow import MLPipelineWorkflow
from client import get_client

import logging
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



async def main():
    logging.basicConfig(level=logging.INFO)



    # Start client
    client = await get_client()


    # Run unique task queue for this particular host
# Create worker
    worker = Worker(
        client,
        task_queue="ml-pipeline-task-queue",
        workflows=[MLPipelineWorkflow],
        activities=[
            ingest_data,
            preprocess_data,
            train_model,
            evaluate_model,
            deploy_model,
            send_notification
        ]
    )
    

    # Start worker
    logger.info("Starting Temporal worker...")
    await worker.run()

if __name__ == "__main__":
    asyncio.run(main())