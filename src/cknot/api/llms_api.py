from fastapi import APIRouter, HTTPException, status, Path
from typing import List
import logging
from cknot.utils.llm_manager import LLMManager
from cknot.utils.redis_client import get_async_redis_client
from cknot.schemas.llm_service import LLMService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/llms", tags=["llms"])

@router.post(
    "/",
    status_code=status.HTTP_201_CREATED,
    summary="Register a new LLM service",
    description="Register a new LLM service configuration (OpenAI, vLLM, etc.) in the system."
)
async def register_llm_service(config: LLMService):
    mgr = LLMManager(get_async_redis_client())
    await mgr.aregister_llm_service(config)
    logger.info(f"LLM service '{config.id}' registered successfully.")
    return {"message": f"LLM service '{config.id}' registered successfully."}

@router.get(
    "/",
    response_model=List[LLMService],
    summary="List all LLM services",
    description="Retrieve a list of all registered LLM service configurations."
)
async def list_llm_services():
    mgr = LLMManager(get_async_redis_client())
    return await mgr.alist_llm_services()

@router.get(
    "/{service_id}",
    response_model=LLMService,
    summary="Get LLM service details",
    description="Get configuration details for a specific LLM service by its ID.",
    responses={404: {"description": "LLM service not found"}}
)
async def get_llm_service(
    service_id: str = Path(..., description="The unique ID of the LLM service", example="openai_gpt4")
):
    mgr = LLMManager(get_async_redis_client())
    config = await mgr.aget_llm_service(service_id)
    if not config:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="LLM service not found.")
    return config

@router.delete(
    "/{service_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete an LLM service",
    description="Remove an LLM service configuration from the system.",
    responses={404: {"description": "LLM service not found"}}
)
async def delete_llm_service(
    service_id: str = Path(..., description="The unique ID of the LLM service to delete", example="vllm_local")
):
    mgr = LLMManager(get_async_redis_client())
    await mgr.adelete_llm_service(service_id)
    logger.info(f"LLM service '{service_id}' deleted.")
    return