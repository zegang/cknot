from fastapi import APIRouter, HTTPException, status, Path, Query
from typing import List, Optional
import logging
from cknot.utils.llm_manager import LLMManager
from cknot.schemas.llm_service import LLMService, LLMServiceType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/llms", tags=["llms"])

@router.post(
    "/",
    status_code=status.HTTP_201_CREATED,
    summary="Register a new LLM service",
    description="Register a new LLM service configuration (OpenAI, vLLM, etc.) in the system."
)
async def register_llm_service(config: LLMService):
    mgr = LLMManager()
    await mgr.aregister_llm_service(config)
    logger.info(f"LLM service '{config.id}' registered successfully.")
    return {"message": f"LLM service '{config.id}' registered successfully."}

@router.get(
    "/",
    response_model=List[LLMService],
    summary="List all LLM services",
    description="Retrieve a list of all registered LLM service configurations."
)
async def list_llm_services(
    service_type: Optional[LLMServiceType] = Query(None, description="Filter services by type (e.g., chat, embedding)")
):
    mgr = LLMManager()
    return await mgr.alist_llm_services(service_type=service_type)

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
    mgr = LLMManager()
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
    mgr = LLMManager()
    await mgr.adelete_llm_service(service_id)
    logger.info(f"LLM service '{service_id}' deleted.")
    return