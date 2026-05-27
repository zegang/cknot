from fastapi import APIRouter, HTTPException, status, Path
from typing import List
from cknot.tools.tool_manager import ToolManager
from cknot.utils.redis_client import get_async_redis_client
from cknot.schemas.tool_config import ToolConfig

router = APIRouter(prefix="/tools", tags=["tools"])

@router.get(
    "/",
    response_model=List[ToolConfig],
    summary="List all registered tools",
    description="Retrieve a comprehensive list of all tools currently registered in the system, including their enabled status and cumulative usage metrics."
)
async def list_tools():
    mgr = ToolManager(get_async_redis_client())
    return await mgr.alist_tool_configs()

@router.get(
    "/{tool_id}",
    response_model=ToolConfig,
    summary="Get tool status",
    description="Get detailed configuration and usage statistics for a specific tool by its unique identifier.",
    responses={404: {"description": "Tool not found"}}
)
async def get_tool_status(
    tool_id: str = Path(..., description="The unique ID of the tool", example="web_search")
):
    mgr = ToolManager(get_async_redis_client())
    cfg = await mgr.aget_tool_config(tool_id)
    if not cfg:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Tool '{tool_id}' not found."
        )
    return cfg

@router.post(
    "/{tool_id}/enable",
    summary="Enable a tool",
    description="Activate a tool to allow the agent to use it in workflows. Note that a graph restart or session refresh might be required for the change to be picked up by active agents.",
    responses={404: {"description": "Tool not found"}}
)
async def enable_tool(
    tool_id: str = Path(..., description="The unique ID of the tool to enable", example="read_log_file")
):
    mgr = ToolManager(get_async_redis_client())
    cfg = await mgr.aget_tool_config(tool_id)
    if not cfg:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Tool '{tool_id}' not found."
        )
    cfg.is_enabled = True
    await mgr.asave_tool_config(cfg)
    return {"message": f"Tool '{tool_id}' has been enabled. Note: Graph restart may be required."}

@router.post(
    "/{tool_id}/disable",
    summary="Disable a tool",
    description="Deactivate a tool to prevent the agent from using it in workflows.",
    responses={404: {"description": "Tool not found"}}
)
async def disable_tool(
    tool_id: str = Path(..., description="The unique ID of the tool to disable", example="wikipedia")
):
    mgr = ToolManager(get_async_redis_client())
    cfg = await mgr.aget_tool_config(tool_id)
    if not cfg:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Tool '{tool_id}' not found."
        )
    cfg.is_enabled = False
    await mgr.asave_tool_config(cfg)
    return {"message": f"Tool '{tool_id}' has been disabled. Note: Graph restart may be required."}