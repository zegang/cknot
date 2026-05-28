from fastapi import APIRouter, HTTPException, status, Path
from typing import List
from cknot.agents.registry import AgentRegistry
from pydantic import BaseModel, Field

router = APIRouter(prefix="/agents", tags=["agents"])

class AgentCapabilities(BaseModel):
    good_at: List[str] = Field(..., description="Capabilities the agent excels at.")
    poor_at: List[str] = Field(..., description="Tasks the agent struggles with.")

class AgentResponse(BaseModel):
    id: str = Field(..., description="The unique registration ID of the agent class.")
    capabilities: AgentCapabilities

@router.get(
    "/",
    response_model=List[AgentResponse],
    summary="List all registered agents",
    description="Retrieves a list of all agents currently registered in the system, providing a high-level view of the team's specialized strengths and weaknesses."
)
async def list_agents():
    """
    Retrieves the team directory from the AgentRegistry.
    """
    all_caps = AgentRegistry.get_all_capabilities()
    return [
        AgentResponse(id=name, capabilities=AgentCapabilities(**caps))
        for name, caps in all_caps.items()
    ]

@router.get(
    "/{agent_id}",
    response_model=AgentResponse,
    summary="Get agent details",
    description="Retrieves detailed capabilities for a specific agent node.",
    responses={404: {"description": "Agent not found"}}
)
async def get_agent(
    agent_id: str = Path(..., description="The registration ID of the agent", example="deep_search")
):
    all_caps = AgentRegistry.get_all_capabilities()
    if agent_id not in all_caps:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent '{agent_id}' not found in registry."
        )
    return AgentResponse(id=agent_id, capabilities=AgentCapabilities(**all_caps[agent_id]))