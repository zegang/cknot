from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
import logging
from typing import Optional, List, Any, Annotated
from contextlib import asynccontextmanager
from cknot.utils.logging_config import setup_logging
from cknot.graphs.orchestrator import create_graph
from cknot.utils.llm_manager import LLMManager
from cknot.schemas.llm_service import LLMService
from cknot.utils.redis_client import get_redis_client, get_async_redis_client
from cknot.api.tools_api import router as tools_router
from cknot.api.llms_api import router as llms_router
from cknot.api.agents_api import router as agents_router
from cknot.api.users_api import router as users_router
from cknot.api.auth import get_current_user, get_user_key, create_access_token
from cknot.utils.user_manager import UserManager
from cknot.config.config import settings
from langchain_core.messages import HumanMessage, AIMessage
import os
from fastapi.middleware.cors import CORSMiddleware

# Initialize global logging for the API
setup_logging()
logger = logging.getLogger(__name__)

# Lifespan context to handle async setup of infrastructure components
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize checkpointer if it requires async setup (e.g., AsyncRedisSaver)
    if hasattr(orchestrator.checkpointer, "asetup"):
        await orchestrator.checkpointer.asetup()
    yield
    # Shutdown logic can be added here if needed

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="""
    ## CKnot Agentic Orchestration API
    
    The CKnot API allows you to interact with a hierarchical multi-agent system designed for system debugging and general assistance.
    
    ### Key Features:
    * **Multi-Agent Orchestration**: Dynamic task delegation via the Boss Agent.
    * **Infrastructure Management**: CRUD operations for LLM providers and System Tools.
    * **Persistence**: Redis-backed session isolation and checkpointing.
    * **Security**: OAuth2 with JWT and role-based access control.
    """,
    version="0.0.1-alpha",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Token(BaseModel):
    access_token: str = Field(..., description="The JWT access token.")
    token_type: str = Field(..., description="The type of token (typically 'bearer').")

@app.post(
    "/token", 
    response_model=Token, 
    tags=["auth"], 
    summary="Obtain OAuth2 Access Token"
)
async def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    """
    Authenticates a user and returns a JWT access token.
    This token must be used in the 'Authorization: Bearer <token>' header for protected endpoints.
    """
    mgr = UserManager(get_async_redis_client())
    user_data = await mgr.aauthenticate(form_data.username, form_data.password)
    
    if user_data:
        access_token = create_access_token(data={"sub": form_data.username})
        logger.info(f"User '{form_data.username}' logged in successfully.")
        return {"access_token": access_token, "token_type": "bearer"}
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect username or password",
        headers={"WWW-Authenticate": "Bearer"},
    )

# The compiled graph is thread-safe and handles concurrency internally
# using the thread_id provided in the config.
orchestrator = create_graph()

# Include the agents management router
app.include_router(
    agents_router,
    dependencies=[Depends(get_current_user)]
)

# Include the LLM services management router
app.include_router(
    llms_router,
    dependencies=[Depends(get_current_user)]
)

# Include the tools management router
app.include_router(
    tools_router,
    dependencies=[Depends(get_current_user)]
)

# Include the users management router
app.include_router(users_router)

class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's input message to the agent.")
    session_id: str = Field(..., description="Unique thread ID for state isolation and persistence.")
    user_id: str = Field(..., description="The unique identifier for the user.")
    current_task: Optional[str] = Field("general_assistance", description="Optional hint for LLM capability selection.")

class ChatResponse(BaseModel):
    content: str = Field(..., description="The response text from the agentic graph.")
    session_id: str = Field(..., description="The session ID associated with this turn.")
    requires_action: bool = Field(..., description="Indicates if the workflow is paused awaiting human approval.")
    next_node: Optional[str] = Field(None, description="The name of the next node (specialist or tool) if approval is required.")

@app.post(
    "/chat", 
    response_model=ChatResponse, 
    tags=["orchestration"], 
    summary="Interact with the Agentic Graph"
)
async def chat(payload: ChatRequest, current_user: Annotated[str, Depends(get_current_user)]):
    """
    Main entry point for the agent. Handles state isolation via session_id.
    """
    config = {
        "configurable": {
            "thread_id": payload.session_id,
            "session_id": payload.session_id,
            "user_id": current_user,
            "current_task": payload.current_task or "general_assistance"
        }
    }
    
    # 1. Check if the current session is already waiting for an interrupt
    state = await orchestrator.aget_state(config)
    if state.next:
        raise HTTPException(
            status_code=400, 
            detail=f"Session {payload.session_id} is pending approval for node: {state.next}"
        )

    # 2. Prepare inputs
    inputs = {
        "messages": [HumanMessage(content=payload.message)]
    }
    
    try:
        # Run the graph. ainvoke will stop automatically if it hits a breakpoint.
        final_output = await orchestrator.ainvoke(inputs, config)
        
        # Inspect the state to see if we hit the 'tools' breakpoint
        new_state = await orchestrator.aget_state(config)
        last_message = final_output["messages"][-1]

        return ChatResponse(
            content=last_message.content if hasattr(last_message, "content") else str(last_message),
            session_id=payload.session_id,
            requires_action=len(new_state.next) > 0,
            next_node=new_state.next[0] if new_state.next else None
        )
    except Exception as e:
        logger.error(f"Error in chat for session {payload.session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/approve/{session_id}", 
    response_model=ChatResponse, 
    tags=["orchestration"], 
    summary="Approve Interrupted Workflow"
)
async def approve(session_id: str, current_user: Annotated[str, Depends(get_current_user)]):
    """
    Resumes the workflow for a session that is currently interrupted.
    """
    config = {"configurable": {"thread_id": session_id}}
    try:
        # Retrieve the current state to verify ownership
        current_state = await orchestrator.aget_state(config)

        # Check if the session exists at all
        if not current_state:
            logger.warning(f"Attempt to approve non-existent session {session_id} by user {current_user}.")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found or has expired."
            )

        # Verify that the session's user_id matches the authenticated user from config
        config_data = current_state.config.get("configurable", {})
        owner_id = config_data.get("user_id")
        if owner_id != current_user:
            logger.warning(f"User {current_user} attempted to approve session {session_id} owned by {owner_id}.")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Session {session_id} does not belong to user {current_user}."
            )

        # Passing None to ainvoke tells LangGraph to resume from the last checkpoint
        final_output = await orchestrator.ainvoke(None, config)
        new_state = await orchestrator.aget_state(config)
        last_message = final_output["messages"][-1]

        return ChatResponse(
            content=last_message.content,
            session_id=session_id,
            requires_action=len(new_state.next) > 0,
            next_node=new_state.next[0] if new_state.next else None
        )
    except Exception as e:
        # Re-raise specific HTTPExceptions, otherwise wrap in a generic 500
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Error in approve for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")