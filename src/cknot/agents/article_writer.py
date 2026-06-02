import logging
import re
import contextvars
import operator
from datetime import datetime
from typing import List, Any, Dict, Union, Optional, TypedDict, Annotated
from pydantic import Field
from langgraph.graph import StateGraph, END
from langgraph.graph.state import RunnableConfig, CompiledStateGraph
from langgraph.constants import Send
from langgraph.graph.message import add_messages
from cknot.schemas.state import CknotAgentState, CKnotConfig
from cknot.agents.base import CKnotBaseAgent
from cknot.agents.system_prompts import (
    ARTICLE_WRITER_PROMPT, ARTICLE_PLANNER_PROMPT, ARTICLE_RESEARCHER_PROMPT,
    ARTICLE_DRAFTER_PROMPT, ARTICLE_EDITOR_PROMPT, ARTICLE_REFINER_PROMPT,
    ARTICLE_SUMMARIZER_PROMPT
)
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langgraph.prebuilt import ToolNode

logger = logging.getLogger(__name__)

# Context variable to hold the task-local system prompt during parallel execution
_current_prompt: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("article_writer_prompt", default=None)

class ArticleWriterState(TypedDict):
    """Specific state schema for the Article Writer sub-graph."""
    messages: Annotated[List[BaseMessage], add_messages]
    topic: str
    outline: str
    # Dict of section names to retrieved facts. Uses ior (|) to merge results in Map-Reduce.
    research_data: Annotated[Dict[str, str], operator.ior]
    draft: str
    feedback: str
    iteration_count: int
    agent_summary: Annotated[Dict[str, Dict[str, Any]], operator.ior]

class ArticleWriterAgent(CKnotBaseAgent):
    """
    Specialist agent for complex, multi-stage article writing.
    Handles planning, research, drafting, editing, and refining.
    """
    system_prompt: str = Field(default=ARTICLE_WRITER_PROMPT)
    good_at: List[str] = Field(default_factory=lambda: ["complex article writing", "long-form content", "detailed outlines", "editorial review"])
    poor_at: List[str] = Field(default_factory=lambda: ["code debugging", "system log analysis", "real-time chat support"])
    refine_counts: int = Field(default=1)

    def _get_messages(self, state: Union[CknotAgentState, ArticleWriterState]) -> List[BaseMessage]:
        """
        Overrides base message generation to use a task-local system prompt if set.
        Ensures that parallel research tasks do not interfere with each other's prompts.
        """
        # Use the context-local prompt if available, otherwise fallback to the default
        prompt = _current_prompt.get() or self.system_prompt
        
        # Filter out delegation/trigger messages to prevent the LLM from thinking it has already spoken.
        # We primarily want the system instructions and the original human intent.
        all_messages = state.get("messages", []) if isinstance(state, dict) else state.messages
        filtered_messages = [
            m for m in all_messages 
            if not (isinstance(m, AIMessage) and "TRIGGER_" in m.content)
        ]

        # Best Practice: Ensure local LLM turn finality with a Human prompt
        full_messages = [SystemMessage(content=prompt)] + filtered_messages
        if full_messages and isinstance(full_messages[-1], AIMessage):
            full_messages.append(
                HumanMessage(content="Please proceed with the assigned writing/planning task based on the context above.")
            )
        return full_messages

    def _create_node(self, prompt: str, output_key: Optional[str] = None):
        """Helper to create a node function for a specific phase."""
        async def node(state: ArticleWriterState, config: RunnableConfig):
            section_context = ""
            if isinstance(state, dict) and "section" in state:
                section_context = f"\n\nCURRENT SECTION TO RESEARCH: {state['section']}"

            # Set the context variable for the duration of this task's execution
            token = _current_prompt.set(f"{prompt}{section_context}")
            try:
                result = await self.ainvoke(state, config)
                
                # Prepare the update dictionary
                last_msg_content = result["messages"][-1].content
                update = {"messages": result["messages"]}
                
                if output_key:
                    update[output_key] = last_msg_content
                
                return update
            finally:
                _current_prompt.reset(token)
        return node

    def _map_research_tasks(self, state: ArticleWriterState) -> List[Send]:
        """
        Map Logic: Parses the outline from the planner's response 
        and creates parallel research tasks for each heading.
        """
        last_message = state["messages"][-1].content
        # Simple regex to find headings (e.g., "1. Introduction" or "## Section")
        sections = re.findall(r'(?:^|\n)(?:\d+\.|\#+)\s*(.*)', last_message)
        
        if not sections:
            logger.warning("ArticleWriter could not parse outline sections. Falling back to single research task.")
            return [Send("researcher", {**state, "section": "General Overview"})]

        logger.info(f"ArticleWriter mapping {len(sections)} research tasks.")
        return [
            Send("researcher", {**state, "section": section_desc}) 
            for section_desc in sections
        ]

    def get_subgraph(self) -> CompiledStateGraph:
        """
        Constructs the 5-stage sub-graph for article writing.
        """
        workflow = StateGraph(ArticleWriterState, config_schema=CKnotConfig)

        # Define nodes for each stage
        workflow.add_node("planner", self._create_node(ARTICLE_PLANNER_PROMPT, "outline"))
        workflow.add_node("summarizer", self._create_node(ARTICLE_SUMMARIZER_PROMPT, "research_data"))
        workflow.add_node("drafter", self._create_node(ARTICLE_DRAFTER_PROMPT, "draft"))
        workflow.add_node("editor", self._create_node(ARTICLE_EDITOR_PROMPT, "feedback"))

        # Specialized Refiner node to increment iteration_count
        async def refiner_node(state: ArticleWriterState, config: RunnableConfig):
            node_func = self._create_node(ARTICLE_REFINER_PROMPT, "draft")
            result = await node_func(state, config)
            result["iteration_count"] = state.get("iteration_count", 0) + 1
            return result
            
        workflow.add_node("refiner", refiner_node)

        # Specialized Researcher node for Map-Reduce (handles nested section state)
        async def researcher_node(state: Dict, config: RunnableConfig):
            # Set the context-local prompt for the parallel task
            section_name = state.get("section", "General Research")
            token = _current_prompt.set(f"{ARTICLE_RESEARCHER_PROMPT}\n\nSECTION: {section_name}")
            try:
                # invoke the LLM logic
                result = await self.ainvoke(state, config)
                # Return data to be merged into ArticleWriterState via Annotated operators
                return {
                    "research_data": {section_name: result["messages"][-1].content},
                    "messages": result["messages"]
                }
            finally:
                _current_prompt.reset(token)

        workflow.add_node("researcher", researcher_node)
        workflow.add_node("writer_tools", ToolNode(self.tools))

        async def final_summarizer_node(state: ArticleWriterState, config: RunnableConfig):
            """Generates a summary of the draft to return to the main graph."""
            prompt = f"Summarize the following article in two sentences:\n\n{state['draft']}"
            # Re-use the agent's LLM to generate the summary
            result = await self.ainvoke({"messages": [HumanMessage(content=prompt)]}, config)
            summary = result["messages"][-1].content
            return {
                "agent_summary": {
                    "article_writer": {
                        "content": summary, 
                        "status": "SUCCESS",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                }
            }

        workflow.add_node("final_summarizer", final_summarizer_node)

        # Define internal transitions
        async def init_node(state: ArticleWriterState):
            # Initialize the specialized state fields
            # Look for the actual user request in the message history
            topic = state.get("topic")
            if not topic:
                human_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
                topic = human_msgs[-1].content if human_msgs else "General Topic"
            
            return {"topic": topic, "iteration_count": 0, "research_data": {}}

        workflow.add_node("init", init_node)
        workflow.set_entry_point("init")
        workflow.add_edge("init", "planner")

        # Transition from Planner to Researcher uses the Map-Reduce pattern (Send)
        workflow.add_conditional_edges("planner", self._map_research_tasks, ["researcher"])
        # Standard edge from planner to summarizer acts as the synchronization/join point
        workflow.add_edge("planner", "summarizer")

        def researcher_router(state: Dict):
            """Routes between tools and the completion of a parallel branch."""
            last_msg = state["messages"][-1]
            return "writer_tools" if getattr(last_msg, "tool_calls", None) else END

        workflow.add_conditional_edges("researcher", researcher_router)
        workflow.add_edge("writer_tools", "researcher")
        
        workflow.add_edge("summarizer", "drafter")
        workflow.add_edge("drafter", "editor")

        def editor_router(state: ArticleWriterState):
            # Check loop exit conditions
            if state.get("iteration_count", 0) >= self.refine_counts or "APPROVED" in state.get("feedback", "").upper():
                return "final_summarizer"
            return "refiner"

        workflow.add_conditional_edges("editor", editor_router)
        workflow.add_edge("refiner", "editor") # Loop back for editorial review
        workflow.add_edge("final_summarizer", END)

        return workflow.compile()
