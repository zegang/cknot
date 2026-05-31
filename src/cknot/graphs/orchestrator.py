import os
import logging
from redis.asyncio import Redis as AsyncRedis
from cknot.agents.boss import CKnotBossAgent
from cknot.agents.code_fixer import CodeFixerAgent
from cknot.agents.deep_search import DeepSearchAgent
from cknot.agents.log_parser import LogParserAgent
from cknot.utils.llm_manager import LLMManager
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from cknot.config.config import settings
from cknot.tools.tool_manager import ToolManager
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.redis import AsyncRedisSaver
from cknot.schemas.state import AgentState, CKnotConfig
from cknot.tools.web_search import web_search
from cknot.tools.file_ops import read_log_file
from cknot.tools.log_analysis import LogSearchTool # Assuming this is still a function-based tool
from cknot.agents.registry import AgentRegistry
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

logger = logging.getLogger(__name__)

def create_graph() -> CompiledStateGraph:
    # 1. Initialize LLM with Tools
    llm_manager = LLMManager()
    tool_manager = ToolManager()

    # 2. Register Tools with the Manager
    tool_manager.register_tool_instance("web_search", web_search)
    tool_manager.register_tool_instance("read_log_file", read_log_file)
    tool_manager.register_tool_instance("wikipedia", WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()))
    tool_manager.register_tool_instance("log_search", LogSearchTool())

    # Retrieve only enabled tools for the graph
    tools = tool_manager.get_runnable_tools()
    def_llm_svc = llm_manager.get_llm_service(settings.DEFAULT_LLM_SERVICE)

    # Instantiate Class-based Agents
    deep_search_agent = DeepSearchAgent()
    log_parser_agent = LogParserAgent()
    code_fixer_agent = CodeFixerAgent()
    boss_agent = CKnotBossAgent(name="cknot", llm_services=[def_llm_svc], tools=tools, 
                               sub_agents=[deep_search_agent, log_parser_agent, code_fixer_agent])
    AgentRegistry.register_agent(deep_search_agent)
    AgentRegistry.register_agent(log_parser_agent)
    AgentRegistry.register_agent(code_fixer_agent)
    AgentRegistry.register_agent(boss_agent)

    # 3. Build the Graph
    workflow = StateGraph(AgentState, config_schema=CKnotConfig)

    # Define node functions that await the agent's ainvoke method
    async def cknot_node(state: AgentState, config: RunnableConfig):
        return await boss_agent.ainvoke(state, config)

    async def log_parser_node(state: AgentState, config: RunnableConfig):
        return await log_parser_agent.ainvoke(state, config)

    async def code_fixer_node(state: AgentState, config: RunnableConfig):
        return await code_fixer_agent.ainvoke(state, config)

    async def deep_search_node(state: AgentState, config: RunnableConfig):
        return await deep_search_agent.ainvoke(state, config)

    workflow.add_node("cknot", cknot_node)
    workflow.add_node("log_parser", log_parser_node)
    workflow.add_node("code_fixer", code_fixer_node)
    workflow.add_node("deep_search", deep_search_node)
    workflow.add_node("tools", ToolNode(tools))

    # Define Edges
    workflow.set_entry_point("cknot")
    
    def should_continue(state: AgentState):
        """Router logic for the cknot boss."""
        # Retrieve the last AI message to avoid issues with trailing tool or metadata fragments
        last_message = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), state["messages"][-1])
        
        logger.debug(f'Agent Graph Node CKnot Routing Check: {last_message}')
        if getattr(last_message, "tool_calls", None):
            return "tools"
            
        content = last_message.content.upper() if last_message.content else ""
        if "TRIGGER_LOG_ANALYSIS" in content:
            return "log_parser"
        if "TRIGGER_DEEP_SEARCH" in content:
            return "deep_search"
        return END

    workflow.add_conditional_edges("cknot", should_continue)
    workflow.add_edge("tools", "cknot")

    # Debugging Workflow: Parser -> Fixer
    workflow.add_edge("log_parser", "code_fixer")
    workflow.add_edge("code_fixer", END)
    
    # Deep Search Workflow
    def should_continue_search(state: AgentState):
        # Retrieve the last AI message to check for tool calls
        last_message = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), state["messages"][-1])
        logger.debug(f'Agent Graph Node DeepSearch: {last_message}')
        return "tools" if getattr(last_message, "tool_calls", None) else "cknot"

    workflow.add_conditional_edges("deep_search", should_continue_search)

    # 4. Compile with Memory and Interruption
    if settings.CHECKPOINTER_TYPE == "redis":
        # LangGraph requires a binary-safe connection (decode_responses=False).
        # We use a dedicated client instance to avoid conflicts with the global singleton.
        redis_client = AsyncRedis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=False)
        memory = AsyncRedisSaver(redis_client=redis_client)
    else:
        memory = MemorySaver()

    return workflow.compile(
        checkpointer=memory,
        interrupt_before=["tools", "log_parser", "deep_search"]  # Human confirm before potentially costly actions
    )
