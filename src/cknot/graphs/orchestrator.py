import os
import logging
from typing import Dict, List, Any, Optional
from redis.asyncio import Redis as AsyncRedis
from cknot.agents.boss import CKnotBossAgent
from cknot.agents.code_fixer import CodeFixerAgent
from cknot.agents.deep_search import DeepSearchAgent
from cknot.agents.article_writer import ArticleWriterAgent
from cknot.agents.log_parser import LogParserAgent
from cknot.utils.llm_manager import LLMManager
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from cknot.config.config import settings
from cknot.tools.tool_manager import ToolManager
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.redis import AsyncRedisSaver
from cknot.schemas.state import CknotAgentState, CKnotConfig
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

class GraphOrchestrator:
    """
    Handles the construction and compilation of the LangGraph workflow.
    """
    def __init__(self):
        self.llm_manager = LLMManager()
        self.tool_manager = ToolManager()
        self.agents: Dict[str, Any] = {}
        self.tools: List[Any] = []

    def _setup_tools(self):
        """Registers and retrieves all system tools."""
        self.tool_manager.register_tool_instance("web_search", web_search)
        self.tool_manager.register_tool_instance("read_log_file", read_log_file)
        self.tool_manager.register_tool_instance("wikipedia", WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()))
        self.tool_manager.register_tool_instance("log_search", LogSearchTool())
        self.tools = self.tool_manager.get_runnable_tools()

    def _setup_agents(self):
        """Initializes agents and registers them with the AgentRegistry."""
        def_llm_svc = self.llm_manager.get_llm_service(settings.DEFAULT_LLM_SERVICE)

        deep_search = DeepSearchAgent()
        log_parser = LogParserAgent()
        code_fixer = CodeFixerAgent()
        article_writer = ArticleWriterAgent(tools=self.tools)
        boss = CKnotBossAgent(
            name="cknot", 
            llm_services=[def_llm_svc], 
            tools=self.tools, 
            sub_agents=[deep_search, log_parser, code_fixer, article_writer]
        )

        self.agents = {
            "cknot": boss,
            "deep_search": deep_search,
            "log_parser": log_parser,
            "code_fixer": code_fixer,
            "article_writer": article_writer
        }

        for agent in self.agents.values():
            AgentRegistry.register_agent(agent)

    @staticmethod
    def _boss_router(state: CknotAgentState):
        """Routing logic for the central orchestrator."""
        # Specifically look for the last message from the orchestrator (cknot)
        last_message = state["messages"][-1]
        
        logger.debug(f'Agent Graph Node CKnot Routing Check: {last_message}')
        if getattr(last_message, "tool_calls", None):
            return "tools"
            
        content = last_message.content.upper() if last_message.content else ""
        if "TRIGGER_LOG_ANALYSIS" in content:
            return "log_parser"
        if "TRIGGER_DEEP_SEARCH" in content:
            return "deep_search"
        if "TRIGGER_ARTICLE_WRITING" in content:
            return "article_writer"
        return END

    @staticmethod
    def _search_router(state: CknotAgentState):
        """Routing logic for the deep search specialist."""
        last_message = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), state["messages"][-1])
        logger.debug(f'Agent Graph Node DeepSearch: {last_message}')
        return "tools" if getattr(last_message, "tool_calls", None) else "cknot"

    def build(self) -> CompiledStateGraph:
        """Assembles and compiles the graph."""
        self._setup_tools()
        self._setup_agents()

        workflow = StateGraph(CknotAgentState, config_schema=CKnotConfig)

        # Define node functions using agent ainvoke methods
        workflow.add_node("cknot", self.agents["cknot"].ainvoke)
        workflow.add_node("log_parser", self.agents["log_parser"].ainvoke)
        workflow.add_node("code_fixer", self.agents["code_fixer"].ainvoke)
        workflow.add_node("deep_search", self.agents["deep_search"].ainvoke)
        workflow.add_node("article_writer", self.agents["article_writer"].get_subgraph())
        workflow.add_node("tools", ToolNode(self.tools))

        # Define graph flow
        workflow.set_entry_point("cknot")
        workflow.add_conditional_edges("cknot", self._boss_router)
        workflow.add_edge("tools", "cknot")
        workflow.add_edge("log_parser", "code_fixer")
        workflow.add_edge("code_fixer", END)
        workflow.add_conditional_edges("deep_search", self._search_router)
        
        # After the sub-graph finishes, it returns to the Boss for final delivery
        workflow.add_edge("article_writer", "cknot")

        # Initialize persistence
        if settings.CHECKPOINTER_TYPE == "redis":
            redis_client = AsyncRedis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=False)
            memory = AsyncRedisSaver(redis_client=redis_client)
        else:
            memory = MemorySaver()

        return workflow.compile(
            checkpointer=memory,
            interrupt_before=["tools", "log_parser", "deep_search"]
        )

    def visualize_ascii(self):
        """
        Prints an ASCII representation of the graph to the console.
        Useful for quick debugging directly in the terminal.
        """
        compiled_graph = self.build()
        # print_ascii() outputs the graph structure using text characters
        compiled_graph.get_graph().print_ascii()

def create_graph() -> CompiledStateGraph:
    """
    Compatibility factory function that builds and returns the graph.
    """
    return GraphOrchestrator().build()
