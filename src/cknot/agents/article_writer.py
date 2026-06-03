import logging
import os
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
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from cknot.tools.knowledge_base import LlamaIndexRetrieverTool
from cknot.tools.file_ops import write_file
from langgraph.prebuilt import ToolNode

logger = logging.getLogger(__name__)

ARTICLE_WRITER_PROMPT = (
    "You are an Elite Content Strategist and Article Writer. You produce authoritative, long-form content.\n"
    "Your workflow MUST follow these stages:\n"
    "1. ANALYST: Ingest and summarize local documents or research papers if paths are provided by the user. Use this context to inform the planning and drafting phases.\n"
    "2. PLANNER: Create a comprehensive outline with headings and sub-points based on the topic and source material.\n"
    "3. RESEARCHER: Identify key facts, data points, or information needed for each section (use tools if required).\n"
    "4. DRAFTER: Write the content section-by-section, ensuring consistency with the outline, research, and source material.\n"
    "5. EDITOR: Critique the draft for clarity, SEO, and structural integrity.\n"
    "6. REFINER: Finalize the article by incorporating edits and ensuring it meets high professional standards.\n"
    "   If the user has not specified a file path to save the article, suggest a suitable filename and ask if they would like to save it locally.\n"
    "7. SAVER: Use the 'write_file' tool to store or append the final content to a local file if a path is provided.\n"
    "Explicitly mention which stage you are currently in during the process."
)

ARTICLE_ANALYST_PROMPT = "You are a Document Analyst. Extract and summarize the most relevant information from the provided local source documents to support the article topic."

ARTICLE_PLANNER_PROMPT = "You are a Content Planner. Analyze the topic and generate a detailed outline with headings and sub-points."

ARTICLE_RESEARCHER_PROMPT = (
    "You are a Fact Researcher. For each section of the provided outline, retrieve key data, facts, and "
    "supporting information using your available tools."
)

ARTICLE_DRAFTER_PROMPT = (
    "You are a Content Drafter. Write the full article section-by-section based on the outline and research results. "
    "Maintain a consistent professional tone."
)

ARTICLE_EDITOR_PROMPT = "You are a Senior Editor. Critique the draft for flow, clarity, SEO, and factual accuracy. Provide specific feedback for the refiner."

ARTICLE_REFINER_PROMPT = (
    "You are a Content Refiner. Rewrite the article by strictly following the Editor's critique and polishing the final prose. "
    "Additionally, if the user has not yet specified a file path to save the article, suggest a suitable filename "
    "and ask if they would like to save it locally."
)

ARTICLE_SUMMARIZER_PROMPT = (
    "You are a Research Synthesizer. You will be provided with multiple research reports for different sections of an article. "
    "Your task is to consolidate these reports into a single, cohesive summary that highlights the most critical facts, "
    "data points, and quotes for the drafter. Ensure no information is lost, but remove redundancies."
)

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
    total_sections: int
    output_file_path: Optional[str]
    append_file: bool
    is_save_only: bool
    agent_summary: Annotated[Dict[str, Dict[str, Any]], operator.ior]
    source_paths: List[str]
    source_material: str
    progress_report: Annotated[Dict[str, Dict[str, Any]], operator.ior]

class ArticleWriterAgent(CKnotBaseAgent):
    """
    Specialist agent for complex, multi-stage article writing.
    Handles planning, research, drafting, editing, and refining.
    """
    system_prompt: str = Field(default=ARTICLE_WRITER_PROMPT)
    expert_in: List[str] = Field(default_factory=lambda: ["aritcle paper or context summarization and writing", "long-form content", "detailed outlines", "editorial review", "saving articles to local files"])
    avoid_for: List[str] = Field(default_factory=lambda: ["code debugging", "system log analysis", "real-time chat support"])
    refine_counts: int = Field(default=1)

    def _get_messages(self, state: Union[CknotAgentState, ArticleWriterState]) -> List[BaseMessage]:
        """
        Overrides base message generation to use a task-local system prompt if set.
        Ensures that parallel research tasks do not interfere with each other's prompts.
        """
        # Use the context-local prompt if available, otherwise fallback to the default
        prompt = _current_prompt.get() or self.system_prompt
        
        # Inject source material context if available
        source_material = state.get("source_material")
        if source_material:
            prompt += f"\n\n[LOCAL SOURCE CONTEXT]\n{source_material}\n[END SOURCE CONTEXT]"

        # Filter out delegation/trigger messages to prevent the LLM from thinking it has already spoken.
        # We primarily want the system instructions and the original human intent.
        all_messages = state.get("messages", []) if isinstance(state, dict) else state.messages
        filtered_messages = [
            m for m in all_messages 
            if not (isinstance(m, AIMessage) and re.search(r'Agent\s+\w+', m.content))
        ]

        # Best Practice: Ensure local LLM turn finality with a Human prompt
        full_messages = [SystemMessage(content=prompt)] + filtered_messages
        if full_messages and isinstance(full_messages[-1], AIMessage):
            full_messages.append(
                HumanMessage(content="Please proceed with the assigned writing/planning task based on the context above.")
            )
        return full_messages

    def _create_node(self, prompt: str, output_key: Optional[str] = None, progress_desc: Optional[str] = None, step_name: str = "WORKING"):
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
                usage = getattr(result["messages"][-1], "usage_metadata", None)
                total_tokens = usage.get("total_tokens", 0) if usage else 0
                update = {
                    "messages": result["messages"],
                    "progress_report": {
                        f"{self.name.lower()}_{step_name.lower()}": {
                            "step": step_name,
                            "description": progress_desc or f"{self.name} is working...",
                            "status": "done",
                            "percentage": 100.0,
                            "total_tokens": total_tokens
                        }
                    }
                }
                
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
        async def analyst_node(state: ArticleWriterState, config: RunnableConfig):
            """Stage 1: ANALYST - Reads and summarizes local documents."""
            paths = state.get("source_paths", [])
            if not paths:
                return {"source_material": "No source documents provided."}

            # Use the first path; the LlamaIndexRetrieverTool handles path resolution
            source_path = paths[0]
            session_id = config["configurable"].get("thread_id", "default_session")
            storage_dir = f"article_writer/{session_id}"
            
            token = _current_prompt.set(ARTICLE_ANALYST_PROMPT)
            try:
                # Initialize the LlamaIndex-based retriever tool with session-specific paths
                rag_tool = LlamaIndexRetrieverTool(data_path=source_path, storage_dir=storage_dir)

                # Perform a targeted RAG query to extract relevant context for the topic
                query = f"Analyze the source documents and extract all key facts, technical data, and relevant context to support writing an article about: {state['topic']}"
                source_material = await rag_tool._arun(query)

                # Use the LLM with the ARTICLE_ANALYST_PROMPT to refine and summarize the RAG output
                inputs = {"messages": [HumanMessage(content=f"Topic: {state['topic']}\n\nRetrieved Context from Documents:\n{source_material}")]}
                result = await self.ainvoke(inputs, config)
                final_analysis = result["messages"][-1].content

                return {
                    "source_material": final_analysis,
                    "messages": result["messages"],
                    "progress_report": {
                        "analyst": {
                            "step": "ANALYSIS",
                            "description": f"RAG analysis complete for {source_path}",
                            "status": "done",
                            "percentage": 100.0
                        }
                    }
                }
            except Exception as e:
                logger.error(f"RAG analysis failed: {e}")
                return {"source_material": f"Error during RAG analysis: {str(e)}"}
            finally:
                _current_prompt.reset(token)

        workflow.add_node("analyst", analyst_node)

        async def planner_node(state: ArticleWriterState, config: RunnableConfig):
            token = _current_prompt.set(ARTICLE_PLANNER_PROMPT)
            try:
                result = await self.ainvoke(state, config)
                last_msg_content = result["messages"][-1].content
                usage = getattr(result["messages"][-1], "usage_metadata", None)
                total_tokens = usage.get("total_tokens", 0) if usage else 0
                # Calculate total sections for progress tracking
                sections = re.findall(r'(?:^|\n)(?:\d+\.|\#+)\s*(.*)', last_msg_content)
                total = len(sections) if sections else 1
                return {
                    "messages": result["messages"],
                    "outline": last_msg_content,
                    "total_sections": total,
                    "progress_report": {
                        "planner": {
                            "step": "PLANNING",
                            "description": "Planning article structure...",
                            "total": total,
                            "current": 1
                        }
                    }
                }
            finally:
                _current_prompt.reset(token)

        workflow.add_node("planner", planner_node)

        async def summarizer_node(state: ArticleWriterState, config: RunnableConfig):
            node_func = self._create_node(ARTICLE_SUMMARIZER_PROMPT, progress_desc="Synthesizing research...", step_name="RESEARCH")
            result = await node_func(state, config)
            result["research_data"] = {"consolidated": result["messages"][-1].content}
            return result
            
        workflow.add_node("summarizer", summarizer_node)
        workflow.add_node("drafter", self._create_node(ARTICLE_DRAFTER_PROMPT, "draft", "Drafting content...", step_name="WRITING"))
        workflow.add_node("editor", self._create_node(ARTICLE_EDITOR_PROMPT, "feedback", "Reviewing draft...", step_name="EDITING"))

        # Specialized Refiner node to increment iteration_count
        async def refiner_node(state: ArticleWriterState, config: RunnableConfig):
            node_func = self._create_node(ARTICLE_REFINER_PROMPT, "draft", "Refining prose...", step_name="REFINING")
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
                logger.info(f"Research task completed for section: {section_name}")
                usage = getattr(result["messages"][-1], "usage_metadata", None)
                total_tokens = usage.get("total_tokens", 0) if usage else 0
                # Return data to be merged into ArticleWriterState via Annotated operators
                return {
                    "research_data": {section_name: result["messages"][-1].content},
                    "messages": result["messages"],
                    "progress_report": {
                        f"researcher:{section_name}": {
                            "step": "RESEARCH",
                            "description": f"Completed: {section_name}",
                            "current": 1,
                            "total": 1,
                            "status": "done",
                            "percentage": 100.0,
                            "total_tokens": total_tokens
                        }
                    }
                }
            finally:
                _current_prompt.reset(token)

        workflow.add_node("researcher", researcher_node)
        workflow.add_node("writer_tools", ToolNode(self.tools))

        async def saver_node(state: ArticleWriterState, config: RunnableConfig):
            """Stage 6: SAVER - Automatically saves the draft if a path is detected."""
            path = state.get("output_file_path")
            if path and state.get("draft"):
                result = write_file.invoke({"file_path": path, "content": state["draft"], "append": state.get("append_file", False)})
                return {
                    "messages": [AIMessage(content=f"STAGE 6: SAVER\n{result}")],
                    "progress_report": {
                        "saver": {
                            "step": "SAVING",
                            "description": "Saving to file...",
                            "status": "done",
                            "percentage": 100.0
                        }
                    }
                }
            return {
                "messages": [AIMessage(content="STAGE 6: SAVER\nNo output path provided. Skipping.")],
                "progress_report": {
                    "saver": {
                        "step": "SAVING",
                        "description": "No output path provided. Skipping.",
                        "status": "skipped",
                        "percentage": 100.0
                    }
                }
            }

        workflow.add_node("saver", saver_node)

        async def save_confirmation_node(state: ArticleWriterState, config: RunnableConfig):
            """Stage 6.1: PRE-SAVER - Asks for confirmation."""
            path = state.get("output_file_path")
            return {
                "messages": [AIMessage(content=f"Article finalized. I am ready to save it to `{path}`. Please authorize to proceed.")],
                "progress_report": {
                    "saver": {
                        "step": "SAVING",
                        "description": "Awaiting authorization...",
                        "status": "pending",
                        "percentage": 0.0
                    }
                }
            }

        workflow.add_node("save_confirmation", save_confirmation_node)

        async def final_summarizer_node(state: ArticleWriterState, config: RunnableConfig):
            """Generates a summary of the draft to return to the main graph."""
            # Calculate research progress percentage based on completed sections
            outline = state.get("outline", "")
            sections = re.findall(r'(?:^|\n)(?:\d+\.|\#+)\s*(.*)', outline)
            total_expected = len(sections) if sections else 1
            completed_count = len(state.get("research_data", {}))
            progress_pct = min(100.0, (completed_count / total_expected) * 100)

            prompt = f"Summarize the following article in two sentences:\n\n{state['draft']}"
            # Re-use the agent's LLM to generate the summary
            result = await self.ainvoke({"messages": [HumanMessage(content=prompt)]}, config)
            summary = result["messages"][-1].content
            return {
                "agent_summary": {
                    "article_writer": {
                        "content": summary, 
                        "status": "SUCCESS",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "progress": f"{progress_pct:.1f}%",
                        }
                },
                "progress_report": {
                    "summarizer": {
                        "step": "FINALIZING",
                        "description": "Finalizing report...",
                        "status": "done",
                        "percentage": 100.0
                    }
                }
            }

        workflow.add_node("final_summarizer", final_summarizer_node)

        # Define internal transitions
        async def init_node(state: ArticleWriterState):
            # Initialize the specialized state fields
            human_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
            topic = state.get("topic")
            if not topic:
                topic = human_msgs[-1].content if human_msgs else "General Topic"
            
            # Extract output path if mentioned in the prompt
            output_path = None
            append_file = False
            if human_msgs:
                content = human_msgs[-1].content
                # Check for append intent first
                append_match = re.search(r'append\s+(?:to|as|at)?\s*([^\s]+\.[a-zA-Z0-9]+)', content, re.IGNORECASE)
                save_match = re.search(r'(?:save\s+(?:to|as)|file:)\s*([^\s]+\.[a-zA-Z0-9]+)', content, re.IGNORECASE)

                if append_match:
                    output_path = append_match.group(1)
                    append_file = True
                elif save_match:
                    output_path = save_match.group(1)
            
            # Optimization: If we already have a draft and the user just provided a save path, skip directly to saving.
            is_save_only = False
            if state.get("draft") and output_path:
                 # Check if the last human message was primarily about saving
                 last_msg = human_msgs[-1].content.lower() if human_msgs else ""
                 if any(keyword in last_msg for keyword in ["save", "append", "file:"]):
                     is_save_only = True
            
            # Extract local document paths using a robust regex for keywords and quoted filenames
            source_paths = []
            if human_msgs:
                content = human_msgs[-1].content
                doc_matches = re.finditer(r'(?:docs?|refs?|documents?|papers?|files?)(?:\s+(?:docs?|refs?|documents?|papers?|files?))*\s*[:\s]?\s*(?:["\']([^"\']+)["\']|([^\s,]+))', content, re.IGNORECASE)
                for match in doc_matches:
                    src_path = match.group(1) or match.group(2)
                    if src_path:
                        source_paths.append(src_path.strip())

            return {
                "topic": topic,
                "iteration_count": 0,
                "research_data": {}, 
                "output_file_path": output_path,
                "append_file": append_file,
                "is_save_only": is_save_only,
                "source_paths": source_paths,
                "source_material": "",
                "progress_report": {
                    "init": {
                        "step": "INITIALIZING",
                        "status": "done",
                        "percentage": 100.0
                    }
                }
            }

        workflow.add_node("init", init_node)
        workflow.set_entry_point("init")

        def init_router(state: ArticleWriterState):
            if state.get("is_save_only"):
                return "save_confirmation"
            if state.get("source_paths"):
                return "analyst"
            return "planner"

        workflow.add_conditional_edges("init", init_router)
        
        workflow.add_edge("analyst", "planner")

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
                if state.get("output_file_path"):
                    return "save_confirmation"
                return "final_summarizer"
            return "refiner"

        workflow.add_conditional_edges("editor", editor_router)
        workflow.add_edge("refiner", "editor") # Loop back for editorial review
        workflow.add_edge("save_confirmation", "saver")
        workflow.add_edge("saver", "final_summarizer")
        workflow.add_edge("final_summarizer", END)

        return workflow.compile()
