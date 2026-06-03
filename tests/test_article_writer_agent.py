import pytest
import re
import sys
import os
from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Inject the plugins directory into sys.path so the agent can be imported for testing
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
plugins_path = os.path.join(project_root, "plugins")
if plugins_path not in sys.path:
    sys.path.insert(0, plugins_path)

from agents.article_writer import ArticleWriterAgent, ArticleWriterState

@pytest.fixture
def agent():
    """Fixture to provide a clean instance of the ArticleWriterAgent for each test."""
    # We initialize with empty tools to avoid dependency on external tool definitions
    return ArticleWriterAgent(tools=[])

def test_article_writer_metadata(agent):
    """Verify the agent identifies itself and its capabilities correctly."""
    assert agent.name == "ArticleWriterAgent"
    assert "article paper or context summarization and writing" in agent.expert_in
    assert "long-form content" in agent.expert_in
    assert "code debugging" in agent.avoid_for

def test_article_writer_subgraph_construction(agent):
    """Ensure the subgraph builds without errors and contains all required workflow nodes."""
    subgraph = agent.get_subgraph()
    nodes = subgraph.get_graph().nodes
    # Verify all stages of the ArticleWriter pipeline are present
    expected_nodes = [
        "init", "analyst", "planner", "researcher", 
        "summarizer", "drafter", "editor", "refiner", 
        "saver", "save_confirmation", "final_summarizer"
    ]
    for node in expected_nodes:
        assert node in nodes, f"Node '{node}' is missing from the agent subgraph."

def test_source_path_extraction_logic():
    """Verify the robust regex used to extract local document/paper paths from user input."""
    # This regex mimics the one found in ArticleWriterAgent's init_node
    source_regex = r'(?:docs?|refs?|documents?|papers?|files?)(?:\s+(?:docs?|refs?|documents?|papers?|files?))*\s*[:\s]?\s*(?:["\']([^"\']+)["\']|([^\s,]+))'
    
    test_cases = [
        ("Write about AI, docs: paper.pdf", ["paper.pdf"]),
        ("Analyze document 'notes on agents.txt'", ["notes on agents.txt"]),
        ("Read paper document \"Architecture Analysis.pdf\"", ["Architecture Analysis.pdf"]),
        ("ref: data.csv", ["data.csv"]),
    ]
    
    for content, expected in test_cases:
        matches = list(re.finditer(source_regex, content, re.IGNORECASE))
        paths = [m.group(1) or m.group(2) for m in matches]
        assert paths == expected

def test_map_research_tasks_logic(agent):
    """Verify the Map-Reduce logic that splits an outline into parallel researcher tasks."""
    state = {
        "messages": [AIMessage(content="1. Introduction\n2. Technical Deep Dive\n3. Conclusion")]
    }
    sends = agent._map_research_tasks(state)
    
    assert len(sends) == 3
    assert sends[0].node == "researcher"
    assert sends[0].arg["section"] == "Introduction"
    assert sends[1].arg["section"] == "Technical Deep Dive"
    assert sends[2].arg["section"] == "Conclusion"

@pytest.mark.asyncio
async def test_get_messages_context_injection(agent):
    """Verify that source material (RAG) context is correctly injected into the system prompt."""
    state = {
        "messages": [HumanMessage(content="Write an article")],
        "agent_data": {
            "analyst_source_material": "Context extracted from local PDF papers."
        }
    }
    messages = agent._get_messages(state)
    
    # The system prompt is usually the first message
    system_msg = messages[0]
    assert isinstance(system_msg, SystemMessage)
    assert "[LOCAL SOURCE CONTEXT]" in system_msg.content
    assert "Context extracted from local PDF papers." in system_msg.content

def test_article_writer_state_keys():
    """Ensure the ArticleWriterState contains the necessary keys for multi-stage execution."""
    # This test is just to document requirements for the TypedDict
    assert "research_data" in ArticleWriterState.__annotations__
    assert "progress_report" in ArticleWriterState.__annotations__