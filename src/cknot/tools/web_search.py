import logging
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from ddgs import DDGS

logger = logging.getLogger(__name__)

@tool
def web_search(query: str, config: RunnableConfig):
    """
    Search the web for real-time information using DuckDuckGo. 
    Optimized for AI agents to retrieve concise, relevant, and factual content.
    """
    try:
        results = []
        # We map the results to match the 'url' and 'content' schema expected by the agents.
        # Increasing timeout to 20 seconds to allow for engine failover overhead.
        with DDGS(timeout=20) as ddgs:
            ddgs_gen = ddgs.text(query, max_results=5, region="wt-wt", safesearch="moderate")
            for r in ddgs_gen:
                results.append({
                    "url": r.get("href"),
                    "content": r.get("body"),
                    "title": r.get("title")
                })
        
        if not results:
            logger.warning(f"Web search returned 0 results for query: {query}")
            return "No relevant information found on the web at this time."

        logger.debug(f'Web search results for query "{query}": {results}')
        return results
    except Exception as e:
        logger.error(f"DuckDuckGo search failed for query '{query}': {e}", exc_info=True)
        return f"Error: The web search service encountered an issue. Details: {str(e)}"
