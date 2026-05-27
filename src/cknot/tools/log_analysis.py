from typing import Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
import os

class LogSearchInput(BaseModel):
    """Input schema for the log search tool."""
    file_path: str = Field(description="The absolute path to the log file on disk.")
    keyword: str = Field(description="The keyword, error code, or regex pattern to search for.")
    context_lines: int = Field(default=3, description="Number of lines to show before and after the match.")

class LogSearchTool(BaseTool):
    """A tool for performing contextual searches within log files."""
    name: str = "log_search"
    description: str = "Searches a log file for a specific keyword and returns the match with surrounding context lines."
    args_schema: Type[BaseModel] = LogSearchInput

    def _run(self, file_path: str, keyword: str, context_lines: int = 3) -> str:
        """Synchronous implementation of the tool."""
        if not os.path.exists(file_path):
            return f"Error: File {file_path} not found."
        
        try:
            results = []
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if keyword.lower() in line.lower():
                        start = max(0, i - context_lines)
                        end = min(len(lines), i + context_lines + 1)
                        context = "".join(lines[start:end])
                        results.append(f"--- Match at line {i+1} ---\n{context}")
            return "\n".join(results) if results else "No matches found."
        except Exception as e:
            return f"Error searching log: {str(e)}"