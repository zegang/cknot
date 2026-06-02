import os
import logging
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

@tool
def read_log_file(file_path: str) -> str:
    """Reads the contents of a log file from the disk."""
    try:
        if not os.path.exists(file_path):
            return f"Error: Log file at {file_path} not found."
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

@tool
def write_file(file_path: str, content: str, append: bool = False) -> str:
    """Writes or appends content to a local file."""
    try:
        mode = "a" if append else "w"
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, mode, encoding="utf-8") as f:
            f.write(content)
        return f"Successfully {'appended to' if append else 'wrote to'} {file_path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"