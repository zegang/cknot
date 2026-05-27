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