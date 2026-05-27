import os
import re
import logging
from logging.handlers import TimedRotatingFileHandler
from contextvars import ContextVar

# Context variable to store the current user_id for logging
user_id_ctx = ContextVar("user_id", default="system")

class UserContextFilter(logging.Filter):
    """Filter that injects user_id from context into log records."""
    def filter(self, record: logging.LogRecord) -> bool:
        record.user_id = user_id_ctx.get()
        return True

class DynamicUserRotatingFileHandler(logging.Handler):
    """Handler that routes logs to user-specific files based on record.user_id."""
    def __init__(self, log_dir: str, backup_count: int = 30):
        super().__init__()
        self.log_dir = log_dir
        self.backup_count = backup_count
        self._handlers = {}

    def emit(self, record: logging.LogRecord):
        try:
            user_id = getattr(record, "user_id", "system")
            # Sanitize user_id for filename safety
            safe_user_id = re.sub(r'[^\w\-]', '_', user_id)
            
            if safe_user_id not in self._handlers:
                log_file = os.path.join(self.log_dir, f"user_{safe_user_id}.log")
                handler = TimedRotatingFileHandler(
                    log_file, when="midnight", interval=1,
                    backupCount=self.backup_count, encoding="utf-8"
                )
                if self.formatter:
                    handler.setFormatter(self.formatter)
                self._handlers[safe_user_id] = handler
            
            self._handlers[safe_user_id].emit(record)
        except Exception:
            self.handleError(record)

    def close(self):
        for handler in self._handlers.values():
            handler.close()
        super().close()

class RedactingFilter(logging.Filter):
    """Filter that masks sensitive information in log records using regex."""
    def __init__(self, patterns: list[str]):
        super().__init__()
        self._patterns = [re.compile(p, re.IGNORECASE) for p in patterns]

    def filter(self, record: logging.LogRecord) -> bool:
        msg = str(record.msg)
        for pattern in self._patterns:
            # Replaces the value group (group 3) with asterisks
            msg = pattern.sub(r"\1\2********", msg)
        record.msg = msg
        return True

def setup_logging(log_dir: str = "logs", log_level: int = logging.DEBUG):
    """Initializes the global logging configuration with rotation and redaction."""
    os.makedirs(log_dir, exist_ok=True)

    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Define patterns to mask: keys like password, api_key, secret, etc.
    sensitive_patterns = [
        r"(password|api_key|secret|token|access_token)(['\" :\s=]+)([^'\"\s&,]+)"
    ]
    redactor = RedactingFilter(sensitive_patterns)
    user_filter = UserContextFilter()

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Only show critical info to user
    console_handler.setFormatter(log_formatter)
    console_handler.addFilter(redactor)
    console_handler.addFilter(user_filter)

    # Dynamic User-specific File Handler
    file_handler = DynamicUserRotatingFileHandler(log_dir, backup_count=30)
    file_handler.setLevel(log_level)  # File gets INFO or DEBUG as configured
    file_handler.setFormatter(log_formatter)
    file_handler.addFilter(redactor)
    file_handler.addFilter(user_filter)

    # Root Logger setup
    root_logger = logging.getLogger()
    root_logger.setLevel(min(log_level, logging.INFO))
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Silence Noisy Third-Party Libraries
    logging.getLogger("langgraph").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("markdown_it").setLevel(logging.WARNING)