import sys
import os

def clear_lines(n: int) -> None:
    """
    Moves the cursor up by n lines and clears everything from that point to the end of the screen.
    
    Note: This uses ANSI escape sequences which are supported natively on macOS, Linux,
    and modern Windows (10+).

    Args:
        n: The number of lines to move up before clearing.
    """
    if n > 0:
        # \033[{n}A: Move cursor up n lines
        # \r: Move cursor to the beginning of the line
        # \033[J: Clear from cursor to the end of the screen
        sys.stdout.write(f"\x1b[{n}A\r\x1b[J")
        sys.stdout.flush()
