# This module defines common constants and enums across the project.

import enum

class LlmNames(enum.Enum):
    """
    # LLM Names
    This is an enumeration of the LLM names.
    """
    GEMMA = "gemma"
    LLAMA = "llama"