"""
This module manages LLM connection and sessions.
"""

import abc

# LLamaCpp is a C++ implementation of LLM, which can be run in local machine.
from langchain_community.llms import LlamaCpp

# HuggingfaceTextGenInference is a handy wrapper for Huggingface's text generation API.
from langchain_community.llms.huggingface_text_gen_inference import HuggingfaceTextGenInference 

from langchain_core.language_models.llms import LLM


class LlmConnector(abc.ABC):...


class LlmConnectorLlamacpp(LlmConnector):
    """
    Connect to local LLamaCpp model.
    """
    def __init__(self, model_path: str, verbose: bool = False, max_tokens: int = 2048):

        # Ensure we only need to load the model once.
        self.llm_instance = LlamaCpp(
            model_path=model_path,
            verbose=verbose,
            max_tokens=max_tokens,
            n_ctx=max_tokens,   # NOTE: Not tested, if output still got truncated, try to increase this.
        )

    
