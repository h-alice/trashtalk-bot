"""
This module manages LLM connection and sessions.
"""

import abc

from typing import NamedTuple, List, Iterator

# We may not need LlamaCpp, therefore we make a try-except block.
try:
    # LLamaCpp is a C++ implementation of LLM, which can be run in local machine.
    from langchain_community.llms import LlamaCpp
    USE_LLAMACPP = True
except ImportError:
    USE_LLAMACPP = False

# We may not need HuggingfaceTextGenInference, therefore we make a try-except block.
try:
    # HuggingfaceTextGenInference is a handy wrapper for Huggingface's text generation API.
    from langchain_community.llms.huggingface_text_gen_inference import HuggingFaceTextGenInference
    USE_HUGGINGFACE_TEXT_GEN = True
except ImportError:
    USE_HUGGINGFACE_TEXT_GEN = False

from langchain_core.language_models.llms import LLM

from .llm_prompt_crafter import PromptCrafter
from .common import LlmNames, ModelProviders



# This is a named tuple for the LLM generation parameters.
# It's a way to pass parameters to the LLM generation function. And included some most common parameters.
#
# 
class LlmGenerationParameters(NamedTuple):
    """
    # LLM Generation Parameters
    This is a named tuple for the LLM generation parameters. And provides a way to pass parameters to the LLM generating function.

    Some mostly used parameters are included in this named tuple, with recommended default values.
    Parameters:
    - max_new_tokens: int, The upper limit of the number of tokens to generate.
    - top_k: int, The number of top tokens to sample from.
    - top_p: float, The cumulative probability of the top tokens to sample from. Lower value means more precise output.
    - temperature: float, The randomness of the output. Lower value means more precise output.
    - repetition_penalty: float, The penalty for repeating tokens in the output.
    """
    max_new_tokens: int
    top_k: int
    top_p: float
    temperature: float
    repetition_penalty: float
    @classmethod
    def new_generation_parameter(cls, max_new_tokens=1024, top_k=10, top_p=0.9, temperature=0.1, repetition_penalty=1.3) -> 'LlmGenerationParameters':
        return cls(max_new_tokens=max_new_tokens,
                   top_k=top_k, 
                   top_p=top_p, 
                   temperature=temperature, 
                   repetition_penalty=repetition_penalty)


class LlmConnector(abc.ABC):
    def __init__(self, model_name: str, model_provider: str):
        self.model_name = model_name
        self.model_provider = model_provider
        self.history: List[PromptCrafter] = []
        self.llm_instance: LLM = None

    def store_history(self, history: PromptCrafter):
        """
        ## Store History
        Store the history of the prompt crafter.
        """
        self.history.append(history)

    def llm_stream_result(self, prompt: str, rag_content :list=[], llm_parameter: LlmGenerationParameters=None) -> Iterator[str]:
        """
        ## Streaming Result from Language Model
        This method streams the result from the language model.

        The return is an iterator of strings, which return the result, one token at a time, until the completion of the generation.

        It's very useful if you want to display the result in a streaming fashion, like the chatbot is actually typing the response.

        Parameters:
        - llm: LLM, The language model instance.
        - prompt: str, The prompt to generate the response. It should be in the proper prompt format which the language model can understand.
        - rag_content: List[Document], The list of RAG documents, optional.
        - llm_parameter: LlmGenerationParameters, The generation parameters for the language model.
        """

        # LLM parameter.
        if not llm_parameter:
            llm_parameter = LlmGenerationParameters.new_generation_parameter()


        # Prompt crafting.
        prompt_crafter = PromptCrafter.new_prompt_crafter(self.model_name)

        # Set user input.
        crafted_prompt_to_llm = prompt_crafter.craft_prompt(prompt, rag_content)

        # Setup streamer.
        llm_streamer = self.llm_instance.stream(
            crafted_prompt_to_llm,
            top_k=llm_parameter.top_k,
            top_p=llm_parameter.top_p,
            temperature=llm_parameter.temperature,
            repeat_penalty=llm_parameter.repetition_penalty,
            max_tokens=llm_parameter.max_new_tokens,
            )
        
        # Stream the result.
        model_output = "" # Placeholder for the model output.
        
        for token in llm_streamer:
            model_output += token
            yield token
        
        # Finish the prompt.
        final_output = prompt_crafter.finish_prompt(model_output)
        self.store_history(prompt_crafter)

    def get_last_result(self) -> str:
        """
        ## Get Last Result
        Get the last result from the language model.
        """
        if not self.history:
            raise ValueError("No history found.") 
        
        return self.history[-1].get_model_response()

class LlmConnectorLlamacpp(LlmConnector):
    """
    ## LLM Connector for LLamaCpp
    Connect to local LLamaCpp model.

    Parameters:
    - model_name: str, The name of the model (e.g. gemma).
    - model_path: str, The path to the model.
    - verbose: bool, Whether to output verbose information. Should be false since it outputs REALLY a lot of information.
    - max_tokens: int, The maximum number of tokens to generate. Increase this if the output is truncated.
    """
    def __init__(self, model_name: str, model_path: str, verbose: bool = False, max_tokens: int = 2048):

        # Call the parent class constructor
        super().__init__(model_provider=ModelProviders.LLAMACPP, model_name=model_name)

        if not USE_LLAMACPP:
            raise ValueError("LLamaCpp is not available. Please install the required package.")
        
        # Ensure we only need to load the model once.
        self.llm_instance = LlamaCpp(
            model_path=model_path,
            verbose=verbose,
            max_tokens=max_tokens,
            n_ctx=max_tokens,   # NOTE: Not tested, if output still got truncated, try to increase this.
        )

    @classmethod
    def new_llm_connector(cls, model_name: str, model_path: str, verbose: bool = False, max_tokens: int = 2048) -> 'LlmConnectorLlamacpp':
        return cls(model_name=model_name, model_path=model_path, verbose=verbose, max_tokens=max_tokens)
    
class LlmConnectorHuggingface(LlmConnector):
    """
    ## LLM Connector for Huggingface
    Connect to Huggingface model.

    It may be a running docker container or a remote server.

    Parameters:
    - model_name: str, The name of the model (e.g. llama).
    - model_endpoint: str, The endpoint of the model.
    - max_tokens: int, The maximum number of tokens to generate. Increase this if the output is truncated.

    """
    def __init__(self, model_name: str, model_endpoint: str, max_tokens: int = 2048):

        # Call the parent class constructor
        super().__init__(model_provider=ModelProviders.HUGGINGFACE, model_name=model_name)

        if not USE_HUGGINGFACE_TEXT_GEN:
            raise ValueError("Huggingface text generation is not available. Please install the required package.")

        # Ensure we only need to load the model once.
        self.llm_instance = HuggingFaceTextGenInference(
            inference_server_url=model_endpoint,
            max_new_tokens=max_tokens,
        )    

    @classmethod
    def new_llm_connector(cls, model_name: str, model_endpoint: str, max_tokens: int = 2048) -> 'LlmConnectorHuggingface':
        return cls(model_name=model_name, model_endpoint=model_endpoint, max_tokens=max_tokens)

