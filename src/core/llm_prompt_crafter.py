"""
LLM Connector
"""
from typing import NamedTuple, List, Iterator
from langchain_core.documents.base import Document
from langchain_core.language_models.llms import LLM

from .common import LlmNames


# Some prompt templates.
PROMPT_FORMAT = {
    LlmNames.LLAMA: {
        "prompt": "<s>[INST] <<SYS>>\n{sys}\n<</SYS>>\n{full_user_prompt} [/INST]\n{model}\n",
        "after_generation": ""
    },
    LlmNames.GEMMA: {
        "prompt": "<start_of_turn>user\n{full_user_prompt}<end_of_turn>\n<start_of_turn>model\n{model}",
        "after_generation": "<end_of_turn>\n"
    },
}

SAMPLE_SYS_PROMPT = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.

Answer the question in Markdown format for readability, use bullet points if possible.
"""

RAG_STEM = """
If there is nothing in the context relevant to the question at hand, just resuse to answer it. Don't try to make up an answer.

Anything between the following `context` html blocks is retrieved from a knowledge bank, not part of the conversation with the user. 

"""

class PromptCrafter:
    def __init__(self, model_type="gemma"):
        self.model_type = model_type
        self.flag_is_finished = False # The flag to check if the prompt is finished.
        self.user_input = ""
        self.rag_content = ""  # Note that this is the flattened content of the RAG documents.
        self.current_prompt = "" # The current prompt.
        try:
            self.prompt_template = PROMPT_FORMAT[model_type.lower()] # Apply `lower` to make it case-insensitive.
        except KeyError:
            raise ValueError(f"Model type {model_type} is not supported.")
    
    @classmethod
    def new_prompt_crafter(cls, model_type="gemma") -> 'PromptCrafter':
        return cls(model_type=model_type)

    def __str__(self) -> str:
        return self.current_prompt
        
    def is_finished(self):
        # Check if there's still `{model}` in current prompt.
        try:
            assert "{model}" not in self.current_prompt
        except AssertionError:
            raise ValueError("WARNING: This is a potential misfunction of the prompt crafter. The `{model}` placeholder is not replaced.")
        return self.flag_is_finished

    def craft_prompt(self, user_input, rag_content: List[Document]=[]):
        """
        ## Craft Prompt with User Input
        Crafts the prompt with the user input.

        This method supports crafting the prompt with RAG content.

        Parameters:
        - user_input: str, The user input.
        - rag_content: List[Document], The list of RAG documents, optional.

        """

        # Check if finished flag is on.
        if self.flag_is_finished:
            raise ValueError("The prompt is already finished and should not be changed anymore.")
        
        self.user_input = user_input # Store the user input.

        # Prompt crafting.
        rag_prompt = "" # Placeholder for RAG content.
        # The `full_user` is constructed with "RAG" part and actual user input.
        if rag_content:
            rag_documents = "\n".join([x.page_content for x in rag_content]) # Join all the documents.
            rag_prompt = RAG_STEM + f"<context>\n{rag_documents}\n</context>\n"

        # Complete the prompt with the user input.
        full_user_prompt = rag_prompt + self.user_input
        self.current_prompt = self.prompt_template["prompt"].format(full_user_prompt=full_user_prompt, model="{model}") # The `{model}` placeholder is preserved.

        return self.current_prompt.format(model="") # Return the prompt with the model placeholder replaced by empty string.
    
    def finish_prompt(self, model_response) -> str:
        """
        ## Finish Prompt with Model Response
        This method finishes the prompt with the model response.

        From now on, the prompt is considered finished. It shouldn't be changed anymore and can be store in the chat history database for future reference.
        """
        self.current_prompt = self.current_prompt.format(model=model_response) # Replace the `{model}` placeholder with the model response.
        self.current_prompt += self.prompt_template["after_generation"] # Add the after generation part.
        self.flag_is_finished = True # Set the flag to finished.
        return self.current_prompt

    
"""
def craft_prompt(user_input, model_type="gemma", rag_content: List[Document]=[], keep_placeholder=False):

    # Get the prompt format.
    try:
        prompt_template = PROMPT_TEMPLATE[model_type]
    except KeyError:
        raise ValueError(f"Model type {model_type} is not supported.")

    prompt = PromptTemplate(
        input_variables=["full_user"],
        template=prompt_template["prompt"],
        partial_variables={"model": ""},
        )
    # Prompt crafting.
    # user_prompt = prompt.partial(sys=SAMPLE_SYS_PROMPT)  # TODO: User defined system prompt.
    user_prompt = prompt

    # if rag_content:
    if False:
        rag_documents = "\n".join([x.page_content for x in rag_content])
        rag_prompt = RAG_STEM + f"<context>\n{rag_documents}\n</context>\n"
        user_prompt = user_prompt.partial(rag=rag_prompt)
        
    if keep_placeholder:
        prompt = user_prompt.partial(user=user_input) 
    else:
        prompt = user_prompt.format(user=user_input)

    return prompt

def craft_result_with_prompt(user_input, model_response):
    return craft_prompt(user_input, keep_placeholder=True).format(model=model_response)
    """