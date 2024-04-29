import random
from typing import Tuple, List
from core import PromptCrafter
from datasets import load_dataset, Dataset
import logging

def craft_training_prompt(user_input: str, expected_model_output: str) -> str:
    prompt_crafter = PromptCrafter.new_prompt_crafter("gemma")
    _ = prompt_crafter.craft_prompt(user_input, [])
    _ = prompt_crafter.finish_prompt(expected_model_output)
    return str(prompt_crafter)

def create_random_message_stack_from_adjacent_records(dataset: Dataset, stack_size: int) -> List[str]:
    """
    ## Create Random Message Stack from Adjacent Records
    Create a message stack from the dataset by randomly selecting a message index.

    The message stack is created by first selecting a random message index, then selecting the next `stack_size - 1` messages.

    Parameters:
    - dataset: Dataset, The dataset.
    - stack_size: int, The size of the stack.
    """
    min_message_idx = 0
    max_message_idx = len(dataset['message'])

    # Randomly select a message index.
    message_idx = random.randint(min_message_idx, max_message_idx - stack_size + 1)
    return dataset['message'][message_idx:message_idx + stack_size]

def create_message_stack(dataset: Dataset, batch_size: int, stack_size_min: int=2, stack_size_max: int=2, even_stack_size: bool=True) -> List[List[str]]:
    """
    ## Create Message Stack Batch
    Create a message stack batch from the dataset.

    The size of stack will be randomly selected between `stack_size_min` and `stack_size_max`.

    Since the message is tend to be paired, the stack size is recommended to be even number.

    Parameters:
    - dataset: Dataset, The dataset.
    - batch_size: int, The batch size.
    - stack_size_min: int, The minimum size of the stack.
    - stack_size_max: int, The maximum size of the stack.
    - even_stack_size: bool, Whether the stack size should be even. Default is True.
    """
    message_stacks = []
    for _ in range(batch_size):
        stack_size = random.randint(stack_size_min, stack_size_max)
        if even_stack_size and stack_size % 2 != 0:
            stack_size += 1
        message_stack = create_random_message_stack_from_adjacent_records(dataset, stack_size)
        message_stacks.append(message_stack)
    return message_stacks

def convert_message_stack_to_training_promots(message_stacks: List[List[str]]) -> List[str]:
    """
    ## Convert Message Stack to Training Prompts
    Convert the message stack to training prompts.

    Parameters:
    - message_stacks: List[List[str]], The message stacks.
    """
    training_prompts = []
    for message_stack in message_stacks:
        temp_stacks = []

        # Check if the message stack is even.
        if len(message_stack) % 2 != 0:
            logging.warning("The message stack is not even. The last message will be ignored.")

        # Iterate over the message stack.
        for i in range(0, len(message_stack), 2):
            temp_stacks.append(craft_training_prompt(message_stack[i], message_stack[i + 1]))

        # Join the message stack to a single string.
        training_prompts.append("\n".join(temp_stacks))
    return training_prompts

