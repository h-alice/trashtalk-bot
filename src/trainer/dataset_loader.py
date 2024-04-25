import random
from typing import Tuple, List
from core import PromptCrafter

def craft_training_prompt(user_input: str, expected_model_output: str) -> str:
    prompt_crafter = PromptCrafter.new_prompt_crafter("gemma")
    _ = prompt_crafter.craft_prompt(user_input, [])
    _ = prompt_crafter.finish_prompt(expected_model_output)
    return str(prompt_crafter)

def create_random_message_stack_from_adjacent_records(dataset, stack_size) -> List[str]:
    min_message_idx = 0
    max_message_idx = len(dataset['message'])

    # Randomly select a message index.
    message_idx = random.randint(min_message_idx, max_message_idx - stack_size + 1)
    return dataset['message'][message_idx:message_idx + stack_size]

if __name__ == "__main__": # For test.
    user_input = "Hello, how are you?"
    model_output = "I am fine, thank you."
    print(craft_training_prompt(user_input, model_output))

