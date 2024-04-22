import sys
import unittest
import logging
from core.llm_prompt_crafter import PromptCrafter

GEMMA_PROMPT = """<start_of_turn>user
{full_user_prompt}<end_of_turn>
<start_of_turn>model
{model}"""

GEMMA_PROMPT_FINISHED = "<end_of_turn>\n"
class TestPromptCrafter(unittest.TestCase):

    def test_prompt_crafter_gemma(self):

        # Stage 1: Test the creation of PromptCrafter.
        prompt_crafter = PromptCrafter.new_prompt_crafter("gemma")
        self.assertEqual(str(prompt_crafter), "")

        # Stage 2: Test the crafting of prompt.
        user_input = "Hello, how are you?"
        rag_content = []
        crafted_prompt_to_llm = prompt_crafter.craft_prompt(user_input, rag_content)

        # The placeholder should be kept in the internal process.
        self.assertEqual(prompt_crafter.current_prompt, GEMMA_PROMPT.format(full_user_prompt=user_input, model="{model}"))

        # The placeholder should be replaced in the actual prompt to LLM.
        self.assertEqual(crafted_prompt_to_llm, GEMMA_PROMPT.format(full_user_prompt=user_input, model=""))

        # Stage 3: Finish the prompt.
        dummy_model_output = "I am fine, thank you."
        final_output = prompt_crafter.finish_prompt(dummy_model_output)

        # The placeholder should be replaced in the final output.
        self.assertEqual(final_output, GEMMA_PROMPT.format(full_user_prompt=user_input, model=dummy_model_output) + GEMMA_PROMPT_FINISHED)

        print(prompt_crafter)




if __name__ == '__main__':
    unittest.main()