import sys
import unittest
import logging
from datasets import load_dataset, Dataset
from trainer.dataset_loader import create_random_message_stack_from_adjacent_records, create_message_stack, convert_message_stack_to_training_promots


class TestDatasetLoader(unittest.TestCase):

    def test_loading_dataset_from_hf(self):
        dataset = load_dataset("h-alice/chat-cooking-master-boy-100k", split="train")
        self.assertIsInstance(dataset, Dataset)
        self.dataset = dataset

    def test_create_message_stack(self):
        # NOTE: Actually this test doesn't have deterministic result. And should be checked by output.
        print(create_message_stack(self.dataset, batch_size=10, stack_size_min=2, stack_size_max=2, even_stack_size=True))


if __name__ == '__main__':
    unittest.main()