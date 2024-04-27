from typing import List
from transformers import AutoTokenizer, PreTrainedTokenizer, BatchEncoding

def tokenize(tokenizer: PreTrainedTokenizer, text: str, eos_token: bool=False, max_length: int=2048) -> BatchEncoding:
    """
    ## Tokenize Text
    Tokenize the text with the specified tokenizer.

    Parameters:
    - tokenizer: PreTrainedTokenizer, The tokenizer to use.
    - text: str, The text to tokenize.

    Returns:
    - List[str], The list of tokens.
    """

    token_list = tokenizer(
            text, # Prompt.
            truncation=True, # Truncate the text if it exceeds the maximum length.
            max_length=max_length, # Maximum length of the tokenized text.
            padding=False, # No padding.
        )
    
    if eos_token:
        # Append the EOS token. 
        # Note that it will append the token even if it exceeds the maximum length.
        token_list["input_ids"].append(tokenizer.eos_token_id) # type: ignore
        token_list["attention_mask"].append(1) # Set the attention mask to 1. # type: ignore

    return token_list



if __name__ == "__main__": # For test.
    tokenizer = AutoTokenizer.from_pretrained("/Users/h-alice/Desktop/gemma-2b")
    print(tokenize(tokenizer, "來去菜市場看歐郎甲追夠", eos_token=False)) # type: ignore