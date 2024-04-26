from transformers import AutoTokenizer, PreTrainedTokenizer

def tokenize(tokenizer: PreTrainedTokenizer, text: str):
    """
    ## Tokenize Text
    Tokenize the text with the specified tokenizer.

    Parameters:
    - tokenizer: PreTrainedTokenizer, The tokenizer to use.
    - text: str, The text to tokenize.

    Returns:
    - List[str], The list of tokens.
    """

    return tokenizer.tokenize(text)

if __name__ == "__main__": # For test.
    tokenizer = AutoTokenizer.from_pretrained("/Users/h-alice/Desktop/gemma-2b")
    print(tokenize(tokenizer, "來去菜市場看歐郎甲追夠"))