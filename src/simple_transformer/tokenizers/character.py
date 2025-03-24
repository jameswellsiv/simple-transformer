from simple_transformer.tokenizers.base import BaseTokenizer


class CharacterTokenizer(BaseTokenizer):
    def __init__(self, text: str):
        """Character level tokenization

        Args:
            text (str): All of the text in a dataset
        """
        print("Initializing CharacterTokenizer...")
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)

        self.stoi = {c:i for i,c in enumerate(self.chars)}
        self.itos = {i:c for i,c in enumerate(self.chars)}


    def encode(self, input: str) -> list[int]:
        return [self.stoi[c] for c in input]
    
    def decode(self, tokens: list[int]) -> str:
        return "".join([self.itos[t] for t in tokens])
    

if __name__ == "__main__":
    text = open("/root/workspace/my-transformer/data/shakespeare.txt", "r", encoding="utf-8").read()
    tokenizer = CharacterTokenizer(text)
    print(tokenizer.vocab_size)
    input = "Tokenize this."
    tokens = tokenizer.encode(input)
    print(f"Encoded = {tokens}")

    decoded = tokenizer.decode(tokens)
    print(f"Decoded = {decoded}")