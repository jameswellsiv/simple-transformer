import torch
from torch.utils.data import Dataset
from simple_transformer.tokenizers.base import BaseTokenizer

class ShakespeareDataset(Dataset):
    def __init__(self, text: str, tokenizer: BaseTokenizer, block_size: int = 64):
        self.text = text
        self.tokenizer = tokenizer
        self.block_size = block_size

        self.data = self.tokenizer.encode(text)

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]

        assert len(x) == self.block_size
        assert len(y) == self.block_size
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.data) - self.block_size
    

if __name__ == "__main__":
    from simple_transformer.tokenizers.character import CharacterTokenizer

    text = open("/root/workspace/my-transformer/data/shakespeare.txt", "r", encoding="utf-8").read()
    tokenizer = CharacterTokenizer(text)

    dataset = ShakespeareDataset(text, tokenizer)

    print(next(iter(dataset)))