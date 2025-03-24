import typing as t

class BaseTokenizer:
    def __init__(self):
        pass

    def encode(self, input: str) -> list[int]:
        raise NotImplementedError("encode() is not implemented.")
    
    def encode(self, tokens: list[int]) -> str:
        raise NotImplementedError("decode() is not implemented.")