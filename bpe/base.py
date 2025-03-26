import typing
import abc

# train the vocabulary of tokenizer (with possibly different training set)
# translate back and forth between raw text and tokens (later, the LLM sees only the tokens)
class Tokenizer(abc.ABC):

    UTF8_VOCAB_SIZE: typing.Final[int] = 256

    def __init__(
        self,
        log_file: typing.Optional[str] = None
    ) -> None:
        self.vocab: typing.Dict[int, bytes] = {}
        for i in range(self.UTF8_VOCAB_SIZE):
            self.vocab[i] = bytes([i])
        self.merges: typing.Dict[typing.Tuple[int, int], int] = {}
        self.log_file: typing.Optional[str] = log_file

    @abc.abstractmethod
    def train(
        self, 
        text: str, 
        vocab_size: int, 
        verbose: bool = False
    ) -> None:
        pass

    def encode(
        self, 
        text: str
    ) -> typing.List[int]:
        ids = self._encode_utf8(text)
        for pair, id in self.merges.items():
            ids = self._merge(ids, pair, id)
        return ids

    def decode(
        self, 
        ids: typing.List[int], 
        errors: str = 'replace'
    ) -> str:
        tokens = b''.join([self.vocab[id] for id in ids])
        return tokens.decode('utf-8', errors=errors)

    def _encode_utf8(
        self, 
        text: str
    ) -> typing.List[int]:
        return list(map(int, text.encode('utf-8')))

    def _merge(
        self, 
        ids: typing.List[int], 
        pair: typing.Tuple[int, int], 
        idx: int, 
        verbose: bool = False
    ) -> typing.List[int]:
        self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
        self.merges[pair] = idx
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def _log(
        self, 
        text: str, 
        mode: str = 'w'
    ) -> None:
        if self.log_file is None:
            print(text)
        else:
            with open(self.log_file, mode, encoding='utf-8') as file:
                file.write(text)