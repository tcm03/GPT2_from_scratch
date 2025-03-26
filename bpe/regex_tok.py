import regex as re
from base import Tokenizer
import typing

class RegexTokenizer(Tokenizer):

    GPT4_SPLIT_PATTERN: typing.Final[str] = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

    def __init__(
        self, 
        log_file: typing.Optional[str] = None
    ) -> None:
        super().__init__(log_file=log_file)

    def train(
        self, 
        text: str, 
        vocab_size: int, 
        verbose: bool = False
    ) -> None:
        if verbose:
            log_text = ""
            for i in range(self.UTF8_VOCAB_SIZE):
                log_text += f'[{self.decode([i])}] {i}\n'
            self._log(log_text, mode='w')

        text_chunks = re.findall(self.GPT4_SPLIT_PATTERN, text)
        text_ids = []
        for chunk in text_chunks:
            ids = self._encode_utf8(chunk)
            text_ids.append(ids)

        num_merges = vocab_size - self.UTF8_VOCAB_SIZE
        for t in range(num_merges):

            pairs_count: typing.Dict[typing.Tuple[int, int], int] = {}
            for ids in text_ids:
                if len(ids) < 2:
                    continue
                for pair in zip(ids[:-1], ids[1:]):
                    pairs_count[pair] = pairs_count.get(pair, 0) + 1

            if len(pairs_count) == 0:
                break
            sorted_pairs = sorted(pairs_count.items(), key=lambda x: x[1], reverse=True)
            max_pair = sorted_pairs[0][0]
            idx = self.UTF8_VOCAB_SIZE + t
            if verbose:
                log_text = f'[{self.decode([max_pair[0]])}][{self.decode([max_pair[1]])}] -> [{self.decode([max_pair[0]])}{self.decode([max_pair[1]])}] {idx}\n'
                self._log(log_text, mode='a')
            for i, ids in enumerate(text_ids):
                text_ids[i] = self._merge(ids, max_pair, idx, verbose=verbose)

    

    