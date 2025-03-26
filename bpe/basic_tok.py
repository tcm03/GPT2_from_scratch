from base import Tokenizer
import typing

class BasicTokenizer(Tokenizer):

    def __init__(
        self, 
        log_file: typing.Optional[str] = None
    ):
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

        ids = self._encode_utf8(text)
        num_merges = vocab_size - self.UTF8_VOCAB_SIZE
        for t in range(num_merges):
            if len(ids) < 2:
                break
            pairs_count: typing.Dict[typing.Tuple[int, int], int] = {}
            for pair in zip(ids[:-1], ids[1:]):
                pairs_count[pair] = pairs_count.get(pair, 0) + 1
            sorted_pairs = sorted(pairs_count.items(), key=lambda x: x[1], reverse=True)
            max_pair = sorted_pairs[0][0]
            idx = self.UTF8_VOCAB_SIZE + t
            if verbose:
                log_text = f'[{self.decode([max_pair[0]])}][{self.decode([max_pair[1]])}] -> [{self.decode([max_pair[0]])}{self.decode([max_pair[1]])}] {idx}\n'
                self._log(log_text, mode='a')
            ids = self._merge(ids, max_pair, idx, verbose=verbose)

    

        
    

    