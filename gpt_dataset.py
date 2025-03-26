import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):

    def __init__(
        self,
        text: str,
        tokenizer: tiktoken.core.Encoding,
        max_length: int,
        stride: int
    ) -> None:
        self.tokenizer = tokenizer
        text_ids = self.tokenizer.encode(text, allowed_special = {'|<endoftext>|'})
        print(f"Length of tokenized texts: {len(text_ids)}")
        self.inputs_ids = []
        self.targets_ids = []
        for i in range(0, len(text_ids) - max_length, stride):
            input_ids = text_ids[i: i+max_length]
            target_ids = text_ids[i+1: i+max_length+1]
            self.inputs_ids.append(torch.tensor(input_ids))
            self.targets_ids.append(torch.tensor(target_ids))

    def __len__(self):
        return len(self.inputs_ids)

    def __getitem__(self, i: int):
        return self.inputs_ids[i], self.targets_ids[i]
  
def create_dataloader_v1(
    texts: str,
    batch_size: int,
    max_length: int,
    stride: int,
    drop_last: bool = True,
    shuffle: bool = True,
    num_workers: int = 0
):
    tokenizer = tiktoken.get_encoding('gpt2')
    gpt_dataset = GPTDatasetV1(texts, tokenizer, max_length, stride)
    print(f"Number of samples: {len(gpt_dataset)}")
    dataloader = DataLoader(gpt_dataset, batch_size, shuffle, drop_last = drop_last, num_workers = num_workers)
    return dataloader