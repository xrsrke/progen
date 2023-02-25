# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/04_dataset.ipynb.

# %% auto 0
__all__ = ['extract_property_tag', 'ProteinSequence', 'ProteinDataset']

# %% ../nbs/04_dataset.ipynb 4
import re
from typing import Tuple, Callable, TypedDict, List, Dict

import torch
from torch.utils.data import Dataset

# %% ../nbs/04_dataset.ipynb 5
def extract_property_tag(name: str) -> Callable:
    def inner(sequence):
        pattern = f"{name}=(.+?) "
        match = re.search(pattern, sequence)
        if match:
            return match.group(1)
        else:
            return None
    return inner

# %% ../nbs/04_dataset.ipynb 6
class ProteinSequence(TypedDict):
    id: str
    seq: str
    desc: str

# %% ../nbs/04_dataset.ipynb 7
class ProteinDataset(Dataset):
    def __init__(self, data: List[ProteinSequence], tokenizer: Callable, tag_extractor: Callable):
        xs = []
        ys = []
        
        for item in data:
            xs.append(tag_extractor(item["desc"]))
            ys.append(item["seq"])
        
        encoded_ys = tokenizer.encode_batch(ys)
        encoded_ys = [torch.tensor(e.ids) for e in encoded_ys]
        
        self.xs: List[str] = xs
        self.ys: List[str] = encoded_ys

    def __len__(self) -> int:
        return len(self.xs)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        return self.xs[idx], self.ys[idx]
