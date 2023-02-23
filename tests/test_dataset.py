import torch

from progen.dataset import ProteinDataset, extract_property_tag
from progen.tokenizer import create_tokenizer
from progen.utils import fasta2dict


def test_create_a_protein_dataset(default_config):
    SEQ_LEN = 20

    tokenizer_path = default_config["tokenizer"]["path"]
    dataset_path = default_config["dataset"]["path"]

    tokenizer = create_tokenizer(path=tokenizer_path)
    extractor = extract_property_tag("OS")
    data = fasta2dict(path=dataset_path)[:SEQ_LEN]

    dataset = ProteinDataset(
        data=data,
        tokenizer=tokenizer,
        tag_extractor=extractor
    )

    assert len(dataset) == SEQ_LEN
    assert isinstance(dataset[0][1], torch.Tensor)